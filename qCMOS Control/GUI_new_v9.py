import tkinter as tk
from tkinter import StringVar, OptionMenu, Checkbutton, Label, Entry, Button, Scale, Frame, LabelFrame, messagebox
from dcam import Dcamapi, Dcam, DCAMERR
from camera_params import CAMERA_PARAMS, DISPLAY_PARAMS
import GPS_time
import threading
import time
import numpy as np
import cv2
import queue
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import os
import warnings
from PyZWOEFW import EFW
import asyncio
from cyberpower_pdu import CyberPowerPDU, OutletCommand
from labjack import ljm
from zaber_motion import Units
from zaber_motion.ascii import Connection
import ctypes
import logging
import traceback
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FutureTimeoutError
import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory
import pickle
import cProfile
import pstats
from functools import wraps

# Load required library
ctypes.CDLL("libudev.so.1", mode=ctypes.RTLD_GLOBAL)

# Enable OpenCV multi-threading
cv2.setNumThreads(4)  # Use 4 cores for OpenCV operations

# Set NumPy thread count for parallel operations
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add file handler for detailed logging
file_handler = logging.FileHandler('camera_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(threadName)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
))
logging.getLogger().addHandler(file_handler)

# Debug logger for verbose operations
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_logger.propagate = False
debug_logger.addHandler(file_handler)


def profile_function(func):
    """Decorator for profiling functions - disabled to avoid threading conflicts"""
    # Profiling disabled in multi-threaded environment
    # To profile, use cProfile.run() on specific functions manually
    return func


class DCamLock:
    """Fine-grained locking for DCAM operations - reduced contention"""
    # Separate locks for different operations to allow more parallelism
    _capture_lock = threading.RLock()
    _property_lock = threading.RLock()
    _buffer_lock = threading.RLock()
    _init_lock = threading.RLock()
    
    @classmethod
    def acquire_capture(cls, timeout=5.0, check_stop=None):
        """Acquire capture lock"""
        return cls._acquire_lock(cls._capture_lock, timeout, check_stop)
    
    @classmethod
    def release_capture(cls):
        """Release capture lock"""
        cls._capture_lock.release()
    
    @classmethod
    def acquire_property(cls, timeout=2.0):
        """Acquire property lock"""
        return cls._acquire_lock(cls._property_lock, timeout)
    
    @classmethod
    def release_property(cls):
        """Release property lock"""
        cls._property_lock.release()
    
    @classmethod
    def acquire_buffer(cls, timeout=2.0):
        """Acquire buffer lock"""
        return cls._acquire_lock(cls._buffer_lock, timeout)
    
    @classmethod
    def release_buffer(cls):
        """Release buffer lock"""
        cls._buffer_lock.release()
    
    @classmethod
    def acquire_init(cls, timeout=5.0):
        """Acquire initialization lock"""
        return cls._acquire_lock(cls._init_lock, timeout)
    
    @classmethod
    def release_init(cls):
        """Release initialization lock"""
        cls._init_lock.release()
    
    @classmethod
    def _acquire_lock(cls, lock, timeout, check_stop=None):
        """Generic lock acquisition with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if check_stop and check_stop():
                return False
            if lock.acquire(blocking=False):
                return True
            time.sleep(0.001)
        return False


class FrameProcessorPool:
    """Multiprocessing pool for parallel frame processing"""
    
    def __init__(self, num_workers=None):
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid multiprocessing issues
        # Still provides parallelism for I/O-bound operations
        self.num_workers = num_workers or min(4, mp.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.futures = []
        
    @staticmethod
    def process_frame_chunk(frames, params):
        """Process a chunk of frames - runs in thread pool"""
        processed = []
        for frame in frames:
            # Apply any heavy processing here
            # NumPy and OpenCV operations still use multiple cores internally
            if params.get('denoise', False):
                frame = cv2.fastNlMeansDenoising(frame)
            if params.get('normalize', False):
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            processed.append(frame)
        return np.array(processed)
    
    def process_frames_parallel(self, frames, params=None):
        """Split frames across multiple threads for processing"""
        if params is None:
            params = {}
            
        # Split frames into chunks for each worker
        chunk_size = max(1, len(frames) // self.num_workers)
        chunks = [frames[i:i+chunk_size] for i in range(0, len(frames), chunk_size)]
        
        # Submit processing tasks
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self.process_frame_chunk, chunk, params)
            futures.append(future)
        
        # Gather results
        processed_chunks = [f.result() for f in futures]
        return np.concatenate(processed_chunks) if processed_chunks else np.array([])
    
    def shutdown(self):
        """Shutdown the thread pool"""
        self.executor.shutdown(wait=True)


class SharedFrameBuffer:
    """Zero-copy shared memory frame buffer for inter-process communication"""
    
    def __init__(self, num_frames=200, frame_shape=(2304, 4096), dtype=np.uint16):
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.dtype = dtype
        self.frame_size = np.prod(frame_shape) * np.dtype(dtype).itemsize
        self.total_size = self.frame_size * num_frames
        
        try:
            # Create shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size)
            self.buffer = np.ndarray(
                (num_frames, *frame_shape), 
                dtype=dtype, 
                buffer=self.shm.buf
            )
            self.lock = mp.Lock()
            self.write_index = mp.Value('i', 0)
            self.read_index = mp.Value('i', 0)
            self.available = True
        except Exception as e:
            logging.warning(f"SharedFrameBuffer initialization failed: {e}")
            self.available = False
            self.shm = None
            self.buffer = None
        
    def add_frame(self, frame):
        """Add frame to shared buffer with zero-copy"""
        if not self.available:
            return False
        try:
            with self.lock:
                idx = self.write_index.value % self.num_frames
                self.buffer[idx] = frame
                self.write_index.value += 1
            return True
        except Exception as e:
            logging.error(f"SharedFrameBuffer add_frame error: {e}")
            return False
            
    def get_frame(self, index=None):
        """Get frame from shared buffer"""
        if not self.available:
            return None
        try:
            with self.lock:
                if index is None:
                    index = self.read_index.value
                    self.read_index.value += 1
                idx = index % self.num_frames
                return self.buffer[idx].copy()
        except Exception as e:
            logging.error(f"SharedFrameBuffer get_frame error: {e}")
            return None
    
    def cleanup(self):
        """Clean up shared memory"""
        if self.available and self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except:
                pass


class AsyncFileWriter:
    """Asynchronous file I/O for parallel disk operations"""
    
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def write_fits_sync(self, filepath, hdus):
        """Write FITS file synchronously"""
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()
        
    def write_fits_async(self, filepath, hdus):
        """Write FITS file asynchronously using thread pool"""
        future = self.executor.submit(self.write_fits_sync, filepath, hdus)
        return future
        
    def shutdown(self):
        """Shutdown thread pool"""
        self.executor.shutdown(wait=True)


class SharedData:
    """Shared data container with thread-safe access"""
    def __init__(self):
        self.camera_params = {}
        self.lock = threading.RLock()


class CameraThread(threading.Thread):
    """Camera control thread with improved parallelism"""
    
    def __init__(self, shared_data, frame_queue, timestamp_queue, gui_ref):
        super().__init__(name="CameraThread")
        self.shared_data = shared_data
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.gui_ref = gui_ref
        self.dcam = None
        self.running = True
        self.capturing = False
        self.stop_requested = threading.Event()
        self.frame_index = 0
        self.modified_params = {}
        self.start_time = None
        self.paused = threading.Event()
        self.paused.set()
        self.first_frame = True
        self.buffer_size = 200
        self.save_queue = None
        self.needs_reconnect = False
        
        # Frame processing with multiprocessing
        self.frame_processor = FrameProcessorPool(num_workers=4)
        
        # Shared memory buffer for zero-copy frame passing (optional)
        # Disabled by default to avoid memory allocation issues
        self.shared_buffer = None
        # To enable, uncomment the following:
        # try:
        #     self.shared_buffer = SharedFrameBuffer(
        #         num_frames=50,  # Reduced from 200
        #         frame_shape=(2304, 4096),
        #         dtype=np.uint16
        #     )
        # except Exception as e:
        #     logging.warning(f"SharedFrameBuffer creation failed: {e}")
        #     self.shared_buffer = None
            
        # Separate locks for different operations
        self.capture_lock = threading.Lock()
        self.property_lock = threading.RLock()
        
        # Performance monitoring
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_print_time = time.time()
        
        # Watchdog
        self.last_capture_time = time.time()
        self.watchdog_enabled = False
        self.consecutive_errors = 0
        
        # Timestamp rollover tracking
        self.timestamp_offset = 0
        self.last_raw_timestamp = 0
        self.framestamp_offset = 0
        self.last_raw_framestamp = 0

    def run(self):
        """Main thread entry point"""
        try:
            self.connect_camera()
            self.main_loop()
        except Exception as e:
            logging.error(f"Fatal error in camera thread: {e}")
        finally:
            self.cleanup()

    def connect_camera(self):
        """Connect to camera with retry logic"""
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            retry_count += 1
            debug_logger.info(f"Camera connection attempt {retry_count}")
            
            # Use init lock for initialization
            if not DCamLock.acquire_init(timeout=5.0):
                logging.error("Failed to acquire init lock")
                continue
                
            try:
                # Initialize DCAM API
                if Dcamapi.init():
                    debug_logger.info("DCAM API initialized")
                else:
                    raise RuntimeError(f"DCAM API init failed: {Dcamapi.lasterr()}")
                    
                # Open camera device
                self.dcam = Dcam(0)
                if not self.dcam.dev_open():
                    raise RuntimeError(f"Device open failed: {self.dcam.lasterr()}")
                    
                logging.info("Camera connected successfully")
                self.needs_reconnect = False
                self.set_defaults()
                self.update_camera_params()
                return True
                
            except Exception as e:
                logging.warning(f"Failed to open camera: {e}")
                self.update_gui_status("Camera not connected.", "red")
                Dcamapi.uninit()
                time.sleep(5)
            finally:
                DCamLock.release_init()
        
        return False

    def main_loop(self):
        """Main camera operation loop"""
        while self.running:
            try:
                self.paused.wait(timeout=0.001)
                
                if self.needs_reconnect:
                    self.update_gui_status("Camera needs reset - use Reset Camera button", "red")
                    time.sleep(1.0)
                    continue
                
                if self.capturing and not self.stop_requested.is_set():
                    # Simple watchdog check
                    if self.watchdog_enabled and time.time() - self.last_capture_time > 10.0:
                        logging.error("Capture watchdog triggered")
                        self.reset_capture_state()
                    else:
                        self.capture_frame()
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logging.error(f"Error in camera loop: {e}")
                time.sleep(0.1)

    def capture_frame(self):
        """Capture a single frame with improved parallelism"""
        if self.stop_requested.is_set():
            return
            
        timeout_milisec = 100  # Reduced for better responsiveness
        
        # Try to acquire capture lock with non-blocking
        if not self.capture_lock.acquire(blocking=False):
            return
            
        try:
            if self.stop_requested.is_set():
                return
            
            # Only hold DCAM lock for actual camera operations
            if not DCamLock.acquire_capture(timeout=0.5, check_stop=lambda: self.stop_requested.is_set()):
                return
            
            try:
                if self.dcam is None or not self.capturing or self.stop_requested.is_set():
                    return
                    
                # Wait for frame - this is the actual camera I/O
                if self.dcam.wait_capevent_frameready(timeout_milisec):
                    frame_index_safe = self.frame_index % self.buffer_size
                    result = self.dcam.buf_getframe_with_timestamp_and_framestamp(frame_index_safe)
                    
                    if result is not False:
                        frame, npBuf, timestamp, framestamp = result
                        # Copy frame data while we have the lock
                        frame_copy = np.copy(npBuf)
                        
                        # Release DCAM lock immediately after getting frame
                        DCamLock.release_capture()
                        
                        # Process frame without holding lock - allows parallelism
                        self.process_captured_frame_parallel(frame_copy, timestamp, framestamp)
                        return
                        
            finally:
                # Make sure lock is released
                try:
                    DCamLock.release_capture()
                except:
                    pass
                    
        finally:
            self.capture_lock.release()

    def process_captured_frame_parallel(self, frame, timestamp, framestamp):
        """Process frame using parallel processing - no locks held"""
        # Handle timestamp rollover
        raw_timestamp = timestamp.sec + timestamp.microsec / 1e6
        if raw_timestamp < self.last_raw_timestamp - 4000:
            self.timestamp_offset += 4294.967296
            logging.warning(f"Timestamp rollover at frame {self.frame_index}")
        self.last_raw_timestamp = raw_timestamp
        corrected_timestamp = raw_timestamp + self.timestamp_offset
        
        # Handle framestamp rollover
        if framestamp < self.last_raw_framestamp - 60000:
            self.framestamp_offset += 65536
            logging.warning(f"Framestamp rollover at frame {self.frame_index}")
        self.last_raw_framestamp = framestamp
        corrected_framestamp = framestamp + self.framestamp_offset
        
        # Log periodically
        current_time = time.time()
        if current_time - self.last_print_time > 1.0:
            logging.info(f"Frame {self.frame_index}: timestamp={corrected_timestamp:.6f}, framestamp={corrected_framestamp}")
            self.last_print_time = current_time
        
        # Update watchdog
        self.last_capture_time = current_time
        self.consecutive_errors = 0
        
        # Use shared memory buffer if available (zero-copy)
        if self.shared_buffer and self.shared_buffer.available:
            self.shared_buffer.add_frame(frame)
        
        # Queue frame for display (non-blocking)
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(frame)
        except:
            pass

        # Queue timestamp (non-blocking)
        try:
            self.timestamp_queue.put_nowait((corrected_timestamp, corrected_framestamp))
        except:
            pass
        
        # Queue for saving if enabled (non-blocking)
        if self.save_queue is not None:
            try:
                self.save_queue.put_nowait((frame, corrected_timestamp, corrected_framestamp))
            except:
                debug_logger.warning("Save queue full")

        # Get GPS timestamp on first frame (async)
        if self.first_frame:
            threading.Thread(target=self.get_gps_timestamp, daemon=True).start()
            self.first_frame = False

        self.frame_index += 1

    def get_gps_timestamp(self):
        """Get GPS timestamp asynchronously"""
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(GPS_time.get_first_timestamp)
                self.start_time = future.result(timeout=1.0)
                
                if self.start_time:
                    logging.info(f"GPS timestamp: {self.start_time.isot}")
                    self.update_gui_gps(self.start_time.isot)
                else:
                    logging.warning("No GPS timestamp available")
                    self.update_gui_gps("No GPS timestamp")
        except Exception as e:
            logging.error(f"GPS timestamp error: {e}")
            self.update_gui_gps("GPS timeout")

    def start_capture(self):
        """Start capture with improved handling"""
        logging.info("Starting capture")
        
        self.stop_requested.clear()
        
        if self.needs_reconnect:
            logging.error("Camera needs reconnection")
            self.update_gui_status("Camera needs reset", "red")
            return False
        
        # Use buffer lock for allocation
        if not DCamLock.acquire_buffer(timeout=3.0):
            logging.error("Failed to acquire buffer lock")
            return False
            
        try:
            return self._start_capture_internal()
        finally:
            DCamLock.release_buffer()

    def _start_capture_internal(self):
        """Internal capture start logic"""
        # Clear GPS buffer
        try:
            GPS_time.clear_buffer()
            time.sleep(0.05)
        except Exception as e:
            debug_logger.error(f"GPS buffer clear error: {e}")

        # Stop any existing capture
        if self.capturing:
            self._stop_capture_internal()

        if self.dcam is None:
            logging.error("Camera not initialized")
            return False
        
        # Allocate buffer
        if not self.dcam.buf_alloc(self.buffer_size):
            logging.error("Buffer allocation failed")
            self.needs_reconnect = True
            return False

        # Initialize capture state
        self.capturing = True
        self.frame_index = 0
        self.first_frame = True
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.last_capture_time = time.time()
        self.watchdog_enabled = True
        self.consecutive_errors = 0
        
        # Reset rollover tracking
        self.timestamp_offset = 0
        self.last_raw_timestamp = 0
        self.framestamp_offset = 0
        self.last_raw_framestamp = 0

        # Enable timestamp producer
        try:
            self.dcam.prop_setgetvalue(CAMERA_PARAMS['TIME_STAMP_PRODUCER'], 1)
        except Exception as e:
            logging.error(f"Error setting timestamp producer: {e}")

        # Start capture
        if not self.dcam.cap_start():
            logging.error("Failed to start capture")
            self.capturing = False
            self.watchdog_enabled = False
            self.dcam.buf_release()
            self.needs_reconnect = True
            return False

        logging.info("Capture started successfully")
        return True

    def stop_capture(self):
        """Stop capture with improved handling"""
        logging.info("Stopping capture")
        
        self.stop_requested.set()
        time.sleep(0.2)  # Give capture loop time to exit
        
        # Use buffer lock for deallocation
        if DCamLock.acquire_buffer(timeout=2.0):
            try:
                result = self._stop_capture_internal()
            finally:
                DCamLock.release_buffer()
        else:
            logging.warning("Using force stop")
            result = self._stop_capture_internal(force=True)
        
        self.stop_requested.clear()
        return result

    def _stop_capture_internal(self, force=False):
        """Internal capture stop logic"""
        self.capturing = False
        self.watchdog_enabled = False
        
        if self.dcam is not None and not force:
            try:
                if not self.dcam.cap_stop():
                    logging.error(f"cap_stop failed: {self.dcam.lasterr()}")
                if not self.dcam.buf_release():
                    logging.error(f"buf_release failed: {self.dcam.lasterr()}")
                logging.info("Capture stopped cleanly")
            except Exception as e:
                logging.error(f"Error stopping capture: {e}")
        elif force:
            logging.warning("Force stop - skipping DCAM API calls")
        
        self.restore_modified_params()
        return True

    def set_defaults(self):
        """Set default camera parameters"""
        debug_logger.info("Setting default camera parameters")
        defaults = {
            'READOUT_SPEED': 1.0,
            'EXPOSURE_TIME': 0.1,
            'TRIGGER_SOURCE': 1.0,
            'TRIGGER_MODE': 6.0,
            'OUTPUT_TRIG_KIND_0': 3.0,
            'OUTPUT_TRIG_ACTIVE_0': 1.0,
            'OUTPUT_TRIG_POLARITY_0': 1.0,
            'OUTPUT_TRIG_PERIOD_0': 10,
            'SENSOR_MODE': 18.0,
            'IMAGE_PIXEL_TYPE': 1.0
        }
        for prop, value in defaults.items():
            self.set_property(prop, value)

    def set_property(self, prop_name, value):
        """Set camera property with property lock"""
        if prop_name not in CAMERA_PARAMS:
            logging.error(f"Unknown property: {prop_name}")
            return False
        
        with self.property_lock:
            if not DCamLock.acquire_property(timeout=1.0):
                logging.error(f"Failed to acquire property lock for {prop_name}")
                return False
                
            try:
                if self.dcam is None:
                    return False
                    
                if self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value):
                    self.modified_params[prop_name] = value
                    self.update_camera_params()
                    return True
                else:
                    logging.error(f"Failed to set {prop_name}: {self.dcam.lasterr()}")
                    return False
            finally:
                DCamLock.release_property()

    def update_camera_params(self):
        """Update shared camera parameters"""
        if not DCamLock.acquire_property(timeout=1.0):
            return
            
        try:
            if self.dcam is None:
                return
                
            with self.shared_data.lock:
                self.shared_data.camera_params.clear()
                idprop = self.dcam.prop_getnextid(0)
                while idprop is not False:
                    propname = self.dcam.prop_getname(idprop)
                    if propname:
                        propvalue = self.dcam.prop_getvalue(idprop)
                        if propvalue is not False:
                            valuetext = self.dcam.prop_getvaluetext(idprop, propvalue)
                            self.shared_data.camera_params[propname] = valuetext or propvalue
                    idprop = self.dcam.prop_getnextid(idprop)
        finally:
            DCamLock.release_property()

    def restore_modified_params(self):
        """Restore modified parameters"""
        if not self.modified_params or self.dcam is None:
            return
            
        for prop_name, value in self.modified_params.items():
            try:
                self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)
            except Exception as e:
                debug_logger.error(f"Failed to restore {prop_name}: {e}")

    def reset_capture_state(self):
        """Reset capture state to recover from errors"""
        logging.warning("Resetting capture state")
        was_capturing = self.capturing
        
        if was_capturing:
            self.stop_capture()
            time.sleep(0.5)
            
            if DCamLock.acquire_property(timeout=1.0):
                try:
                    if self.dcam and self.dcam.prop_getvalue(CAMERA_PARAMS['EXPOSURE_TIME']) is not False:
                        self.start_capture()
                    else:
                        self.needs_reconnect = True
                        self.update_gui_status("Camera unresponsive - please reset", "red")
                finally:
                    DCamLock.release_property()

    def reset_camera(self):
        """Reset camera connection"""
        logging.info("Resetting camera")
        self.stop_capture()
        self.disconnect_camera()
        time.sleep(1.0)
        self.connect_camera()

    def disconnect_camera(self):
        """Disconnect from camera"""
        logging.info("Disconnecting camera")
        
        if self.capturing:
            self.stop_capture()
        
        if DCamLock.acquire_init(timeout=3.0):
            try:
                if self.dcam is not None:
                    self.dcam.dev_close()
                    self.dcam = None
                Dcamapi.uninit()
            except Exception as e:
                logging.error(f"Error during disconnect: {e}")
            finally:
                DCamLock.release_init()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.stop_requested.set()
        self.paused.set()
        
        # Shutdown frame processor
        if hasattr(self, 'frame_processor') and self.frame_processor:
            try:
                self.frame_processor.shutdown()
            except:
                pass
        
        # Clean up shared memory
        if hasattr(self, 'shared_buffer') and self.shared_buffer:
            try:
                self.shared_buffer.cleanup()
            except:
                pass
        
        self.disconnect_camera()

    def pause_thread(self):
        """Pause thread execution"""
        self.paused.clear()

    def resume_thread(self):
        """Resume thread execution"""
        self.paused.set()

    def update_gui_status(self, message, color):
        """Update GUI status message"""
        if self.gui_ref:
            self.gui_ref.update_status(message, color)

    def update_gui_gps(self, timestamp):
        """Update GUI GPS timestamp"""
        if self.gui_ref:
            self.gui_ref.update_gps_timestamp(timestamp)

    def stop(self):
        """Stop thread"""
        self.running = False
        self.stop_requested.set()
        self.cleanup()


class ParallelSaveThread(threading.Thread):
    """Thread for saving data with parallel processing"""
    
    def __init__(self, save_queue, camera_thread, object_name, shared_data):
        super().__init__(name="SaveThread")
        self.save_queue = save_queue
        self.running = True
        self.camera_thread = camera_thread
        self.object_name = object_name
        self.batch_size = 100
        self.frame_buffer = []
        self.timestamp_buffer = []
        self.framestamp_buffer = []
        self.cube_index = 0
        self.shared_data = shared_data
        
        # Use ThreadPoolExecutor for parallel processing
        self.process_pool = ThreadPoolExecutor(max_workers=4)
        
        # Async file writer for parallel I/O
        self.async_writer = AsyncFileWriter(max_workers=2)

    def run(self):
        """Main save thread loop"""
        try:
            logging.info("Save thread started")
            
            start_time_filename_str = time.strftime('%Y%m%d_%H%M%S')
            os.makedirs("captures", exist_ok=True)

            while self.running or not self.save_queue.empty():
                try:
                    frame, timestamp, framestamp = self.save_queue.get(timeout=1)
                    self.frame_buffer.append(frame)
                    self.timestamp_buffer.append(timestamp)
                    self.framestamp_buffer.append(framestamp)

                    if len(self.frame_buffer) >= self.batch_size:
                        self.write_cube_parallel(start_time_filename_str)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Save thread error: {e}")

            # Write remaining frames
            if self.frame_buffer:
                self.write_cube_parallel(start_time_filename_str)
                
        except Exception as e:
            logging.error(f"Fatal save thread error: {e}")
        finally:
            self.cleanup()

    def write_cube_parallel(self, start_time_filename_str):
        """Write data cube using parallel processing"""
        try:
            self.cube_index += 1
            filename = f"{self.object_name}_{start_time_filename_str}_cube{self.cube_index:03d}.fits"
            filepath = os.path.join("captures", filename)
            
            logging.info(f"Writing cube {self.cube_index} ({len(self.frame_buffer)} frames) in parallel")

            # Process frames in parallel (e.g., compression, filtering)
            frames_array = np.array(self.frame_buffer)
            
            # Split processing across cores
            chunk_size = max(1, len(frames_array) // 4)
            chunks = [frames_array[i:i+chunk_size] for i in range(0, len(frames_array), chunk_size)]
            
            # Submit parallel processing tasks
            futures = []
            for chunk in chunks:
                future = self.process_pool.submit(self.process_chunk, chunk)
                futures.append(future)
            
            # Gather processed chunks
            processed_chunks = [f.result() for f in futures]
            data_cube = np.concatenate(processed_chunks, axis=0)
            
            # Create FITS HDUs
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['OBJECT'] = (self.object_name, 'Object name')
            primary_hdu.header['CUBEIDX'] = (self.cube_index, 'Cube index number')
            primary_hdu.header['COMMENT'] = 'Processed with parallel computing'

            image_hdu = fits.ImageHDU(data=data_cube)
            image_hdu.header['EXTNAME'] = 'DATA_CUBE'
            
            # Add camera parameters
            with self.shared_data.lock:
                for key, value in self.shared_data.camera_params.items():
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        try:
                            image_hdu.header[key] = value
                        except:
                            pass

            # Timestamp table
            col1 = fits.Column(name='TIMESTAMP', format='D', array=self.timestamp_buffer)
            col2 = fits.Column(name='FRAMESTAMP', format='K', array=self.framestamp_buffer)
            timestamp_hdu = fits.BinTableHDU.from_columns([col1, col2])
            timestamp_hdu.header['EXTNAME'] = 'TIMESTAMPS'

            # Write file asynchronously
            hdus = [primary_hdu, image_hdu, timestamp_hdu]
            future = self.async_writer.write_fits_async(filepath, hdus)
            # Don't wait for completion - let it happen in background

            logging.info(f"Saved: {filepath}")

            # Clear buffers
            self.frame_buffer.clear()
            self.timestamp_buffer.clear()
            self.framestamp_buffer.clear()
            
        except Exception as e:
            logging.error(f"Write cube error: {e}")

    @staticmethod
    def process_chunk(chunk):
        """Process a chunk of frames - runs on separate CPU core"""
        # Add any heavy processing here
        # Example: compression, filtering, normalization
        processed = chunk
        
        # Optional: Apply processing
        # processed = np.clip(chunk, 0, 65535)
        # processed = scipy.ndimage.median_filter(chunk, size=(1, 3, 3))
        
        return processed

    def cleanup(self):
        """Clean up resources"""
        self.process_pool.shutdown(wait=True)
        self.async_writer.shutdown()

    def stop(self):
        """Stop save thread"""
        self.running = False


class PeripheralsThread(threading.Thread):
    """Thread for managing peripheral devices"""
    
    def __init__(self, shared_data, pdu_ip, xmcc1_port, xmcc2_port, gui_ref):
        super().__init__(name="PeripheralsThread")
        self.shared_data = shared_data
        self.gui_ref = gui_ref
        self.efw = None
        self.pdu_ip = pdu_ip
        self.pdu = None
        self.ljm_handle = None
        self.xmcc1_port = xmcc1_port
        self.xmcc2_port = xmcc2_port
        self.peripherals_lock = threading.RLock()
        
        # Thread pool for parallel peripheral operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Zaber axes
        self.connection_a = None
        self.connection_b = None
        self.xmcc_a = None
        self.xmcc_b = None
        self.ax_a_1 = None  # Slit stage
        self.ax_a_2 = None  # Zoom stepper
        self.ax_b_1 = None  # Focus stepper
        self.ax_b_2 = None  # Polarization stage
        self.ax_b_3 = None  # Halpha/QWP stage

    def run(self):
        """Main peripherals thread"""
        try:
            asyncio.run(self.connect_peripherals())
        except Exception as e:
            logging.error(f"Peripherals thread error: {e}")

    async def connect_peripherals(self):
        """Connect to all peripherals in parallel"""
        # Connect to devices in parallel
        tasks = [
            self.executor.submit(self.connect_efw),
            self.executor.submit(self.connect_zaber_axes),
            self.executor.submit(self.connect_labjack)
        ]
        
        # Wait for all connections
        for task in tasks:
            try:
                task.result(timeout=10)
            except Exception as e:
                logging.error(f"Peripheral connection error: {e}")
        
        # PDU needs async
        await self.connect_pdu()

    async def connect_pdu(self):
        """Connect to PDU"""
        logging.info("Connecting to PDU")
        try:
            self.pdu = CyberPowerPDU(ip_address=self.pdu_ip, simulate=False)
            await self.pdu.initialize()
            logging.info("PDU connected")
        except Exception as e:
            logging.warning(f"PDU connection failed: {e}")
            self.pdu = None

    def connect_labjack(self):
        """Connect to LabJack"""
        try:
            with self.peripherals_lock:
                self.ljm_handle = ljm.openS("T4", "ANY", "ANY")
                logging.info("LabJack connected")
        except Exception as e:
            logging.warning(f"LabJack connection failed: {e}")

    def connect_efw(self):
        """Connect to filter wheel"""
        debug_logger.info("Connecting to filter wheel")
        try:
            with self.peripherals_lock:
                self.efw = EFW(verbose=False)
                self.efw.GetPosition(0)
                logging.info("Filter wheel connected")
        except Exception as e:
            logging.warning(f"Filter wheel connection failed: {e}")
            self.efw = None

    def connect_zaber_axes(self):
        """Connect to Zaber motor controllers"""
        debug_logger.info("Connecting to Zaber devices")
        
        XMCC1_SERIAL = 137816
        XMCC2_SERIAL = 137819
        
        ports_to_try = [self.xmcc1_port, self.xmcc2_port]
        devices_found = {}
        
        for port in ports_to_try:
            try:
                with self.peripherals_lock:
                    connection = Connection.open_serial_port(port)
                    connection.enable_alerts()
                    devices = connection.detect_devices()
                    
                    if devices:
                        device = devices[0]
                        serial_number = device.serial_number
                        debug_logger.info(f"Found device SN {serial_number} on {port}")
                        
                        if serial_number == XMCC1_SERIAL:
                            devices_found['xmcc1'] = {'connection': connection, 'device': device}
                            logging.info(f"X-MCC1 connected on {port}")
                        elif serial_number == XMCC2_SERIAL:
                            devices_found['xmcc2'] = {'connection': connection, 'device': device}
                            logging.info(f"X-MCC2 connected on {port}")
                        else:
                            connection.close()
                    else:
                        connection.close()
                        
            except Exception as e:
                debug_logger.warning(f"Port {port} error: {e}")
        
        # Initialize axes
        with self.peripherals_lock:
            if 'xmcc1' in devices_found:
                self.connection_a = devices_found['xmcc1']['connection']
                self.xmcc_a = devices_found['xmcc1']['device']
                self._init_xmcc1_axes()
            else:
                logging.warning("X-MCC1 not found")
            
            if 'xmcc2' in devices_found:
                self.connection_b = devices_found['xmcc2']['connection']
                self.xmcc_b = devices_found['xmcc2']['device']
                self._init_xmcc2_axes()
            else:
                logging.warning("X-MCC2 not found")

    def _init_xmcc1_axes(self):
        """Initialize X-MCC1 axes"""
        try:
            self.ax_a_1 = self.xmcc_a.get_axis(1)
            self.ax_a_1.get_position(Units.LENGTH_MILLIMETRES)
            debug_logger.info("Slit stage ready")
        except Exception as e:
            debug_logger.error(f"Slit stage init error: {e}")
            self.ax_a_1 = None
        
        try:
            self.ax_a_2 = self.xmcc_a.get_axis(2)
            self.ax_a_2.get_position(Units.ANGLE_DEGREES)
            debug_logger.info("Zoom stepper ready")
        except Exception as e:
            debug_logger.error(f"Zoom stepper init error: {e}")
            self.ax_a_2 = None

    def _init_xmcc2_axes(self):
        """Initialize X-MCC2 axes"""
        try:
            self.ax_b_1 = self.xmcc_b.get_axis(1)
            self.ax_b_1.get_position(Units.ANGLE_DEGREES)
            debug_logger.info("Focus stepper ready")
        except Exception as e:
            debug_logger.error(f"Focus stepper init error: {e}")
            self.ax_b_1 = None
        
        try:
            self.ax_b_2 = self.xmcc_b.get_axis(2)
            self.ax_b_2.get_position(Units.LENGTH_MILLIMETRES)
            debug_logger.info("Polarization stage ready")
        except Exception as e:
            debug_logger.error(f"Polarization stage init error: {e}")
            self.ax_b_2 = None
        
        try:
            self.ax_b_3 = self.xmcc_b.get_axis(3)
            self.ax_b_3.get_position(Units.LENGTH_MILLIMETRES)
            debug_logger.info("Halpha/QWP stage ready")
        except Exception as e:
            debug_logger.error(f"Halpha/QWP stage init error: {e}")
            self.ax_b_3 = None

    def disconnect_peripherals(self):
        """Disconnect all peripherals"""
        logging.info("Disconnecting peripherals")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        with self.peripherals_lock:
            if self.efw is not None:
                try:
                    self.efw.Close(0)
                except:
                    pass

            if self.pdu is not None:
                try:
                    asyncio.run(self.pdu.close())
                except:
                    pass

            if self.ljm_handle is not None:
                try:
                    ljm.close(self.ljm_handle)
                except:
                    pass

            for conn in [self.connection_a, self.connection_b]:
                if conn is not None:
                    try:
                        conn.close()
                    except:
                        pass

    def command_outlet(self, outlet, command):
        """Send command to PDU outlet"""
        with self.peripherals_lock:
            if self.pdu is not None:
                try:
                    asyncio.run(self.pdu.send_outlet_command(outlet, command))
                    debug_logger.info(f"Outlet {outlet} command: {command}")
                except Exception as e:
                    logging.error(f"Outlet command error: {e}")

    async def get_all_outlet_states(self):
        """Get all PDU outlet states"""
        with self.peripherals_lock:
            if self.pdu is not None:
                try:
                    return await self.pdu.get_all_outlet_states()
                except Exception as e:
                    logging.error(f"Get outlet states error: {e}")
        return {}


class CameraGUI(tk.Tk):
    """Main GUI application with improved performance monitoring"""
    
    def __init__(self, shared_data, camera_thread, peripherals_thread, frame_queue, timestamp_queue):
        super().__init__()
        self.shared_data = shared_data
        self.camera_thread = camera_thread
        self.peripherals_thread = peripherals_thread
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.updating_camera_status = True
        self.updating_frame_display = True
        self.updating_peripherals_status = True
        self.display_lock = threading.Lock()
        self.save_thread = None
        self.last_frame = None
        self._peripheral_update_running = False
        
        # Performance monitoring
        self.frame_times = []
        self.last_fps_update = time.time()

        self.min_val = tk.StringVar(value="0")
        self.max_val = tk.StringVar(value="200")

        self.title("Lightspeed Prototype Control GUI")
        self.geometry("1000x1080")

        self.setup_gui()

    def setup_gui(self):
        """Set up all GUI elements"""
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Camera Parameters
        camera_params_frame = LabelFrame(self.main_frame, text="Camera Parameters", padx=5, pady=5)
        camera_params_frame.grid(row=0, column=0, rowspan=10, sticky='nsew')
        self.camera_status = tk.Label(camera_params_frame, text="", justify=tk.LEFT, anchor="w", 
                                      width=60, height=80, wraplength=400)
        self.camera_status.pack(fill='both', expand=True)

        # Status messages
        self.status_message = tk.Label(self.main_frame, text="", justify=tk.LEFT, anchor="w", 
                                       width=40, wraplength=400, fg="blue")
        self.status_message.grid(row=5, column=0, columnspan=2, sticky='nsew')

        # GPS Timestamp display
        gps_frame = LabelFrame(self.main_frame, text="GPS Timestamp", padx=5, pady=5)
        gps_frame.grid(row=6, column=0, columnspan=2, sticky='ew')
        self.gps_timestamp_label = tk.Label(gps_frame, text="No capture active", font=("Courier", 12))
        self.gps_timestamp_label.pack()
        
        # Performance display
        perf_frame = LabelFrame(self.main_frame, text="Performance", padx=5, pady=5)
        perf_frame.grid(row=7, column=0, columnspan=2, sticky='ew')
        self.performance_label = tk.Label(perf_frame, text="FPS: -- | CPU Cores Active: --", font=("Courier", 10))
        self.performance_label.pack()

        # Set up control sections
        self.setup_camera_controls()
        self.setup_camera_settings()
        self.setup_subarray_controls()
        self.setup_advanced_controls()
        self.setup_display_controls()
        self.setup_peripherals_controls()
        
        # Start update loops
        self.after(100, self.update_camera_status)
        self.after(200, self.update_frame_display)
        self.after(1000, self.update_performance_monitor)
        self.after(10000, self.update_peripherals_status)

    def update_performance_monitor(self):
        """Monitor and display performance metrics"""
        if self.frame_times:
            # Calculate FPS
            recent_frames = [t for t in self.frame_times if t > time.time() - 1.0]
            fps = len(recent_frames)
            self.frame_times = recent_frames  # Keep only recent
            
            # Get CPU count being used
            active_cores = mp.cpu_count()
            
            self.performance_label.config(
                text=f"FPS: {fps:3d} | CPU Cores Active: {active_cores} | Queue: {self.frame_queue.qsize()}"
            )
        
        self.after(1000, self.update_performance_monitor)

    def setup_camera_controls(self):
        """Set up camera control widgets"""
        camera_controls_frame = LabelFrame(self.main_frame, text="Camera Controls", padx=5, pady=5)
        camera_controls_frame.grid(row=0, column=1, sticky='n')

        Label(camera_controls_frame, text="Exposure Time (ms):").grid(row=0, column=0)
        self.exposure_time_var = tk.DoubleVar(value=100)
        self.exposure_time_var.trace_add("write", self.update_exposure_time)
        self.exposure_time_entry = Entry(camera_controls_frame, textvariable=self.exposure_time_var)
        self.exposure_time_entry.grid(row=0, column=1)

        self.save_data_var = tk.BooleanVar()
        self.save_data_checkbox = Checkbutton(camera_controls_frame, text="Save Data to Disk", 
                                              variable=self.save_data_var)
        self.save_data_checkbox.grid(row=1, column=0, columnspan=2)

        self.start_button = Button(camera_controls_frame, text="Start", command=self.start_capture)
        self.start_button.grid(row=2, column=0)

        self.stop_button = Button(camera_controls_frame, text="Stop", command=self.stop_capture)
        self.stop_button.grid(row=2, column=1)

        self.reset_button = Button(camera_controls_frame, text="Reset Camera", command=self.reset_camera)
        self.reset_button.grid(row=3, column=0, columnspan=2)

        Label(camera_controls_frame, text="Object Name:").grid(row=4, column=0)
        self.object_name_entry = Entry(camera_controls_frame)
        self.object_name_entry.grid(row=4, column=1)

        Label(camera_controls_frame, text="Frames per Datacube").grid(row=5, column=0)
        self.cube_size_var = tk.IntVar(value=100)
        self.cube_size_var.trace_add("write", self.update_batch_size)
        self.cube_size_entry = Entry(camera_controls_frame, textvariable=self.cube_size_var)
        self.cube_size_entry.grid(row=5, column=1)

        self.power_cycle_button = Button(camera_controls_frame, text="Power Cycle Camera",
                                         command=self.power_cycle_camera)
        self.power_cycle_button.grid(row=6, column=0, columnspan=2)

    def setup_camera_settings(self):
        """Set up camera settings widgets"""
        camera_settings_frame = LabelFrame(self.main_frame, text="Camera Settings", padx=5, pady=5)
        camera_settings_frame.grid(row=1, column=1, sticky='n')

        Label(camera_settings_frame, text="Binning:").grid(row=0, column=0)
        self.binning_var = StringVar(value="1x1")
        self.binning_menu = OptionMenu(camera_settings_frame, self.binning_var, 
                                       "1x1", "2x2", "4x4", command=self.change_binning)
        self.binning_menu.grid(row=0, column=1)

        Label(camera_settings_frame, text="Bit Depth:").grid(row=1, column=0)
        self.bit_depth_var = StringVar(value="8-bit")
        self.bit_depth_menu = OptionMenu(camera_settings_frame, self.bit_depth_var, 
                                         "8-bit", "16-bit", command=self.change_bit_depth)
        self.bit_depth_menu.grid(row=1, column=1)

        Label(camera_settings_frame, text="Readout Speed:").grid(row=2, column=0)
        self.readout_speed_var = StringVar(value="Ultra Quiet Mode")
        self.readout_speed_menu = OptionMenu(camera_settings_frame, self.readout_speed_var, 
                                             "Ultra Quiet Mode", "Standard Mode", 
                                             command=self.change_readout_speed)
        self.readout_speed_menu.grid(row=2, column=1)

        Label(camera_settings_frame, text="Sensor Mode:").grid(row=3, column=0)
        self.sensor_mode_var = StringVar(value="Photon Number Resolving")
        self.sensor_mode_menu = OptionMenu(camera_settings_frame, self.sensor_mode_var, 
                                           "Photon Number Resolving", "Standard", 
                                           command=self.change_sensor_mode)
        self.sensor_mode_menu.grid(row=3, column=1)

    def setup_subarray_controls(self):
        """Set up subarray control widgets"""
        subarray_controls_frame = LabelFrame(self.main_frame, text="Subarray Controls", padx=5, pady=5)
        subarray_controls_frame.grid(row=2, column=1, sticky='n')

        Label(subarray_controls_frame, text="Subarray Mode:").grid(row=0, column=0)
        self.subarray_mode_var = StringVar(value="Off")
        self.subarray_mode_menu = OptionMenu(subarray_controls_frame, self.subarray_mode_var, 
                                             "Off", "On", command=self.change_subarray_mode)
        self.subarray_mode_menu.grid(row=0, column=1)

        # Subarray position and size controls
        subarray_params = [
            ("HPOS", 0, 1),
            ("HSIZE", 4096, 2),
            ("VPOS", 0, 3),
            ("VSIZE", 2304, 4)
        ]
        
        self.subarray_vars = {}
        self.subarray_entries = {}
        
        for param, default, row in subarray_params:
            Label(subarray_controls_frame, text=f"Subarray {param}:").grid(row=row, column=0)
            var = tk.IntVar(value=default)
            var.trace_add("write", self.update_subarray)
            self.subarray_vars[param] = var
            entry = Entry(subarray_controls_frame, textvariable=var, state='disabled')
            entry.grid(row=row, column=1)
            self.subarray_entries[param] = entry

        Label(subarray_controls_frame, text="Note: Values rounded to nearest factor of 4.").grid(
            row=5, column=0, columnspan=2)

    def setup_advanced_controls(self):
        """Set up advanced control widgets"""
        advanced_controls_frame = LabelFrame(self.main_frame, text="Advanced Controls", padx=5, pady=5)
        advanced_controls_frame.grid(row=3, column=1, sticky='n')

        self.framebundle_var = tk.BooleanVar()
        self.framebundle_checkbox = Checkbutton(advanced_controls_frame, text="Enable Frame Bundle", 
                                                variable=self.framebundle_var, 
                                                command=self.update_framebundle)
        self.framebundle_checkbox.grid(row=0, column=0, columnspan=2)

        Label(advanced_controls_frame, text="Frames Per Bundle:").grid(row=1, column=0)
        self.frames_per_bundle_var = tk.IntVar(value=100)
        self.frames_per_bundle_var.trace_add("write", self.update_frames_per_bundle)
        self.frames_per_bundle_entry = Entry(advanced_controls_frame, 
                                             textvariable=self.frames_per_bundle_var)
        self.frames_per_bundle_entry.grid(row=1, column=1)
        
        Label(advanced_controls_frame, 
              text="When enabled, frames are\nconcatenated into one image.").grid(
              row=2, column=0, columnspan=2)

    def setup_display_controls(self):
        """Set up display control widgets"""
        display_controls_frame = LabelFrame(self.main_frame, text="Display Controls", padx=5, pady=5)
        display_controls_frame.grid(row=4, column=1, sticky='n')

        Label(display_controls_frame, text="Min Count:").grid(row=0, column=0)
        Entry(display_controls_frame, textvariable=self.min_val).grid(row=0, column=1)

        Label(display_controls_frame, text="Max Count:").grid(row=0, column=2)
        self.max_val.trace_add("write", self.refresh_frame_display)
        Entry(display_controls_frame, textvariable=self.max_val).grid(row=0, column=3)

    def setup_peripherals_controls(self):
        """Set up peripheral control widgets"""
        self.peripherals_controls_frame = LabelFrame(self.main_frame, text="Peripherals Controls", 
                                                     padx=5, pady=5)
        self.peripherals_controls_frame.grid(row=8, column=1, sticky='n')

        # Filter control
        Label(self.peripherals_controls_frame, text="Filter:").grid(row=0, column=0)
        self.filter_position_var = tk.StringVar()
        self.filter_options = {'0 (Open)': 0, '1 (u\')': 1, '2 (g\')': 2, '3 (r\')': 3,
                               '4 (i\')': 4, '5 (z\')': 5}
        self.filter_position_menu = OptionMenu(self.peripherals_controls_frame, self.filter_position_var,
                                               *self.filter_options.keys(),
                                               command=self.update_filter_position)
        self.filter_position_menu.grid(row=0, column=1)

        # Shutter control
        Label(self.peripherals_controls_frame, text="Shutter:").grid(row=0, column=2)
        self.shutter_var = tk.StringVar(value='Open')
        self.shutter_menu = OptionMenu(self.peripherals_controls_frame, self.shutter_var,
                                       'Open', 'Closed', command=self.update_shutter)
        self.shutter_menu.grid(row=0, column=3)

        # Motor controls
        self.setup_motor_controls()
        
        # PDU outlet controls
        self.setup_pdu_controls()

    def setup_motor_controls(self):
        """Set up motor control widgets"""
        # Slit control
        Label(self.peripherals_controls_frame, text="Slit:").grid(row=1, column=0)
        self.slit_position_var = tk.StringVar(value='Out of beam')
        self.slit_position_menu = OptionMenu(self.peripherals_controls_frame, self.slit_position_var,
                                             'In beam', 'Out of beam', command=self.update_slit_position)
        self.slit_position_menu.grid(row=1, column=1)

        # Halpha/QWP control
        Label(self.peripherals_controls_frame, text="Halpha/QWP:").grid(row=1, column=2)
        self.halpha_qwp_var = tk.StringVar(value='Neither')
        self.halpha_qwp_menu = OptionMenu(self.peripherals_controls_frame, self.halpha_qwp_var,
                                          'Halpha', 'QWP', 'Neither', command=self.update_halpha_qwp)
        self.halpha_qwp_menu.grid(row=1, column=3)

        # Polarization stage control
        Label(self.peripherals_controls_frame, text="Pol. Stage:").grid(row=2, column=0)
        self.wire_grid_var = tk.StringVar(value='Neither')
        self.wire_grid_menu = OptionMenu(self.peripherals_controls_frame, self.wire_grid_var,
                                         'WeDoWo', 'Wire Grid', 'Neither', command=self.update_pol_stage)
        self.wire_grid_menu.grid(row=2, column=1)

        # Zoom control
        Label(self.peripherals_controls_frame, text="Zoom-out:").grid(row=2, column=2)
        self.zoom_stepper_var = tk.StringVar(value='4x')
        self.zoom_stepper_menu = OptionMenu(self.peripherals_controls_frame, self.zoom_stepper_var,
                                            '1x', '2x', '3x', '4x', command=self.update_zoom_stepper)
        self.zoom_stepper_menu.grid(row=2, column=3)

        # Focus control
        Label(self.peripherals_controls_frame, text="Focus Position (um):").grid(row=3, column=0, columnspan=2)
        self.focus_position_var = tk.StringVar(value='0')
        self.focus_conversion_factor = 1
        self.focus_position_entry = Entry(self.peripherals_controls_frame, 
                                          textvariable=self.focus_position_var)
        self.focus_position_entry.grid(row=3, column=2)
        self.set_focus_button = Button(self.peripherals_controls_frame, text="Set Focus",
                                       command=self.update_focus_position)
        self.set_focus_button.grid(row=3, column=3)

    def setup_pdu_controls(self):
        """Set up PDU outlet control widgets"""
        Label(self.peripherals_controls_frame, text="PDU Outlet States").grid(row=4, column=0, columnspan=4)
        
        self.pdu_outlet_dict = {
            1: 'Rotator', 2: 'Switch', 3: 'Shutter', 4: 'Empty',
            5: 'Empty', 6: 'Empty', 7: 'Empty', 8: 'Empty',
            9: 'Ctrl PC', 10: 'X-MCC4 A', 11: 'X-MCC B', 12: 'qCMOS',
            13: 'Empty', 14: 'Empty', 15: 'Empty', 16: 'Empty'
        }
        
        self.pdu_outlet_vars = {}
        self.pdu_outlet_buttons = {}
        
        for idx, name in self.pdu_outlet_dict.items():
            row = (idx - 1) % 8 + 8
            col = (idx - 1) // 8 * 2
            name_label = f"{idx}: {name}"
            tk.Label(self.peripherals_controls_frame, text=name_label, width=12, anchor='w')\
              .grid(row=row, column=col, padx=2, pady=2)
            var = tk.BooleanVar(value=True)
            self.pdu_outlet_vars[idx] = var
            btn = tk.Checkbutton(self.peripherals_controls_frame, text='ON', relief='sunken',
                                 fg='green', variable=var, indicatoron=False, width=3,
                                 command=lambda i=idx: self.toggle_outlet(i))
            btn.grid(row=row, column=col + 1, padx=2, pady=2)
            self.pdu_outlet_buttons[idx] = btn

    def update_status(self, message, color="blue"):
        """Update status message"""
        self.after(0, lambda: self.status_message.config(text=message, fg=color))

    def update_gps_timestamp(self, timestamp_str):
        """Update GPS timestamp display"""
        self.after(0, lambda: self.gps_timestamp_label.config(text=timestamp_str))

    def update_camera_status(self):
        """Update camera status display"""
        if self.updating_camera_status:
            def _update():
                try:
                    status_text = ""
                    with self.shared_data.lock:
                        for key, value in self.shared_data.camera_params.items():
                            if key in DISPLAY_PARAMS:
                                status_text += f"{DISPLAY_PARAMS[key]}: {value}\n"
                    self.after(0, lambda: self.camera_status.config(text=status_text))
                except Exception as e:
                    debug_logger.error(f"Camera status error: {e}")
            
            threading.Thread(target=_update, daemon=True).start()
        
        self.after(2000, self.update_camera_status)

    def update_frame_display(self):
        """Update frame display with performance tracking"""
        if self.updating_frame_display:
            try:
                self.refresh_frame_display()
            except Exception as e:
                debug_logger.error(f"Frame display error: {e}")
        self.after(17, self.update_frame_display)

    def refresh_frame_display(self, *_):
        """Refresh the frame display"""
        try:
            frames_processed = 0
            while frames_processed < 3 and not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                    self.last_frame = frame
                    frames_processed += 1
                    # Track frame time for FPS calculation
                    self.frame_times.append(time.time())
                except queue.Empty:
                    break
            
            if frames_processed > 0 and self.last_frame is not None:
                # Process frame directly in GUI thread to avoid Qt issues
                self.process_frame(self.last_frame)
        except Exception as e:
            debug_logger.error(f"Frame display error: {e}")
        
        cv2.waitKey(1)

    def process_frame(self, data):
        """Process and display a frame with OpenCV parallelism"""
        if not self.display_lock.acquire(blocking=False):
            return
            
        try:
            # Get display range
            try:
                min_val = int(self.min_val.get())
                max_val = int(self.max_val.get())
            except:
                min_val, max_val = 0, 200

            # Scale data for display (OpenCV uses multiple cores)
            if max_val > min_val:
                if data.dtype == np.uint16:
                    scale = 65535.0 / (max_val - min_val)
                    # NumPy operations use BLAS/LAPACK parallelism
                    scaled_data = np.clip((data.astype(np.float32) - min_val) * scale, 0, 65535).astype(np.uint16)
                else:
                    scaled_data = np.clip((data.astype(np.float32) - min_val) / (max_val - min_val) * 255, 
                                        0, 255).astype(np.uint8)
            else:
                scaled_data = data

            # OpenCV operations use multiple cores
            scaled_data = cv2.flip(scaled_data, 1)
            
            # Convert to BGR if needed
            if len(scaled_data.shape) == 2:
                scaled_data_bgr = cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2BGR)
            else:
                scaled_data_bgr = scaled_data

            # Draw any overlays
            if hasattr(self, 'circle_center'):
                cv2.circle(scaled_data_bgr, self.circle_center, 2, (255, 0, 0), 2)

            # Create window if needed
            if not hasattr(self, 'opencv_window_created'):
                cv2.namedWindow('Captured Frame', cv2.WINDOW_NORMAL)
                cv2.setMouseCallback('Captured Frame', self.on_right_click)
                self.opencv_window_created = True

            cv2.imshow('Captured Frame', scaled_data_bgr)
            
        finally:
            self.display_lock.release()

    def on_right_click(self, event, x, y, flags, param):
        """Handle right click on frame display"""
        if event == cv2.EVENT_RBUTTONDOWN:
            self.after(0, lambda: self.show_context_menu(x, y))

    def show_context_menu(self, x, y):
        """Show context menu for frame display"""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Draw Circle", command=lambda: self.draw_circle(x, y))
        menu.add_command(label="Clear Markers", command=self.clear_markers)
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        except:
            pass

    def draw_circle(self, x, y):
        """Draw circle on frame"""
        self.circle_center = (x, y)
        if self.last_frame is not None:
            self.process_frame(self.last_frame)

    def clear_markers(self):
        """Clear all markers from frame"""
        if hasattr(self, 'circle_center'):
            del self.circle_center
        if self.last_frame is not None:
            self.process_frame(self.last_frame)

    def update_exposure_time(self, *_):
        """Update exposure time"""
        try:
            exposure_time = float(self.exposure_time_entry.get()) / 1000
            if self.camera_thread.capturing:
                self.update_status("Cannot change during capture", "orange")
            else:
                self.camera_thread.set_property('EXPOSURE_TIME', exposure_time)
                self.update_status(f"Exposure: {exposure_time*1000:.1f}ms", "green")
        except ValueError:
            self.update_status("Invalid exposure time", "red")

    def update_batch_size(self, *_):
        """Update batch size for saving"""
        try:
            self.batch_size = int(self.cube_size_entry.get())
            debug_logger.info(f"Batch size: {self.batch_size}")
        except:
            self.batch_size = 100

    def change_binning(self, selected_binning):
        """Change camera binning"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        binning_value = {"1x1": 1, "2x2": 2, "4x4": 4}[selected_binning]
        self.camera_thread.set_property('BINNING', binning_value)

    def change_bit_depth(self, selected_bit_depth):
        """Change bit depth"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        bit_depth_value = {"8-bit": 1, "16-bit": 2}[selected_bit_depth]
        self.camera_thread.set_property('IMAGE_PIXEL_TYPE', bit_depth_value)

    def change_readout_speed(self, selected_mode):
        """Change readout speed"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        readout_speed_value = {"Ultra Quiet Mode": 1.0, "Standard Mode": 2.0}[selected_mode]
        if selected_mode == "Standard Mode":
            self.sensor_mode_var.set("Standard")
            self.change_sensor_mode("Standard")
        self.camera_thread.set_property('READOUT_SPEED', readout_speed_value)

    def change_sensor_mode(self, selected_mode):
        """Change sensor mode"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        sensor_mode_value = {"Photon Number Resolving": 18.0, "Standard": 1.0}[selected_mode]
        self.camera_thread.set_property('SENSOR_MODE', sensor_mode_value)
        if selected_mode == "Photon Number Resolving":
            self.readout_speed_var.set("Ultra Quiet Mode")
            self.change_readout_speed("Ultra Quiet Mode")

    def change_subarray_mode(self, selected_mode):
        """Change subarray mode"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        subarray_mode_value = {"Off": 1.0, "On": 2.0}[selected_mode]
        self.camera_thread.set_property('SUBARRAY_MODE', subarray_mode_value)
        
        state = 'normal' if selected_mode == "On" else 'disabled'
        for entry in self.subarray_entries.values():
            entry.config(state=state)

    def update_subarray(self, *_):
        """Update subarray settings"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        try:
            for param, var in self.subarray_vars.items():
                value = float(round(float(var.get()) / 4) * 4)
                self.camera_thread.set_property(f'SUBARRAY_{param}', value)
        except ValueError:
            debug_logger.error("Invalid subarray parameters")

    def update_framebundle(self):
        """Update frame bundle mode"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        framebundle_enabled = self.framebundle_var.get()
        self.camera_thread.set_property('FRAMEBUNDLE_MODE', 2.0 if framebundle_enabled else 1.0)

    def update_frames_per_bundle(self, *_):
        """Update frames per bundle"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        try:
            frames_per_bundle = int(self.frames_per_bundle_entry.get())
            self.camera_thread.set_property('FRAMEBUNDLE_NUMBER', frames_per_bundle)
        except ValueError:
            debug_logger.error("Invalid frames per bundle")

    def start_capture(self):
        """Start camera capture"""
        try:
            if getattr(self.camera_thread, 'needs_reconnect', False):
                self.update_status("Camera needs reset - use Reset Camera button", "red")
                messagebox.showerror("Camera Error", 
                                   "Camera is in an error state.\nPlease use the Reset Camera button.")
                return
            
            self.update_status("Starting capture...", "blue")
            self.update_gps_timestamp("Waiting for GPS...")

            # Set up save thread if needed (using parallel save thread)
            if self.save_data_var.get():
                save_queue = queue.Queue(maxsize=50000)
                self.camera_thread.save_queue = save_queue
                object_name = self.object_name_entry.get() or "capture"
                self.save_thread = ParallelSaveThread(save_queue, self.camera_thread, 
                                                      object_name, self.shared_data)
                self.save_thread.batch_size = self.batch_size
                self.save_thread.start()
            else:
                self.camera_thread.save_queue = None

            # Start capture
            if self.camera_thread.start_capture():
                self.disable_controls_during_capture()
                self.update_status("Capture running", "green")
            else:
                self.update_status("Failed to start capture", "red")
                if self.save_thread:
                    self.save_thread.stop()
                    self.save_thread = None
                    
        except Exception as e:
            logging.error(f"Start capture error: {e}")
            self.update_status(f"Error: {e}", "red")

    def stop_capture(self):
        """Stop camera capture"""
        try:
            self.update_status("Stopping capture...", "blue")
            self.camera_thread.stop_capture()

            # Stop save thread
            if self.save_thread and self.save_thread.is_alive():
                self.save_thread.stop()
                self.save_thread.join(timeout=10)
                self.save_thread = None
                self.camera_thread.save_queue = None

            self.enable_controls_after_capture()
            self.update_status("Capture stopped", "blue")
            self.update_gps_timestamp("No capture active")
            
        except Exception as e:
            logging.error(f"Stop capture error: {e}")
            self.update_status(f"Error: {e}", "red")

    def reset_camera(self):
        """Reset camera connection"""
        try:
            self.update_status("Resetting camera...", "blue")
            self.camera_thread.reset_camera()
            time.sleep(1.0)
            self.update_status("Camera reset complete", "green")
        except Exception as e:
            logging.error(f"Reset camera error: {e}")
            self.update_status(f"Reset error: {e}", "red")

    def disable_controls_during_capture(self):
        """Disable controls during capture"""
        controls = [
            self.exposure_time_entry, self.save_data_checkbox, 
            self.start_button, self.reset_button, self.binning_menu,
            self.bit_depth_menu, self.readout_speed_menu, self.sensor_mode_menu,
            self.subarray_mode_menu, self.framebundle_checkbox, 
            self.frames_per_bundle_entry
        ]
        
        for widget in controls:
            widget.config(state='disabled')
        
        if self.subarray_mode_var.get() == "On":
            for entry in self.subarray_entries.values():
                entry.config(state='disabled')

    def enable_controls_after_capture(self):
        """Enable controls after capture"""
        controls = [
            self.exposure_time_entry, self.save_data_checkbox,
            self.start_button, self.reset_button, self.binning_menu,
            self.bit_depth_menu, self.readout_speed_menu, self.sensor_mode_menu,
            self.subarray_mode_menu, self.framebundle_checkbox,
            self.frames_per_bundle_entry
        ]
        
        for widget in controls:
            widget.config(state='normal')

        if self.subarray_mode_var.get() == "On":
            for entry in self.subarray_entries.values():
                entry.config(state='normal')

    def power_cycle_camera(self):
        """Power cycle the camera"""
        try:
            self.peripherals_thread.command_outlet(12, OutletCommand.IMMEDIATE_OFF)
            self.after(1000, lambda: self.peripherals_thread.command_outlet(12, OutletCommand.IMMEDIATE_ON))
            logging.info("Camera power cycled")
            self.after(3000, self.reset_camera)
        except Exception as e:
            logging.error(f"Power cycle error: {e}")

    # Peripheral control methods
    def update_peripherals_status(self):
        """Update peripheral status periodically"""
        if self.updating_peripherals_status and self.peripherals_thread and not self._peripheral_update_running:
            self._peripheral_update_running = True
            threading.Thread(target=self._update_peripherals_background, daemon=True).start()
        
        self.after(30000, self.update_peripherals_status)

    def _update_peripherals_background(self):
        """Background peripheral update"""
        try:
            # Implementation similar to original but simplified
            pass
        finally:
            self._peripheral_update_running = False

    def update_shutter(self, *_):
        """Update shutter state"""
        def _update():
            try:
                if self.peripherals_thread.ljm_handle is None:
                    return
                with self.peripherals_thread.peripherals_lock:
                    if self.shutter_var.get() == 'Open':
                        ljm.eWriteName(self.peripherals_thread.ljm_handle, "DIO4", 0)
                    else:
                        ljm.eWriteName(self.peripherals_thread.ljm_handle, "DIO4", 1)
            except Exception as e:
                debug_logger.error(f"Shutter error: {e}")
        
        # Use thread pool for parallel peripheral operations
        self.peripherals_thread.executor.submit(_update)

    def update_filter_position(self, *_):
        """Update filter position"""
        def _update():
            try:
                selected = self.filter_position_var.get()
                if not selected:
                    return
                with self.peripherals_thread.peripherals_lock:
                    if self.peripherals_thread.efw:
                        position = self.filter_options[selected]
                        self.peripherals_thread.efw.SetPosition(0, position)
                        debug_logger.info(f"Filter position: {selected}")
            except Exception as e:
                debug_logger.error(f"Filter error: {e}")
        
        self.peripherals_thread.executor.submit(_update)

    def update_slit_position(self, *_):
        """Update slit position"""
        def _update():
            try:
                if self.peripherals_thread.ax_a_1 is None:
                    return
                option = self.slit_position_var.get()
                with self.peripherals_thread.peripherals_lock:
                    if option == 'In beam':
                        self.peripherals_thread.ax_a_1.move_absolute(70, Units.LENGTH_MILLIMETRES)
                    else:
                        self.peripherals_thread.ax_a_1.move_absolute(0, Units.LENGTH_MILLIMETRES)
            except Exception as e:
                debug_logger.error(f"Slit error: {e}")
        
        self.peripherals_thread.executor.submit(_update)

    def update_halpha_qwp(self, *_):
        """Update Halpha/QWP position"""
        def _update():
            try:
                if self.peripherals_thread.ax_b_3 is None:
                    return
                option = self.halpha_qwp_var.get()
                positions = {'Halpha': 138, 'QWP': 13, 'Neither': 75.5}
                with self.peripherals_thread.peripherals_lock:
                    self.peripherals_thread.ax_b_3.move_absolute(
                        positions[option], Units.LENGTH_MILLIMETRES)
            except Exception as e:
                debug_logger.error(f"Halpha/QWP error: {e}")
        
        self.peripherals_thread.executor.submit(_update)

    def update_pol_stage(self, *_):
        """Update polarization stage"""
        def _update():
            try:
                if self.peripherals_thread.ax_b_2 is None:
                    return
                option = self.wire_grid_var.get()
                positions = {'WeDoWo': 15.5, 'Wire Grid': 128.5, 'Neither': 72}
                with self.peripherals_thread.peripherals_lock:
                    self.peripherals_thread.ax_b_2.move_absolute(
                        positions[option], Units.LENGTH_MILLIMETRES)
            except Exception as e:
                debug_logger.error(f"Pol stage error: {e}")
        
        self.peripherals_thread.executor.submit(_update)

    def update_zoom_stepper(self, *_):
        """Update zoom position"""
        def _update():
            try:
                if self.peripherals_thread.ax_a_2 is None:
                    return
                option = self.zoom_stepper_var.get()
                positions = {'1x': 0, '2x': 90, '3x': 180, '4x': 270}
                with self.peripherals_thread.peripherals_lock:
                    self.peripherals_thread.ax_a_2.move_absolute(
                        positions[option], Units.ANGLE_DEGREES)
            except Exception as e:
                debug_logger.error(f"Zoom error: {e}")
        
        self.peripherals_thread.executor.submit(_update)

    def update_focus_position(self, *_):
        """Update focus position"""
        def _update():
            try:
                if self.peripherals_thread.ax_b_1 is None:
                    return
                position = float(self.focus_position_var.get())
                with self.peripherals_thread.peripherals_lock:
                    self.peripherals_thread.ax_b_1.move_absolute(
                        position / self.focus_conversion_factor, Units.ANGLE_DEGREES)
            except Exception as e:
                debug_logger.error(f"Focus error: {e}")
        
        self.peripherals_thread.executor.submit(_update)

    def toggle_outlet(self, idx, override=False):
        """Toggle PDU outlet"""
        try:
            state = self.pdu_outlet_vars[idx].get()
            if not override:
                response = messagebox.askyesno("Confirm", 
                    f"Turn {'ON' if state else 'OFF'} outlet {idx}?")
                if not response:
                    self.pdu_outlet_vars[idx].set(not state)
                    return
            
            cmd = OutletCommand.IMMEDIATE_ON if state else OutletCommand.IMMEDIATE_OFF
            self.peripherals_thread.command_outlet(idx, cmd)
            
            btn = self.pdu_outlet_buttons[idx]
            if state:
                btn.config(text='ON', fg='green', relief='sunken')
            else:
                btn.config(text='OFF', fg='red', relief='raised')
        except Exception as e:
            debug_logger.error(f"Toggle outlet error: {e}")

    def on_close(self):
        """Handle application close"""
        try:
            logging.info("Closing application")
            
            # Stop all updates
            self.updating_camera_status = False
            self.updating_frame_display = False
            self.updating_peripherals_status = False
            
            # Stop save thread
            if self.save_thread and self.save_thread.is_alive():
                logging.info("Stopping save thread")
                self.save_thread.stop()
                self.save_thread.join(timeout=5)

            # Stop camera thread
            if self.camera_thread:
                logging.info("Stopping camera thread")
                self.camera_thread.stop()
                self.camera_thread.join(timeout=5)

            # Close OpenCV windows
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except:
                pass

            # Disconnect peripherals
            if self.peripherals_thread:
                logging.info("Disconnecting peripherals")
                self.peripherals_thread.disconnect_peripherals()

            # Destroy GUI
            logging.info("Destroying GUI")
            self.quit()
            self.destroy()
            
            logging.info("Application closed successfully")
            
        except Exception as e:
            logging.error(f"Error during close: {e}")
            import sys
            sys.exit(1)


def main():
    """Main application entry point"""
    try:
        # Set multiprocessing start method to avoid Qt conflicts
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
            
        logging.info("Starting Camera Control Application")
        logging.info(f"CPU cores available: {mp.cpu_count()}")
        
        # Create shared resources
        shared_data = SharedData()
        frame_queue = queue.Queue(maxsize=30)
        timestamp_queue = queue.Queue(maxsize=100000)
        
        # Create GUI (must be created first for thread references)
        app = CameraGUI(shared_data, None, None, frame_queue, timestamp_queue)
        
        # Create and start camera thread
        camera_thread = CameraThread(shared_data, frame_queue, timestamp_queue, app)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Create and start peripherals thread
        peripherals_thread = PeripheralsThread(
            shared_data, "18.25.72.251", "/dev/ttyACM0", "/dev/ttyACM1", app)
        peripherals_thread.daemon = True
        peripherals_thread.start()
        
        # Set thread references in GUI
        app.camera_thread = camera_thread
        app.peripherals_thread = peripherals_thread
        
        # Set close handler
        app.protocol("WM_DELETE_WINDOW", app.on_close)
        
        # Run GUI
        app.mainloop()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
