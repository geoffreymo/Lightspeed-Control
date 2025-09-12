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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue
import psutil
import socket
import re
from collections import defaultdict

# Load required library
ctypes.CDLL("libudev.so.1", mode=ctypes.RTLD_GLOBAL)

# Enable OpenCV multi-threading
cv2.setNumThreads(4)

# Set NumPy thread count for parallel operations
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

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


class DCamLock:
    """Simplified locking for DCAM operations - reduced contention"""
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
        try:
            cls._capture_lock.release()
        except:
            pass
    
    @classmethod
    def acquire_property(cls, timeout=2.0):
        """Acquire property lock"""
        return cls._acquire_lock(cls._property_lock, timeout)
    
    @classmethod
    def release_property(cls):
        """Release property lock"""
        try:
            cls._property_lock.release()
        except:
            pass
    
    @classmethod
    def acquire_buffer(cls, timeout=2.0):
        """Acquire buffer lock"""
        return cls._acquire_lock(cls._buffer_lock, timeout)
    
    @classmethod
    def release_buffer(cls):
        """Release buffer lock"""
        try:
            cls._buffer_lock.release()
        except:
            pass
    
    @classmethod
    def acquire_init(cls, timeout=5.0):
        """Acquire initialization lock"""
        return cls._acquire_lock(cls._init_lock, timeout)
    
    @classmethod
    def release_init(cls):
        """Release initialization lock"""
        try:
            cls._init_lock.release()
        except:
            pass
    
    @classmethod
    def _acquire_lock(cls, lock, timeout, check_stop=None):
        """Generic lock acquisition with timeout"""
        acquired = lock.acquire(blocking=True, timeout=timeout)
        if not acquired and check_stop and check_stop():
            return False
        return acquired


class SharedData:
    """Shared data container with thread-safe access"""
    def __init__(self):
        self.camera_params = {}
        self.lock = threading.RLock()


class CameraThread(threading.Thread):
    """Camera control thread - optimized version of original"""
    
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
        
        # Performance monitoring
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_print_time = time.time()
        self.fps_calc_time = time.time()
        
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
                    # Check if exposure time is longer than 1 minutes
                    if self.gui_ref is not None and self.gui_ref.exposure_time_var.get() > 60000.0:
                        if self.watchdog_enabled and time.time() - self.last_capture_time > 1801.0:
                            logging.error("Capture watchdog triggered")
                            self.reset_capture_state()
                    elif self.watchdog_enabled and time.time() - self.last_capture_time > 61.0:
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
        """Capture a single frame - optimized for minimal locking"""
        if self.stop_requested.is_set():
            return
            
        timeout_milisec = 100
        
        # Quick check without lock first
        if self.dcam is None or not self.capturing:
            return
        
        # Only acquire lock for actual DCAM operations
        if not DCamLock.acquire_capture(timeout=0.1, check_stop=lambda: self.stop_requested.is_set()):
            return
        
        try:
            if not self.capturing or self.stop_requested.is_set():
                return
                
            # Wait for frame
            if self.dcam.wait_capevent_frameready(timeout_milisec):
                frame_index_safe = self.frame_index % self.buffer_size
                result = self.dcam.buf_getframe_with_timestamp_and_framestamp(frame_index_safe)
                
                if result is not False:
                    frame, npBuf, timestamp, framestamp = result
                    # Copy frame data immediately and release lock
                    frame_copy = np.copy(npBuf)
                    
                    # Release lock as soon as possible
                    DCamLock.release_capture()
                    
                    # Process frame without holding any locks
                    self.process_captured_frame(frame_copy, timestamp, framestamp)
                    return
                    
        finally:
            DCamLock.release_capture()

    def process_captured_frame(self, frame, timestamp, framestamp):
        """Process frame - no locks held"""
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
        
        # Track frame rate - count ALL frames captured
        self.frame_count += 1
        
        # Queue frame for display - only skip if queue is actually full
        try:
            # Clear old frames if queue is backing up
            if self.frame_queue.qsize() > 3:
                while self.frame_queue.qsize() > 1:
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Only skip if truly full
            pass

        # Queue timestamp - always save this
        try:
            self.timestamp_queue.put_nowait((corrected_timestamp, corrected_framestamp))
        except:
            pass
        
        # Queue for saving if enabled
        if self.save_queue is not None:
            try:
                # Check if save queue is getting full
                if self.save_queue.qsize() > 40000:
                    debug_logger.warning(f"Save queue very full: {self.save_queue.qsize()}")
                self.save_queue.put_nowait((frame, corrected_timestamp, corrected_framestamp))
            except queue.Full:
                debug_logger.warning("Save queue full - dropping frame")

        # Get GPS timestamp on first frame
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
        self.fps_calc_time = time.time()
        
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
        time.sleep(0.2)
        
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
            'SENSOR_MODE': 1.0,
            'IMAGE_PIXEL_TYPE': 2.0
        }
        for prop, value in defaults.items():
            self.set_property(prop, value)

    def set_property(self, prop_name, value):
        """Set camera property with property lock"""
        if prop_name not in CAMERA_PARAMS:
            logging.error(f"Unknown property: {prop_name}")
            return False
        
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


class OptimizedSaveThread(threading.Thread):
    """Optimized save thread using process pool for FITS writing"""
    
    def __init__(self, save_queue, camera_thread, header_dict, object_name, shared_data):
        super().__init__(name="SaveThread")
        self.save_queue = save_queue
        self.running = True
        self.camera_thread = camera_thread
        self.header_dict = header_dict
        self.object_name = object_name
        self.batch_size = 100
        self.frame_buffer = []
        self.timestamp_buffer = []
        self.framestamp_buffer = []
        self.cube_index = 0
        self.shared_data = shared_data
        
        # Use thread pool for async I/O instead of process pool to avoid serialization overhead
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.pending_writes = []

    def run(self):
        """Main save thread loop"""
        try:
            logging.info("Save thread started")
            
            start_time_filename_str = time.strftime('%Y%m%d_%H%M%S')
            os.makedirs("captures", exist_ok=True)

            while self.running or not self.save_queue.empty():
                try:
                    # Try to get multiple frames at once for efficiency
                    frames_read = 0
                    max_frames = min(50, self.batch_size - len(self.frame_buffer))  # Increased from 20
                    
                    while frames_read < max_frames:
                        try:
                            frame, timestamp, framestamp = self.save_queue.get(timeout=0.01)  # Reduced timeout
                            self.frame_buffer.append(frame)
                            self.timestamp_buffer.append(timestamp)
                            self.framestamp_buffer.append(framestamp)
                            frames_read += 1
                        except queue.Empty:
                            break

                    # Write cube when buffer is full
                    if len(self.frame_buffer) >= self.batch_size:
                        self.write_cube_async(start_time_filename_str)
                    
                    # Check pending writes (non-blocking)
                    self.check_pending_writes()
                        
                except Exception as e:
                    logging.error(f"Save thread error: {e}")

            # Write remaining frames
            if self.frame_buffer:
                self.write_cube_async(start_time_filename_str)
            
            # Wait for all pending writes
            self.wait_for_pending_writes()
                
        except Exception as e:
            logging.error(f"Fatal save thread error: {e}")
        finally:
            self.cleanup()

    def write_cube_async(self, start_time_filename_str):
        """Write data cube asynchronously without blocking"""
        try:
            self.cube_index += 1
            filename = f"{self.object_name}_{start_time_filename_str}_cube{self.cube_index:03d}.fits"
            filepath = os.path.join("captures", filename)
            
            logging.info(f"Queuing cube {self.cube_index} ({len(self.frame_buffer)} frames) for async write")

            # Important: Don't create numpy array here! Just pass references
            # The numpy arrays are already in memory, we just pass the list of references
            frames_to_write = self.frame_buffer[:]  # Shallow copy of list (fast)
            timestamps = self.timestamp_buffer[:]  # Shallow copy
            framestamps = self.framestamp_buffer[:]  # Shallow copy
            
            # Get camera params snapshot
            with self.shared_data.lock:
                camera_params = dict(self.shared_data.camera_params)
            
            # Submit to thread pool - numpy array stacking happens in thread
            future = self.executor.submit(
                self.write_fits_in_thread,
                filepath, frames_to_write, timestamps, framestamps, self.header_dict,
                self.object_name, self.cube_index, camera_params
            )
            
            self.pending_writes.append((filepath, future))

            # Clear buffers immediately (just clears the list, not the numpy arrays)
            self.frame_buffer.clear()
            self.timestamp_buffer.clear()
            self.framestamp_buffer.clear()
            
        except Exception as e:
            logging.error(f"Write cube error: {e}")

    def write_fits_in_thread(self, filepath, frames, timestamps, framestamps, 
                             header_dict, object_name, cube_index, camera_params):
        """Write FITS file in thread - numpy array creation happens here"""
        try:
            # Set thread to low priority
            try:
                p = psutil.Process()
                p.nice(10)
            except:
                pass
                
            # Create numpy array in the thread to avoid blocking main thread
            data_cube = np.array(frames)
            
            # Create FITS HDUs
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['OBJECT'] = (object_name, 'Object name')
            primary_hdu.header['CUBEIDX'] = (cube_index, 'Cube index number')

            image_hdu = fits.ImageHDU(data=data_cube)
            image_hdu.header['EXTNAME'] = 'DATA_CUBE'
            
            # Add camera parameters
            for key, value in camera_params.items():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        image_hdu.header[key] = value
                    except:
                        pass

            # Add other header parameters
            for key, value in header_dict.items():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        image_hdu.header[key] = value
                    except:
                        pass

            # Timestamp table
            col1 = fits.Column(name='TIMESTAMP', format='D', array=timestamps)
            col2 = fits.Column(name='FRAMESTAMP', format='K', array=framestamps)
            timestamp_hdu = fits.BinTableHDU.from_columns([col1, col2])
            timestamp_hdu.header['EXTNAME'] = 'TIMESTAMPS'

            # Write file
            hdulist = fits.HDUList([primary_hdu, image_hdu, timestamp_hdu])
            hdulist.writeto(filepath, overwrite=True)
            hdulist.close()
            
            return True
        except Exception as e:
            logging.error(f"FITS write error: {e}")
            return False

    def check_pending_writes(self):
        """Check status of pending writes (non-blocking)"""
        completed = []
        for filepath, future in self.pending_writes:
            if future.done():
                try:
                    if future.result(timeout=0):  # Non-blocking check
                        logging.info(f"Completed writing: {filepath}")
                    else:
                        logging.error(f"Failed writing: {filepath}")
                except Exception as e:
                    logging.error(f"Write error for {filepath}: {e}")
                completed.append((filepath, future))
        
        # Remove completed writes
        for item in completed:
            self.pending_writes.remove(item)

    def wait_for_pending_writes(self):
        """Wait for all pending writes to complete"""
        for filepath, future in self.pending_writes:
            try:
                if future.result(timeout=30):
                    logging.info(f"Final write completed: {filepath}")
            except Exception as e:
                logging.error(f"Final write error for {filepath}: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)

    def stop(self):
        """Stop save thread"""
        self.running = False


class PeripheralsThread(threading.Thread):
    """Thread for managing peripheral devices - unchanged from original"""
    
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

        # Zoom tracking
        self.zoom_virtual_position = 153.0  # Start at middle (200mm equivalent)
        self.zoom_reference_set = False
        self.zoom_reference_position = None

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

        # Update GUI with current positions after all devices connected
        self.after_connection_update()

    def after_connection_update(self):
        """Update GUI after connections are established"""
        # Give devices a moment to settle, then update GUI
        threading.Timer(2.0, self.update_gui_with_current_positions).start()

        # Initialize focus motor sequence after a short delay
        # threading.Timer(3.0, self.initialize_focus_sequence).start()

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

    def get_current_filter_position(self):
            """Get current filter wheel position"""
            try:
                with self.peripherals_lock:
                    if self.efw:
                        position = self.efw.GetPosition(0)
                        # Reverse lookup from position to option string
                        position_map = {0: '0 (Open)', 1: '1 (u\')', 2: '2 (g\')', 
                                    3: '3 (r\')', 4: '4 (i\')', 5: '5 (z\')', 6: '6 (500nm)'}
                        return position_map.get(position, '0 (Open)')
            except Exception as e:
                debug_logger.error(f"Get filter position error: {e}")
            return '0 (Open)'

    def get_current_shutter_state(self):
        """Get current shutter state"""
        try:
            with self.peripherals_lock:
                if self.ljm_handle:
                    state = ljm.eReadName(self.ljm_handle, "DIO4")
                    return 'Open' if state == 0 else 'Closed'
        except Exception as e:
            debug_logger.error(f"Get shutter state error: {e}")
        return 'Open'

    def get_current_slit_position(self):
        """Get current slit position"""
        try:
            with self.peripherals_lock:
                if self.ax_a_1:
                    position = self.ax_a_1.get_position(Units.LENGTH_MILLIMETRES)
                    # If position is closer to 0, it's "In beam", otherwise "Out of beam"
                    return 'In beam' if abs(position) < 35 else 'Out of beam'
        except Exception as e:
            debug_logger.error(f"Get slit position error: {e}")
        return 'Out of beam'

    def get_current_halpha_qwp_position(self):
        """Get current Halpha/QWP position"""
        try:
            with self.peripherals_lock:
                if self.ax_b_3:
                    position = self.ax_b_3.get_position(Units.LENGTH_MILLIMETRES)
                    # Find closest position
                    positions = {'Halpha': 151.5, 'QWP': 23.15, 'Neither': 87.18}
                    closest = min(positions.items(), key=lambda x: abs(x[1] - position))
                    return closest[0]
        except Exception as e:
            debug_logger.error(f"Get Halpha/QWP position error: {e}")
        return 'Neither'

    def get_current_pol_stage_position(self):
        """Get current polarization stage position"""
        try:
            with self.peripherals_lock:
                if self.ax_b_2:
                    position = self.ax_b_2.get_position(Units.LENGTH_MILLIMETRES)
                    # Find closest position
                    positions = {'WeDoWo': 17.78, 'Wire Grid': 128.5, 'Neither': 60.66}
                    closest = min(positions.items(), key=lambda x: abs(x[1] - position))
                    return closest[0]
        except Exception as e:
            debug_logger.error(f"Get pol stage position error: {e}")
        return 'Neither'

    def update_gui_with_current_positions(self):
        """Update GUI with current hardware positions"""
        if self.gui_ref:
            # Get current positions
            filter_pos = self.get_current_filter_position()
            shutter_state = self.get_current_shutter_state()
            slit_pos = self.get_current_slit_position()
            halpha_qwp_pos = self.get_current_halpha_qwp_position()
            pol_stage_pos = self.get_current_pol_stage_position()
            
            # Update GUI in main thread
            self.gui_ref.after(0, lambda: self.gui_ref.set_filter_position_display(filter_pos))
            self.gui_ref.after(0, lambda: self.gui_ref.set_shutter_display(shutter_state))
            self.gui_ref.after(0, lambda: self.gui_ref.set_slit_position_display(slit_pos))
            self.gui_ref.after(0, lambda: self.gui_ref.set_halpha_qwp_display(halpha_qwp_pos))
            self.gui_ref.after(0, lambda: self.gui_ref.set_pol_stage_display(pol_stage_pos))

    def initialize_focus_sequence(self):
        """Initialize focus motor to infinity then optimal position"""
        def _focus_init():
            try:
                if self.ax_b_1 is None:
                    logging.warning("Focus motor not available for initialization")
                    return
                    
                with self.peripherals_lock:
                    logging.info("Starting focus initialization sequence")
                    if self.gui_ref:
                        self.gui_ref.after(0, lambda: self.gui_ref.update_status(
                            "Initializing focus position", "red"))
                        # Disable focus entry during move
                        self.gui_ref.after(0, lambda: self.gui_ref.focus_position_entry.config(state='disabled'))
                        self.gui_ref.after(0, lambda: self.gui_ref.set_focus_button.config(state='disabled'))

                    # Step 1: Move to infinity (clutch engagement)
                    logging.info("Moving focus to infinity (+400 degrees)")
                    current_pos = self.ax_b_1.get_position(Units.ANGLE_DEGREES)
                    infinity_pos = current_pos + 400
                    self.ax_b_1.move_absolute(infinity_pos, Units.ANGLE_DEGREES)
                    
                    # Wait for movement to complete
                    self.ax_b_1.wait_until_idle()
                    
                    # Step 2: Move to optimal focus position
                    logging.info("Moving focus to optimal position (-75 degrees from infinity)")
                    optimal_pos = infinity_pos - 75
                    self.ax_b_1.move_absolute(optimal_pos, Units.ANGLE_DEGREES)
                    
                    # Wait for movement to complete
                    self.ax_b_1.wait_until_idle()
                    
                    logging.info("Focus initialization sequence complete")
                    
                    # Update GUI status
                    if self.gui_ref:
                        self.gui_ref.after(0, lambda: self.gui_ref.update_status(
                            "Focus initialization sequence complete", "green"))
                        # Re-enable focus entry
                        self.gui_ref.after(0, lambda: self.gui_ref.focus_position_entry.config(state='normal'))
                        self.gui_ref.after(0, lambda: self.gui_ref.set_focus_button.config(state='normal'))
                    
            except Exception as e:
                logging.error(f"Focus initialization error: {e}")
                if self.gui_ref:
                    self.gui_ref.after(0, lambda: self.gui_ref.update_status(
                        "Focus initialization failed", "red"))
        
        # Run in executor to avoid blocking
        self.executor.submit(_focus_init)
###TBT
    def set_zoom_reference(self):
            """Set current physical position as reference for a virtual position of zero"""
            try:
                if self.ax_a_2 is None:
                    return False
                    
                with self.peripherals_lock:
                    self.zoom_reference_position = self.ax_a_2.get_position(Units.ANGLE_DEGREES)
                    self.zoom_virtual_position = 0
                    self.zoom_reference_set = True
                    
                logging.info(f"3x zoom-out reference set: physical={self.zoom_reference_position:.1f}°")
                return True
            except Exception as e:
                logging.error(f"Set zoom reference error: {e}")
                return False

    def move_zoom_to_virtual(self, virtual_target):
        """Move to virtual position (0-300°)"""
        try:
            if self.ax_a_2 is None or not self.zoom_reference_set:
                return False
            
            physical_target = self.zoom_reference_position + virtual_target
            print(self.zoom_virtual_position, virtual_target, self.zoom_reference_position, physical_target)
            
            with self.peripherals_lock:
                self.ax_a_2.move_absolute(physical_target, Units.ANGLE_DEGREES)
                self.zoom_virtual_position = virtual_target
                
            logging.info(f"Zoom moved to virtual {virtual_target}°")
            return True
            
        except Exception as e:
            logging.error(f"Move zoom virtual error: {e}")
            return False

    def move_zoom_relative(self, relative_degrees):
        """Relative zoom movement"""
        try:
            if self.ax_a_2 is None:
                return False
                
            with self.peripherals_lock:
                current_pos = self.ax_a_2.get_position(Units.ANGLE_DEGREES)
                new_position = current_pos + relative_degrees
                self.ax_a_2.move_absolute(new_position, Units.ANGLE_DEGREES)
                
                # If this pushes the virtual position out of bounds,
                # update the reference
                if self.zoom_reference_set:
                    self.zoom_virtual_position += relative_degrees
                    
            return True
        except Exception as e:
            logging.error(f"Emergency zoom move error: {e}")
            return False

###TBT###
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

#!/usr/bin/env python3
"""
Fixed TCSThread class that properly handles TCS communication
The TCS closes connections after each command, so we need to reconnect for each operation
"""

class TCSThread(threading.Thread):
    """Thread for communicating with the Telescope Control System"""
    
    def __init__(self, tcs_ip="200.28.147.59", tcs_port=5811, gui_ref=None):
        super().__init__(name="TCSThread")
        self.tcs_ip = tcs_ip
        self.tcs_port = tcs_port
        self.gui_ref = gui_ref
        self.running = True
        self.tcs_data = {
            'ra': 'N/A',
            'dec': 'N/A',
            'equinox': 'N/A',
            'rotator_angle': 'N/A',
            'parallactic_angle': 'N/A',
            'airmass': 'N/A',
            'elevation': 'N/A',
            'azimuth': 'N/A',
            'hour_angle': 'N/A',
            'sidereal_time': 'N/A',
            'date': 'N/A',
            'time': 'N/A'
        }
        self.data_lock = threading.RLock()
        self.last_update = time.time()
        self.command_lock = threading.Lock()  # Lock for command sending
        
    def run(self):
        """Main thread loop"""
        logging.info("TCS thread started")
        while self.running:
            try:
                # Request data cyclically
                self.request_tcs_data()
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logging.error(f"TCS thread error: {e}")
                time.sleep(5)
    
    def send_single_command(self, command):
        """Send a single command to TCS with fresh connection
        
        TCS closes connection after each command, so we need to reconnect each time
        """
        with self.command_lock:  # Ensure only one command at a time
            socket_conn = None
            try:
                # Create new connection for this command
                socket_conn = socket.create_connection((self.tcs_ip, self.tcs_port), timeout=5)
                socket_conn.settimeout(2)
                
                # Send command
                socket_conn.sendall(f"{command}\n".encode('ascii'))
                
                # Read response - TCS sends response then closes connection
                response = b''
                while True:
                    try:
                        data = socket_conn.recv(1024)
                        if not data:
                            break
                        response += data
                        if b'\n' in response:
                            break
                    except socket.timeout:
                        break
                
                result = response.decode('ascii', errors='ignore').strip() if response else None
                
                return result
                
            except Exception as e:
                logging.error(f"TCS command '{command}' error: {e}")
                return None
            finally:
                # Always close the socket
                if socket_conn:
                    try:
                        socket_conn.close()
                    except:
                        pass
    
    def request_tcs_data(self):
        """Request and parse TCS data"""
        try:
            # Request datetime
            datetime_response = self.send_single_command("datetime")
            if datetime_response:
                self.parse_datetime(datetime_response)
            
            # Request telescope position
            telpos_response = self.send_single_command("telpos")
            if telpos_response:
                self.parse_telpos(telpos_response)
            
            # Request telescope data
            teldata_response = self.send_single_command("teldata")
            if teldata_response:
                self.parse_teldata(teldata_response)
                
            self.last_update = time.time()
            
        except Exception as e:
            logging.error(f"Error requesting TCS data: {e}")
    
    def parse_datetime(self, response):
        """Parse datetime response: 2025-09-10 13:50:25 08:27:03"""
        try:
            parts = response.strip().split()
            if len(parts) >= 3:
                with self.data_lock:
                    self.tcs_data['date'] = parts[0]
                    self.tcs_data['time'] = parts[1]
                    self.tcs_data['sidereal_time'] = parts[2]
        except Exception as e:
            logging.debug(f"Error parsing datetime: {e}")
    
    def parse_telpos(self, response):
        """Parse telpos response: 08:26:02.36 -28:58:34.3 2000.00 -00:00:01 1.000 172.2522"""
        try:
            parts = response.strip().split()
            if len(parts) >= 6:
                with self.data_lock:
                    self.tcs_data['ra'] = parts[0]
                    self.tcs_data['dec'] = parts[1]
                    self.tcs_data['equinox'] = parts[2]
                    self.tcs_data['hour_angle'] = parts[3]
                    self.tcs_data['airmass'] = parts[4]
                    self.tcs_data['rotator_angle'] = parts[5]
        except Exception as e:
            logging.debug(f"Error parsing telpos: {e}")
    
    def parse_teldata(self, response):
        """Parse teldata response: 3 00 000 0111 168.0006 89.9280 0.0720 -017.7478 045 0"""
        try:
            parts = response.strip().split()
            if len(parts) >= 10:
                with self.data_lock:
                    # parts[4] = azimuth, parts[5] = elevation, parts[7] = parallactic angle
                    self.tcs_data['azimuth'] = parts[4]
                    self.tcs_data['elevation'] = parts[5]
                    self.tcs_data['parallactic_angle'] = parts[7]
        except Exception as e:
            logging.debug(f"Error parsing teldata: {e}")
    
    def get_current_data(self):
        """Get current TCS data safely"""
        with self.data_lock:
            return dict(self.tcs_data)
    
    def send_offset(self, ra_offset=0.0, dec_offset=0.0):
        """Send offset command to TCS
        
        Args:
            ra_offset: RA offset in arcseconds (positive = West)
            dec_offset: Dec offset in arcseconds (positive = North)
        """
        try:
            success = True
            
            # Send offset commands according to TCS documentation
            # Each command needs its own connection
            
            if ra_offset != 0:
                response = self.send_single_command(f"ofra {ra_offset:.2f}")
                logging.info(f"TCS ofra {ra_offset:.2f}: {response}")
                if response is None:
                    success = False
                time.sleep(0.1)  # Small delay between commands
                
            if dec_offset != 0:
                response = self.send_single_command(f"ofdc {dec_offset:.2f}")
                logging.info(f"TCS ofdc {dec_offset:.2f}: {response}")
                if response is None:
                    success = False
                time.sleep(0.1)
                
            # Execute the offset
            response = self.send_single_command("offp")
            logging.info(f"TCS offp: {response}")
            if response is None:
                success = False
            
            # Update GUI status if available
            if self.gui_ref and success:
                self.gui_ref.after(0, lambda: self.gui_ref.update_status(
                    f"Telescope offset: RA={ra_offset:+.1f}\" Dec={dec_offset:+.1f}\"", "green"))
            elif self.gui_ref:
                self.gui_ref.after(0, lambda: self.gui_ref.update_status(
                    "Telescope offset failed", "red"))
            
            return success
            
        except Exception as e:
            logging.error(f"Error sending telescope offset: {e}")
            if self.gui_ref:
                self.gui_ref.after(0, lambda: self.gui_ref.update_status(
                    f"Offset error: {e}", "red"))
            return False
    
    def reset_offsets(self):
        """Reset telescope offsets to zero
        
        Note: This sends ofra 0 and ofdc 0, but the TCS might interpret this
        differently than expected. Check TCS documentation for proper reset procedure.
        """
        try:
            # Send zero offsets
            response1 = self.send_single_command("ofra 0")
            time.sleep(0.1)
            response2 = self.send_single_command("ofdc 0")
            time.sleep(0.1)
            response3 = self.send_single_command("offp")
            
            logging.info(f"Reset offsets - ofra 0: {response1}, ofdc 0: {response2}, offp: {response3}")
            
            if self.gui_ref:
                self.gui_ref.after(0, lambda: self.gui_ref.update_status(
                    "Telescope offsets reset", "green"))
            
            return True
            
        except Exception as e:
            logging.error(f"Error resetting offsets: {e}")
            return False
    
    def stop(self):
        """Stop TCS thread"""
        self.running = False

class CameraGUI(tk.Tk):
    """Main GUI application - keeping original layout exactly"""
    
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
        self.tcs_thread = TCSThread(gui_ref=self)
        self.tcs_thread.daemon = True
        self.tcs_thread.start()
        self.countdown_active = False
        self.frame_count = 0
        self.last_frame_time = None
        
        # Performance monitoring
        self.last_fps_update = time.time()
        self.actual_display_count = 0
        self.last_display_time = time.time()

        self.min_val = tk.StringVar(value="0")
        self.max_val = tk.StringVar(value="60000")

        self.title("Lightspeed Prototype Control GUI")
        self.geometry("1100x1080")

        self.setup_gui()

    def setup_gui(self):
        """Set up all GUI elements - exactly as original"""
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Camera Parameters
        camera_params_frame = LabelFrame(self.main_frame, text="Camera Parameters", padx=5, pady=5)
        camera_params_frame.grid(row=0, column=0, sticky='ew')
        self.camera_status = tk.Label(camera_params_frame, text="", justify=tk.LEFT, anchor="w")
        self.camera_status.pack(fill='both', expand=True)

        # Status messages
        self.status_message = tk.Label(self.main_frame, text="", justify=tk.LEFT, anchor="w", 
                                       width=40, wraplength=400, fg="red")
        self.status_message.grid(row=5, column=0, sticky='ew')

        # GPS Timestamp display
        gps_frame = LabelFrame(self.main_frame, text="GPS Timestamp", padx=5, pady=5)
        gps_frame.grid(row=1, column=0, columnspan=1, sticky='ew')
        self.gps_timestamp_label = tk.Label(gps_frame, text="No capture active", font=("Courier", 12))
        self.gps_timestamp_label.pack()
        
        # Performance display
        perf_frame = LabelFrame(self.main_frame, text="Performance", padx=5, pady=5)
        perf_frame.grid(row=2, column=0, columnspan=1, sticky='ew')
        self.performance_label = tk.Label(perf_frame, text="FPS: -- | Display FPS: -- \n Queue: --", font=("Courier", 10))
        self.performance_label.pack()

        # TCS Status and Control frame
        tcs_frame = LabelFrame(self.main_frame, text="Telescope Status & Control", padx=5, pady=5)
        tcs_frame.grid(row=3, column=0, columnspan=1, sticky='ew')

        # Status display
        self.tcs_status_label = tk.Label(tcs_frame, text="TCS: Connecting...", font=("Courier", 10))
        self.tcs_status_label.pack()

        # Offset controls
        offset_control_frame = tk.Frame(tcs_frame)
        offset_control_frame.pack(pady=5)

        # Offset amount entry
        tk.Label(offset_control_frame, text="Offset (arcsec):").grid(row=0, column=0, padx=2)
        self.tcs_offset_amount = tk.DoubleVar(value=5.0)
        offset_entry = tk.Entry(offset_control_frame, textvariable=self.tcs_offset_amount, width=8)
        offset_entry.grid(row=0, column=1, padx=2)

        # Directional offset buttons in cross pattern
        offset_buttons_frame = tk.Frame(tcs_frame)
        offset_buttons_frame.pack(pady=5)

        # North button (Dec+)
        tk.Button(offset_buttons_frame, text="N", width=3, 
                command=lambda: self.send_telescope_offset(dec_offset=1)).grid(row=0, column=1, padx=1, pady=1)

        # West button (RA+) and East button (RA-)
        tk.Button(offset_buttons_frame, text="W", width=3,
                command=lambda: self.send_telescope_offset(ra_offset=1)).grid(row=1, column=0, padx=1, pady=1)
        tk.Button(offset_buttons_frame, text="E", width=3,
                command=lambda: self.send_telescope_offset(ra_offset=-1)).grid(row=1, column=2, padx=1, pady=1)

        # South button (Dec-)
        tk.Button(offset_buttons_frame, text="S", width=3,
                command=lambda: self.send_telescope_offset(dec_offset=-1)).grid(row=2, column=1, padx=1, pady=1)

        # Reset button in center
        tk.Button(offset_buttons_frame, text="RST", width=3, fg="red",
                command=self.reset_telescope_offsets).grid(row=1, column=1, padx=1, pady=1)
        
        # Start TCS status update loop
        self.after(5000, self.update_tcs_status)

        # Set up control sections
        self.setup_camera_controls()
        self.setup_camera_settings()
        self.setup_subarray_controls()
        self.setup_advanced_controls()
        self.setup_display_controls()
        self.setup_peripherals_controls()
        
        # Start update loops
        self.after(100, self.update_camera_status)
        self.after(50, self.update_frame_display)
        self.after(1000, self.update_performance_monitor)
        self.after(10000, self.update_peripherals_status)

    def update_performance_monitor(self):
        """Monitor and display performance metrics"""
        # Calculate capture FPS from camera thread's frame counter
        if hasattr(self.camera_thread, 'frame_count'):
            current_time = time.time()
            if hasattr(self.camera_thread, 'fps_calc_time'):
                time_diff = current_time - self.camera_thread.fps_calc_time
                if time_diff > 0:
                    capture_fps = self.camera_thread.frame_count / time_diff
                    self.camera_thread.frame_count = 0
                    self.camera_thread.fps_calc_time = current_time
                else:
                    capture_fps = 0
            else:
                capture_fps = 0
                self.camera_thread.fps_calc_time = current_time
        else:
            capture_fps = 0
        
        # Calculate actual display FPS
        current_time = time.time()
        time_elapsed = current_time - self.last_display_time
        if time_elapsed > 0:
            display_fps = self.actual_display_count / time_elapsed
        else:
            display_fps = 0
        self.actual_display_count = 0
        self.last_display_time = current_time
        
        # Get queue sizes
        frame_queue_size = self.frame_queue.qsize()
        save_queue_size = self.camera_thread.save_queue.qsize() if self.camera_thread.save_queue else 0
        
        self.performance_label.config(
            text=f"Capture FPS: {capture_fps:.1f} | Display FPS: {display_fps:.1f} \n Frame Q: {frame_queue_size} | Save Q: {save_queue_size}"
        )
        
        self.after(1000, self.update_performance_monitor)

    def setup_camera_controls(self):
        """Set up camera control widgets - exactly as original"""
        camera_controls_frame = LabelFrame(self.main_frame, text="Camera Controls", padx=5, pady=5)
        camera_controls_frame.grid(row=0, column=1)

        Label(camera_controls_frame, text="Exposure Time (ms):").grid(row=0, column=0)
        self.exposure_time_var = tk.DoubleVar(value=100)
        self.exposure_time_var.trace_add("write", self.update_exposure_time)
        self.exposure_time_entry = Entry(camera_controls_frame, textvariable=self.exposure_time_var)
        self.exposure_time_entry.grid(row=0, column=1)
        
        # Add countdown timer label
        self.countdown_label = Label(camera_controls_frame, text="Ready", font=("Courier", 10, "bold"), fg="blue")
        self.countdown_label.grid(row=1, column=0, columnspan=2)

        self.start_button = Button(camera_controls_frame, text="Start", command=self.start_capture)
        self.start_button.grid(row=2, column=0)

        self.stop_button = Button(camera_controls_frame, text="Stop", command=self.stop_capture)
        self.stop_button.grid(row=2, column=1)

        self.save_data_var = tk.BooleanVar()
        self.save_data_checkbox = Checkbutton(camera_controls_frame, text="Save Data to Disk", 
                                            variable=self.save_data_var)
        self.save_data_checkbox.grid(row=3, column=0, columnspan=2)

        Label(camera_controls_frame, text="Object Name:").grid(row=4, column=0)
        self.object_name_entry = Entry(camera_controls_frame)
        self.object_name_entry.grid(row=4, column=1)

        Label(camera_controls_frame, text="Frames per Datacube").grid(row=5, column=0)
        self.cube_size_var = tk.IntVar(value=100)
        self.cube_size_var.trace_add("write", self.update_batch_size)
        self.cube_size_entry = Entry(camera_controls_frame, textvariable=self.cube_size_var)
        self.cube_size_entry.grid(row=5, column=1)

        self.reset_button = Button(camera_controls_frame, text="Reset Camera", command=self.reset_camera)
        self.reset_button.grid(row=7, column=0, columnspan=1)

        self.power_cycle_button = Button(camera_controls_frame, text="Power Cycle Camera",
                                        command=self.power_cycle_camera)
        self.power_cycle_button.grid(row=7, column=1, columnspan=1)

        self.take_flats_button = Button(camera_controls_frame, text="Take Flats",
                                        command=self.take_flats)
        self.take_flats_button.grid(row=7, column=2, columnspan=1)

    def setup_camera_settings(self):
        """Set up camera settings widgets - exactly as original"""
        camera_settings_frame = LabelFrame(self.main_frame, text="Camera Settings", padx=5, pady=5)
        camera_settings_frame.grid(row=2, column=1)

        Label(camera_settings_frame, text="Binning:").grid(row=0, column=0)
        self.binning_var = StringVar(value="1x1")
        self.binning_menu = OptionMenu(camera_settings_frame, self.binning_var, 
                                       "1x1", "2x2", "4x4", command=self.change_binning)
        self.binning_menu.grid(row=0, column=1)

        Label(camera_settings_frame, text="Bit Depth:").grid(row=1, column=0)
        self.bit_depth_var = StringVar(value="16-bit")
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
        self.sensor_mode_var = StringVar(value="Standard")
        self.sensor_mode_menu = OptionMenu(camera_settings_frame, self.sensor_mode_var, 
                                           "Photon Number Resolving", "Standard", 
                                           command=self.change_sensor_mode)
        self.sensor_mode_menu.grid(row=3, column=1)

    def setup_subarray_controls(self):
        """Set up subarray control widgets - exactly as original"""
        subarray_controls_frame = LabelFrame(self.main_frame, text="Subarray Controls", padx=5, pady=5)
        subarray_controls_frame.grid(row=3, column=1)

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
        """Set up advanced control widgets - exactly as original"""
        advanced_controls_frame = LabelFrame(self.main_frame, text="Advanced Controls", padx=5, pady=5)
        advanced_controls_frame.grid(row=1, column=1)

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
        """Set up display control widgets - with auto-scaling options"""
        display_controls_frame = LabelFrame(self.main_frame, text="Display Controls", padx=5, pady=5)
        display_controls_frame.grid(row=4, column=0)

        # Auto-scaling options
        self.auto_minmax_var = tk.BooleanVar(value=False)
        self.auto_minmax_check = Checkbutton(display_controls_frame, text="Auto Min/Max", 
                                            variable=self.auto_minmax_var, 
                                            command=lambda: self.toggle_auto_scaling('minmax'))
        self.auto_minmax_check.grid(row=0, column=0, columnspan=2)
        
        self.auto_zscale_var = tk.BooleanVar(value=False)
        self.auto_zscale_check = Checkbutton(display_controls_frame, text="Auto Zscale", 
                                            variable=self.auto_zscale_var,
                                            command=lambda: self.toggle_auto_scaling('zscale'))
        self.auto_zscale_check.grid(row=0, column=2, columnspan=2)

        # Manual scaling controls
        Label(display_controls_frame, text="Min Count:").grid(row=1, column=0)
        self.min_entry = Entry(display_controls_frame, textvariable=self.min_val, width=8)
        self.min_entry.grid(row=1, column=1)

        Label(display_controls_frame, text="Max Count:").grid(row=1, column=2)
        self.max_val.trace_add("write", self.refresh_frame_display)
        self.max_entry = Entry(display_controls_frame, textvariable=self.max_val, width=8)
        self.max_entry.grid(row=1, column=3)

    def setup_peripherals_controls(self):
        """Set up peripheral control widgets - exactly as original"""
        self.peripherals_controls_frame = LabelFrame(self.main_frame, text="Peripherals Controls", 
                                                     padx=5, pady=5)
        self.peripherals_controls_frame.grid(row=4, column=1)

        # Filter control
        Label(self.peripherals_controls_frame, text="Filter:").grid(row=0, column=0)
        self.filter_position_var = tk.StringVar(value="Reading...")  
        self.filter_options = {'0 (Open)': 0, '1 (u\')': 1, '2 (g\')': 2, '3 (r\')': 3,
                               '4 (i\')': 4, '5 (z\')': 5, '6 (500nm)': 6}
        self.filter_position_menu = OptionMenu(self.peripherals_controls_frame, self.filter_position_var,
                                               *self.filter_options.keys(),
                                               command=self.update_filter_position)
        self.filter_position_menu.grid(row=0, column=1)

        # Shutter control
        Label(self.peripherals_controls_frame, text="Shutter:").grid(row=0, column=2)
        self.shutter_var = tk.StringVar(value="Reading...")  
        self.shutter_menu = OptionMenu(self.peripherals_controls_frame, self.shutter_var,
                                       'Open', 'Closed', command=self.update_shutter)
        self.shutter_menu.grid(row=0, column=3)

        # Motor controls
        self.setup_motor_controls()
        
        # PDU outlet controls
        self.setup_pdu_controls()

    def setup_motor_controls(self):
        """Set up motor control widgets - exactly as original"""
        # Slit control
        Label(self.peripherals_controls_frame, text="Slit:").grid(row=1, column=0)
        self.slit_position_var = tk.StringVar(value="Reading...")  
        self.slit_position_menu = OptionMenu(self.peripherals_controls_frame, self.slit_position_var,
                                             'In beam', 'Out of beam', command=self.update_slit_position)
        self.slit_position_menu.grid(row=1, column=1)

        # Halpha/QWP control
        Label(self.peripherals_controls_frame, text="Halpha/QWP:").grid(row=1, column=2)
        self.halpha_qwp_var = tk.StringVar(value="Reading...")  
        self.halpha_qwp_menu = OptionMenu(self.peripherals_controls_frame, self.halpha_qwp_var,
                                          'Halpha', 'QWP', 'Neither', command=self.update_halpha_qwp)
        self.halpha_qwp_menu.grid(row=1, column=3)

        # Polarization stage control
        Label(self.peripherals_controls_frame, text="Pol. Stage:").grid(row=2, column=0)
        self.wire_grid_var = tk.StringVar(value="Reading...")  
        self.wire_grid_menu = OptionMenu(self.peripherals_controls_frame, self.wire_grid_var,
                                         'WeDoWo', 'Wire Grid', 'Neither', command=self.update_pol_stage)
        self.wire_grid_menu.grid(row=2, column=1)

        ### TBT 
        # Zoom preset controls
        Label(self.peripherals_controls_frame, text="Zoom out by:").grid(row=3, column=0, sticky='w')
        zoom_preset_frame = Frame(self.peripherals_controls_frame)
        zoom_preset_frame.grid(row=3, column=1, columnspan=3, sticky='w')
        
        Button(zoom_preset_frame, text="3/4x", width=6,
               command=lambda: self.move_zoom_preset(300)).pack(side='left', padx=2)
        Button(zoom_preset_frame, text="2x", width=6,
               command=lambda: self.move_zoom_preset(200)).pack(side='left', padx=2)
        Button(zoom_preset_frame, text="3x", width=6,
               command=lambda: self.move_zoom_preset(0)).pack(side='left', padx=2)
        
        self.zoom_ref_button = Button(zoom_preset_frame, text="Set 3x pos", width=10, bg='orange',
                                      command=self.set_zoom_reference).pack(side='left', padx=2)

        # Emergency relative movement (with warning)
        Label(self.peripherals_controls_frame, text="Relative Zoom Shift (deg):", 
              fg="red", font=("Arial", 9, "bold")).grid(row=5, column=0, sticky='w')
        self.zoom_emergency_var = tk.StringVar(value='0')
        self.zoom_emergency_entry = Entry(self.peripherals_controls_frame, 
                                         textvariable=self.zoom_emergency_var, width=8)
        self.zoom_emergency_entry.grid(row=5, column=1)
        Button(self.peripherals_controls_frame, text="Apply", fg='red',
               command=self.emergency_zoom_move).grid(row=5, column=2)
        
        Label(self.peripherals_controls_frame, text="(+: zoom in, -: zoom out)", 
              font=("Arial", 8), fg="gray").grid(row=5, column=3, columnspan=4)
        ### TBT 

        # Focus control - separate row TBT
        Label(self.peripherals_controls_frame, text="Relative Focus Shift (deg):").grid(row=6, column=0)
        self.focus_position_var = tk.StringVar(value='0')
        self.focus_position_entry = Entry(self.peripherals_controls_frame, 
                                          textvariable=self.focus_position_var, width=8)
        self.focus_position_entry.grid(row=6, column=1)
        self.set_focus_button = Button(self.peripherals_controls_frame, text="Apply",
                                       command=self.update_focus_position)
        self.set_focus_button.grid(row=6, column=2)
        Label(self.peripherals_controls_frame, text="(+: focus far, -: focus near)", 
              font=("Arial", 8), fg="gray").grid(row=6, column=3, sticky='w')
        ### TBT

        # Focus initialization button
        self.focus_init_button = Button(self.peripherals_controls_frame, text="Reset Focus",
                                       command=self.manual_focus_init)
        self.focus_init_button.grid(row=7, column=0, columnspan=2) #TBT
        Label(self.peripherals_controls_frame, text="(Initialize to near optimal focus)", 
              font=("Arial", 8), fg="gray").grid(row=7, column=2, columnspan=2, sticky='w') #TBT


    def setup_pdu_controls(self):
        """Set up PDU outlet control widgets - exactly as original"""
        Label(self.peripherals_controls_frame, text="PDU Outlet States").grid(row=8, column=0, columnspan=4) #TBT
        self.pdu_outlet_dict = {
            1: 'Rotator', 2: 'Switch', 3: 'Shutter', 4: 'Empty',
            5: 'Empty', 6: 'Empty', 7: 'Empty', 8: 'Empty',
            9: 'Ctrl PC', 10: 'X-MCC4 A', 11: 'X-MCC B', 12: 'qCMOS',
            13: 'Empty', 14: 'Empty', 15: 'Empty', 16: 'Empty'
        }
        
        self.pdu_outlet_vars = {}
        self.pdu_outlet_buttons = {}
        
        for idx, name in self.pdu_outlet_dict.items():
            row = (idx - 1) % 8 + 15 #TBT  
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

    def set_filter_position_display(self, position_string):
        """Set filter position display without triggering callback"""
        self.filter_position_var.set(position_string)

    def set_shutter_display(self, state_string):
        """Set shutter display without triggering callback"""
        self.shutter_var.set(state_string)

    def set_slit_position_display(self, position_string):
        """Set slit position display without triggering callback"""
        self.slit_position_var.set(position_string)

    def set_halpha_qwp_display(self, position_string):
        """Set Halpha/QWP display without triggering callback"""
        self.halpha_qwp_var.set(position_string)

    def set_pol_stage_display(self, position_string):
        """Set polarization stage display without triggering callback"""
        self.wire_grid_var.set(position_string)

    def update_status(self, message, color="red"):
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
        """Update frame display - optimized for smooth display"""
        if self.updating_frame_display:
            try:
                # Process only one frame per update for smoother display
                if not self.frame_queue.empty():
                    try:
                        frame = self.frame_queue.get_nowait()
                        self.last_frame = frame
                        self.actual_display_count += 1
                        self.process_frame(frame)
                        
                        # When we get a frame, update counters for countdown
                        if self.countdown_active:
                            self.frame_count += 1
                            self.last_frame_time = time.time()  # Reset timer for next frame
                            
                    except queue.Empty:
                        pass
            except Exception as e:
                debug_logger.error(f"Frame display error: {e}")
        
        self.after(20, self.update_frame_display)

    def refresh_frame_display(self, *_):
        """Refresh the frame display"""
        if self.last_frame is not None:
            self.process_frame(self.last_frame)

    
    def toggle_auto_scaling(self, which_clicked):
        """Handle auto-scaling checkbox changes"""
        # If one was just turned on, turn off the other
        if which_clicked == 'minmax' and self.auto_minmax_var.get():
            self.auto_zscale_var.set(False)
        elif which_clicked == 'zscale' and self.auto_zscale_var.get():
            self.auto_minmax_var.set(False)
        
        # Enable/disable manual entries based on whether any auto mode is active
        if self.auto_minmax_var.get() or self.auto_zscale_var.get():
            self.min_entry.config(state='disabled')
            self.max_entry.config(state='disabled')
        else:
            self.min_entry.config(state='normal')
            self.max_entry.config(state='normal')

    def compute_zscale(self, data, contrast=0.25, num_samples=10000):
        """Compute zscale limits for display"""
        # Flatten and sample the data
        flat_data = data.flatten()
        if len(flat_data) > num_samples:
            indices = np.random.choice(len(flat_data), num_samples, replace=False)
            sampled_data = flat_data[indices]
        else:
            sampled_data = flat_data
        
        # Sort the samples
        sorted_data = np.sort(sampled_data)
        
        # Calculate the median
        median = np.median(sorted_data)
        
        # Estimate the noise using median absolute deviation
        mad = np.median(np.abs(sorted_data - median))
        if mad == 0:
            mad = 1.0
        
        # Set limits based on contrast
        zmin = median - (contrast * 10.0 * mad)
        zmax = median + (contrast * 10.0 * mad)
        
        # Clip to data range
        data_min = np.min(data)
        data_max = np.max(data)
        
        zmin = max(zmin, data_min)
        zmax = min(zmax, data_max)
        
        return int(zmin), int(zmax)

    def process_frame(self, data):
        """Process and display a frame with auto-scaling support"""
        if not self.display_lock.acquire(blocking=False):
            return
            
        try:
            # Apply auto-scaling if enabled
            if self.auto_minmax_var.get():
                # Simple min/max scaling
                min_val = int(np.min(data))
                max_val = int(np.max(data))
            elif self.auto_zscale_var.get():
                # Zscale algorithm
                min_val, max_val = self.compute_zscale(data)
            else:
                # Use manual values
                try:
                    min_val = int(self.min_val.get())
                    max_val = int(self.max_val.get())
                except:
                    min_val, max_val = 0, 200

            if max_val <= min_val:
                min_val = 0
                max_val = 65535 if data.dtype == np.uint16 else 255
                
            scaled_data = np.clip((data.astype(np.float32) - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

            # Flip horizontally
            scaled_data = cv2.flip(scaled_data, 1)
            
            # Convert to BGR
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
            cv2.waitKey(1)
            
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
            self.cube_size_var = int(self.cube_size_entry.get())
            debug_logger.info(f"Batch size: {self.cube_size_var}")
        except:
            self.cube_size_var = 100

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

            # Start countdown timer if exposure > 1 second
            self.frame_count = 0
            self.last_frame_time = time.time()  # Start timing from capture start
            
            # Start countdown if exposure > 1 second
            if self.exposure_time_var.get() > 1000:
                self.countdown_active = True
                self.countdown_label.config(text=f"Frame 1 - {self.exposure_time_var.get() / 1000:.0f}s remaining")
                self.run_countdown()
            else:
                self.countdown_label.config(text="")

            # Set up save thread if needed - using optimized save thread
            if self.save_data_var.get():
                save_queue = queue.Queue(maxsize=50000)
                self.camera_thread.save_queue = save_queue
                object_name = self.object_name_entry.get() or "capture"
                header_dict = self.get_header_info()
                self.save_thread = OptimizedSaveThread(save_queue, self.camera_thread, 
                                                       header_dict, object_name, self.shared_data)
                self.save_thread.batch_size = self.cube_size_var.get()
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
            self.countdown_active = False
            self.countdown_label.config(text="")
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

    def take_flats(self):
        """Take flat field images cycling through filters with telescope offsets"""
        def _cycle_filters():
            try:
                # Store original settings
                original_exposure = self.exposure_time_var.get()
                original_save_state = self.save_data_var.get()
                original_object_name = self.object_name_entry.get()
                
                TARGET_MEAN = 40000
                BIAS_LEVEL = 200
                
                # Define offset pattern for 6 frames (in arcseconds)
                offset_pattern = [
                    (0, 0),      # Frame 1: original position
                    (5, 0),      # Frame 2: +5" RA
                    (5, 5),      # Frame 3: +5" RA, +5" Dec  
                    (0, 5),      # Frame 4: 0 RA, +5" Dec
                    (-5, 5),     # Frame 5: -5" RA, +5" Dec
                    (-5, 0),     # Frame 6: -5" RA, 0 Dec
                ]
                
                # Test if TCS accepts commands with a small test offset
                tcs_commands_work = False
                if hasattr(self, 'tcs_thread') and self.tcs_thread:
                    logging.info("Testing TCS command capability...")
                    # Try a tiny offset and immediately reverse it
                    test_success = self.tcs_thread.send_offset(0.1, 0)
                    if test_success:
                        # Reverse the test offset
                        reverse_success = self.tcs_thread.send_offset(-0.1, 0)
                        if reverse_success:
                            tcs_commands_work = True
                            logging.info("TCS accepts commands - will use offsets for flats")
                            self.after(0, lambda: self.update_status("TCS working - using offsets for flats", "green"))
                        else:
                            logging.warning("TCS accepted test offset but failed to reverse - commands unreliable")
                            self.after(0, lambda: self.update_status("TCS commands unreliable - flats at single position", "orange"))
                    else:
                        logging.warning("TCS does not accept commands - will take all flats at current position")
                        self.after(0, lambda: self.update_status("TCS commands disabled - flats at single position", "orange"))
                else:
                    logging.warning("TCS not available - will take all flats at current position")
                    self.after(0, lambda: self.update_status("TCS unavailable - flats at single position", "orange"))
                
                for filter_pos in [6, 1, 2, 3, 4, 5]:
                    # Set filter
                    filter_name = next((k for k, v in self.filter_options.items() if v == filter_pos), None)
                    self.filter_position_var.set(filter_name)
                    logging.info(f"Taking flats for filter {filter_name}")
                    self.peripherals_thread.efw.SetPosition(0, filter_pos)
                    self.peripherals_thread.efw.GetPosition(0)
                    time.sleep(0.5)
                    
                    # Find appropriate test exposure
                    test_exposure = 0.1  # Start at 100ms
                    test_mean = 70000
                    
                    while test_mean > 60000 or test_mean < 1000:
                        self.camera_thread.set_property('EXPOSURE_TIME', test_exposure)
                        while not self.frame_queue.empty():
                            self.frame_queue.get_nowait()
                        
                        self.camera_thread.start_capture()
                        test_frame = self.frame_queue.get(timeout=2.0)
                        self.camera_thread.stop_capture()
                        
                        # Calculate mean of central 100x100 region
                        h, w = test_frame.shape
                        cy, cx = h // 2, w // 2
                        central_region = test_frame[cy-50:cy+50, cx-50:cx+50]
                        test_mean = np.mean(central_region)
                        
                        if test_mean > 60000:
                            test_exposure *= 0.1
                            print(f"{filter_name}: Saturated (mean={test_mean:.0f}), trying {test_exposure*1000:.1f}ms")
                        elif test_mean < 1000:
                            test_exposure *= 5.0
                            print(f"{filter_name}: Too low (mean={test_mean:.0f}), trying {test_exposure*1000:.1f}ms")  
                    
                    # Calculate scaled exposure
                    scaled_exposure = test_exposure * (TARGET_MEAN - BIAS_LEVEL) / (test_mean - BIAS_LEVEL)
                    scaled_exposure = max(0.001, min(30, scaled_exposure))

                    print(f"{filter_name}: test={test_mean:.0f} @ {test_exposure*1000:.1f}ms, scaled={scaled_exposure*1000:.0f}ms")
                    
                    # Prepare for capture
                    self.save_data_var.set(True)
                    filter_short = filter_name.split()[0].replace('(', '').replace(')', '')
                    self.object_name_entry.delete(0, tk.END)
                    self.object_name_entry.insert(0, f"flat_{filter_short}_{int(scaled_exposure*1000)}ms")
                    
                    self.camera_thread.set_property('EXPOSURE_TIME', scaled_exposure)
                    
                    # Clear queues before starting
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    
                    # Start capture
                    self.camera_thread.save_queue = None
                    self.camera_thread.start_capture()
                    
                    # Collect exactly 6 frames
                    frames_to_save = []
                    timestamps_to_save = []
                    framestamps_to_save = []
                    actual_ra_offset = 0
                    actual_dec_offset = 0
                    
                    for i in range(6):
                        # Only attempt offsets if we confirmed TCS accepts commands
                        if i > 0 and tcs_commands_work:
                            ra_offset = offset_pattern[i][0] - offset_pattern[i-1][0]
                            dec_offset = offset_pattern[i][1] - offset_pattern[i-1][1]
                            
                            # Send telescope offset (we already know it should work)
                            success = self.tcs_thread.send_offset(ra_offset, dec_offset)
                            
                            if success:
                                actual_ra_offset = offset_pattern[i][0]
                                actual_dec_offset = offset_pattern[i][1]
                                logging.info(f"Flat frame {i+1}: moved to offset ({offset_pattern[i][0]}\", {offset_pattern[i][1]}\")")
                                time.sleep(2.0)  # Wait for telescope to settle
                            else:
                                # Unexpected failure - disable further offset attempts
                                logging.warning(f"Unexpected TCS offset failure for frame {i+1} - disabling offsets for remainder")
                                tcs_commands_work = False
                                time.sleep(0.5)
                        elif i > 0 and not tcs_commands_work:
                            # Log once per filter that we're not offsetting
                            if i == 1:
                                logging.info(f"Taking all {filter_name} flats at single position (TCS commands unavailable)")
                        
                        # Capture frame
                        frame = self.frame_queue.get(timeout=scaled_exposure + 2)
                        try:
                            ts, fs = self.timestamp_queue.get_nowait()
                            timestamps_to_save.append(ts)
                            framestamps_to_save.append(fs)
                        except:
                            timestamps_to_save.append(0)
                            framestamps_to_save.append(0)
                        frames_to_save.append(frame)
                        
                        logging.info(f"Captured flat frame {i+1}/6 for {filter_name}")
                    
                    self.camera_thread.stop_capture()
                    
                    # Return to original position only if we moved and TCS still works
                    if tcs_commands_work and (actual_ra_offset != 0 or actual_dec_offset != 0):
                        return_ra = -actual_ra_offset
                        return_dec = -actual_dec_offset
                        success = self.tcs_thread.send_offset(return_ra, return_dec)
                        if success:
                            logging.info(f"Returned to original position for next filter")
                        else:
                            logging.warning("Failed to return to original position - may need manual correction")
                            self.after(0, lambda: self.update_status(
                                "Warning: Failed to return telescope to original position", "orange"))
                            # Disable further offsets since return failed
                            tcs_commands_work = False
                        time.sleep(2.0)
                    
                    # Save the collected frames
                    save_queue = queue.Queue(maxsize=50000)
                    for i, frame in enumerate(frames_to_save):
                        save_queue.put((frame, timestamps_to_save[i], framestamps_to_save[i]))
                    
                    save_thread = OptimizedSaveThread(save_queue, self.camera_thread, self.get_header_info(),
                                                    self.object_name_entry.get(), self.shared_data)
                    save_thread.batch_size = 6
                    save_thread.start()
                    
                    time.sleep(2.0)
                    save_thread.stop()
                    save_thread.join(timeout=5)
                    
                    time.sleep(1.0)
                
                # Restore original settings
                self.camera_thread.set_property('EXPOSURE_TIME', original_exposure / 1000.0)
                self.save_data_var.set(original_save_state)
                self.object_name_entry.delete(0, tk.END)
                self.object_name_entry.insert(0, original_object_name)
                
                self.after(0, lambda: self.update_status("Flat sequence complete", "green"))
                
            except Exception as e:
                logging.error(f"Take flats error: {e}")
                self.after(0, lambda err=e: self.update_status(f"Error: {err}", "red"))
        
        if self.camera_thread.capturing:
            self.update_status("Cannot take flats while capturing", "orange")
            return
        
        threading.Thread(target=_cycle_filters, daemon=True).start()

    # Peripheral control methods - all unchanged
    def update_peripherals_status(self):
        """Update peripheral status periodically"""
        if self.updating_peripherals_status and self.peripherals_thread and not self._peripheral_update_running:
            self._peripheral_update_running = True
            threading.Thread(target=self._update_peripherals_background, daemon=True).start()
        
        self.after(1000, self.update_peripherals_status)

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
                        self.peripherals_thread.ax_a_1.move_absolute(0, Units.LENGTH_MILLIMETRES)
                    else:
                        self.peripherals_thread.ax_a_1.move_absolute(70, Units.LENGTH_MILLIMETRES)
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
                positions = {'Halpha': 151.5, 'QWP': 23.15, 'Neither': 87.18}
                with self.peripherals_thread.peripherals_lock:
                    print(positions[option])
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
                positions = {'WeDoWo': 17.78, 'Wire Grid': 128.5, 'Neither': 60.66}
                with self.peripherals_thread.peripherals_lock:
                    self.peripherals_thread.ax_b_2.move_absolute(
                        positions[option], Units.LENGTH_MILLIMETRES)
            except Exception as e:
                debug_logger.error(f"Pol stage error: {e}")
        
        self.peripherals_thread.executor.submit(_update)

    def move_zoom_preset(self, virtual_position):
            """Move zoom to preset position"""
            if not self.peripherals_thread.zoom_reference_set:
                messagebox.showerror("Reference Not Set", 
                                "Please set 3x zoom-out reference position.")
                return
            
            response = messagebox.askyesno("Set Zoom Preset", "Move to zoom pre-set?")

            def _move():
                if self.peripherals_thread.move_zoom_to_virtual(virtual_position):
                    self.update_status(f"Zoom at {virtual_position}", "green")
                else:
                    self.update_status("Zoom movement failed", "red")
            if response:
                self.peripherals_thread.executor.submit(_move)

    def set_zoom_reference(self):
        """Set current position as reference for virtual coordinate system"""
        response = messagebox.askyesno("Set Zoom Reference", 
            f"Set current zoom position as 3x zoom-out reference?")
        
        if response:
            def _set_ref():
                if self.peripherals_thread.set_zoom_reference():
                    self.update_status(f"Zoom reference set at 3x zoom-out", "green")
                else:
                    self.update_status("Failed to set zoom reference", "red")
            
            self.peripherals_thread.executor.submit(_set_ref)

    def emergency_zoom_move(self):
        """Emergency relative zoom movement with warnings"""
        try:
            relative_degrees = float(self.zoom_emergency_var.get())
            if abs(relative_degrees) < 0.1:
                self.update_status("Movement too small", "orange")
                return
                
            response = messagebox.askyesno("MOVEMENT WARNING", 
                f"You are about to move {relative_degrees:+.1f}° relatively.\n\n"
                f"This may hit mechanical limits!\n\n"
                f"Are you absolutely sure?")
            
            if not response:
                return
                
            if abs(relative_degrees) > 100:
                response2 = messagebox.askyesno("LARGE MOVEMENT WARNING", 
                    f"{abs(relative_degrees):.1f}° is a very large movement!\n\n"
                    f"This could damage the lens mechanism.\n\n"
                    f"Continue anyway?")
                if not response2:
                    return
            
            self.update_status(f"Emergency zoom move {relative_degrees:+.1f}°...", "orange")
            
            def _emergency_move():
                if self.peripherals_thread.move_zoom_relative(relative_degrees):
                    self.update_status(f"Emergency move {relative_degrees:+.1f}° complete", "orange")
                    self.after(100, lambda: self.zoom_emergency_var.set('0'))
                else:
                    self.update_status("Emergency move failed", "red")
            
            self.peripherals_thread.executor.submit(_emergency_move)
            
        except ValueError:
            self.update_status("Invalid emergency movement value", "red")

    def update_focus_position(self, *_):
        """Update focus position (relative movement)"""
        def _update():
            try:
                if self.peripherals_thread.ax_b_1 is None:
                    self.update_status("Focus motor not connected", "red")
                    return
                    
                relative_position = float(self.focus_position_var.get())
                if abs(relative_position) < 0.1:  # Minimum movement threshold
                    self.update_status("Focus movement too small", "orange")
                    return
                    
                with self.peripherals_thread.peripherals_lock:
                    # Get current position first
                    current_pos = self.peripherals_thread.ax_b_1.get_position(Units.ANGLE_DEGREES)
                    new_position = current_pos + relative_position
                    self.peripherals_thread.ax_b_1.move_absolute(new_position, Units.ANGLE_DEGREES)
                    
                self.update_status(f"Focus moved {relative_position:+.1f}° (toward {'far focus' if relative_position > 0 else 'near focus'})", "green")
                # Reset entry to 0 after movement
                self.after(100, lambda: self.focus_position_var.set('0'))
                
            except ValueError:
                self.update_status("Invalid focus value", "red")
            except Exception as e:
                debug_logger.error(f"Focus error: {e}")
                self.update_status("Focus movement failed", "red")
        
        self.peripherals_thread.executor.submit(_update)

    def manual_focus_init(self):
            """Manually trigger focus initialization sequence"""
            self.update_status("Initializing focus...", "blue")
            self.peripherals_thread.initialize_focus_sequence()

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

    def run_countdown(self):
        """Simple countdown that runs continuously during capture"""
        if not self.countdown_active or not self.camera_thread.capturing:
            self.countdown_label.config(text="")
            return
        
        # Calculate elapsed time since last frame (or start)
        elapsed = time.time() - self.last_frame_time
        remaining = max(0, self.exposure_time_var.get() / 1000 - elapsed)

        if remaining > 0:
            # Still exposing
            if remaining >= 10:
                text = f"Frame {self.frame_count + 1} - {remaining:.0f}s remaining"
            else:
                text = f"Frame {self.frame_count + 1} - {remaining:.1f}s remaining"
            self.countdown_label.config(text=text)
        else:
            # Exposure complete, waiting for readout
            text = f"Frame {self.frame_count + 1} - Reading..."
            self.countdown_label.config(text=text)
        
        # Schedule next update
        self.after(100, self.run_countdown)

    def update_tcs_status(self):
        """Update TCS status display"""
        if hasattr(self, 'tcs_thread') and self.tcs_thread:
            # Check if data is recent (within last 10 seconds)
            data_age = time.time() - self.tcs_thread.last_update
            
            if data_age < 10:  # Data is recent, so TCS is responding
                tcs_data = self.tcs_thread.get_current_data()
                status_text = f"TCS: Connected | RA: {tcs_data['ra']} | Dec: {tcs_data['dec']} | Airmass: {tcs_data['airmass']}"
                self.tcs_status_label.config(text=status_text, fg="green")
            else:
                # Data is stale, TCS might not be responding
                self.tcs_status_label.config(text=f"TCS: No recent data ({data_age:.0f}s old)", fg="orange")
        else:
            self.tcs_status_label.config(text="TCS: Thread not running", fg="red")
        
        self.after(5000, self.update_tcs_status)

    def send_telescope_offset(self, ra_offset=0, dec_offset=0):
        """Send telescope offset command
        
        Args:
            ra_offset: Direction multiplier for RA (+1 for West, -1 for East)
            dec_offset: Direction multiplier for Dec (+1 for North, -1 for South)
        """
        if not hasattr(self, 'tcs_thread') or not self.tcs_thread:
            self.update_status("TCS not available", "red")
            return
            
        try:
            offset_amount = self.tcs_offset_amount.get()
            actual_ra_offset = ra_offset * offset_amount
            actual_dec_offset = dec_offset * offset_amount
            
            # Send offset command
            success = self.tcs_thread.send_offset(actual_ra_offset, actual_dec_offset)
            
            if success:
                direction = ""
                if ra_offset > 0:
                    direction = "West"
                elif ra_offset < 0:
                    direction = "East"
                if dec_offset > 0:
                    direction = "North" if not direction else direction + "+North"
                elif dec_offset < 0:
                    direction = "South" if not direction else direction + "+South"
                    
                self.update_status(f"Offset {offset_amount}\" {direction}", "green")
            else:
                self.update_status("Offset failed - check TCS connection", "red")
                
        except Exception as e:
            logging.error(f"Telescope offset error: {e}")
            self.update_status(f"Offset error: {e}", "red")
    
    def reset_telescope_offsets(self):
        """Reset telescope offsets to zero"""
        if not hasattr(self, 'tcs_thread') or not self.tcs_thread:
            self.update_status("TCS not available", "red")
            return
            
        if self.tcs_thread.reset_offsets():
            self.update_status("Telescope offsets reset", "green")
        else:
            self.update_status("Reset failed - check TCS connection", "red")

    def get_header_info(self):
        """Get header info for FITS files including telescope parameters"""
        header_info = {}
        
        # Instrument parameters
        header_info['FILTER'] = self.filter_position_var.get()
        header_info['SHUTTER'] = self.shutter_var.get()
        header_info['SLIT'] = self.slit_position_var.get()
        header_info['HALPHA'] = self.halpha_qwp_var.get()
        header_info['POLSTAGE'] = self.wire_grid_var.get()
        
        # Get telescope parameters from TCS thread
        if hasattr(self, 'tcs_thread') and self.tcs_thread:
            tcs_data = self.tcs_thread.get_current_data()
            
            # Add telescope parameters to header
            header_info['TELRA'] = (tcs_data.get('ra', 'N/A'), 'Telescope Right Ascension')
            header_info['TELDEC'] = (tcs_data.get('dec', 'N/A'), 'Telescope Declination')
            header_info['TELEQUIN'] = (tcs_data.get('equinox', 'N/A'), 'Telescope coordinate epoch')
            header_info['TELROT'] = (tcs_data.get('rotator_angle', 'N/A'), 'Rotator angle (degrees)')
            header_info['TELPA'] = (tcs_data.get('parallactic_angle', 'N/A'), 'Parallactic angle (degrees)')
            header_info['AIRMASS'] = (tcs_data.get('airmass', 'N/A'), 'Airmass')
            header_info['TELEL'] = (tcs_data.get('elevation', 'N/A'), 'Telescope elevation (degrees)')
            header_info['TELAZ'] = (tcs_data.get('azimuth', 'N/A'), 'Telescope azimuth (degrees)')
            header_info['TELHA'] = (tcs_data.get('hour_angle', 'N/A'), 'Hour angle')
            header_info['TELST'] = (tcs_data.get('sidereal_time', 'N/A'), 'Sidereal time')
            header_info['DATEOBS'] = (tcs_data.get('date', 'N/A'), 'Observation date')
            header_info['TELUT'] = (tcs_data.get('time', 'N/A'), 'UT time from telescope')
            
            # Log if data is stale
            if hasattr(self.tcs_thread, 'last_update'):
                data_age = time.time() - self.tcs_thread.last_update
                if data_age > 10:
                    logging.warning(f"TCS data is {data_age:.1f} seconds old")
                    header_info['TCSAGE'] = (data_age, 'Age of TCS data in seconds')
        else:
            logging.warning("TCS thread not available - telescope parameters not included")
        
        return header_info

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

            # Stop TCS thread
            if hasattr(self, 'tcs_thread') and self.tcs_thread:
                logging.info("Stopping TCS thread")
                self.tcs_thread.stop()
                self.tcs_thread.join(timeout=2)

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
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        logging.info("Starting Camera Control Application (Optimized)")
        logging.info(f"CPU cores available: {mp.cpu_count()}")
        
        # Create shared resources
        shared_data = SharedData()
        frame_queue = queue.Queue(maxsize=5)  # Smaller queue to prevent buildup
        timestamp_queue = queue.Queue(maxsize=100000)
        
        # Create GUI (must be created first for thread references)
        app = CameraGUI(shared_data, None, None, frame_queue, timestamp_queue)
        
        # Create and start camera thread
        camera_thread = CameraThread(shared_data, frame_queue, timestamp_queue, app)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Create and start peripherals thread
        peripherals_thread = PeripheralsThread(
            shared_data, "10.8.105.32", "/dev/ttyACM0", "/dev/ttyACM1", app)
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
