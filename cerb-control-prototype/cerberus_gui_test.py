import tkinter as tk
from tkinter import StringVar, OptionMenu, Checkbutton, Label, Entry, Button, Frame, LabelFrame, messagebox
from dcam import Dcamapi, Dcam
from camera_params import CAMERA_PARAMS, DISPLAY_PARAMS
import threading
import time
import numpy as np
import cv2
import queue
from astropy.io import fits
from datetime import datetime, timezone
import os
import json
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# OpenCV optimizations for high-speed display
cv2.setNumThreads(8)  # Use more threads on Threadripper for image operations
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # Disable slow video backends

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add file handler
file_handler = logging.FileHandler('camera_simple.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
))
logging.getLogger().addHandler(file_handler)


class DCamLock:
    """Simplified locking for DCAM operations"""
    _capture_lock = threading.RLock()
    _property_lock = threading.RLock()
    
    @classmethod
    def acquire_capture(cls, timeout=5.0):
        return cls._capture_lock.acquire(blocking=True, timeout=timeout)
    
    @classmethod
    def release_capture(cls):
        try:
            cls._capture_lock.release()
        except:
            pass
    
    @classmethod
    def acquire_property(cls, timeout=2.0):
        return cls._property_lock.acquire(blocking=True, timeout=timeout)
    
    @classmethod
    def release_property(cls):
        try:
            cls._property_lock.release()
        except:
            pass


class SharedData:
    """Shared data container"""
    def __init__(self):
        self.camera_params = {}
        self.lock = threading.RLock()


class CameraThread(threading.Thread):
    """Camera control thread"""
    
    def __init__(self, shared_data, frame_queue, gui_ref):
        super().__init__(name="CameraThread")
        self.shared_data = shared_data
        self.frame_queue = frame_queue
        self.gui_ref = gui_ref
        self.dcam = None
        self.running = True
        self.capturing = False
        self.stop_requested = threading.Event()
        self.frame_index = 0
        self.buffer_size = 1000  # Increased for high frame rates (0.5s at 2000 fps)
        self.save_queue = None
        self.is_connected = False
        self.cleanup_done = False  # Flag to prevent double cleanup
        
        # Performance monitoring
        self.frame_count = 0
        self.fps_calc_time = time.time()
        
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
        """Connect to camera"""
        retry_count = 0
        max_retries = 3
        
        while self.running and retry_count < max_retries:
            retry_count += 1
            logging.info(f"Camera connection attempt {retry_count}")
            
            try:
                # Initialize DCAM API
                if Dcamapi.init():
                    logging.info("DCAM API initialized")
                else:
                    raise RuntimeError(f"DCAM API init failed: {Dcamapi.lasterr()}")
                    
                # Open camera device
                self.dcam = Dcam(0)
                if not self.dcam.dev_open():
                    raise RuntimeError(f"Device open failed: {self.dcam.lasterr()}")
                    
                logging.info("Camera connected successfully")
                self.set_defaults()
                self.update_camera_params()
                self.is_connected = True
                self.update_gui_status("Camera connected", "green")
                
                # Perform warmup capture to initialize camera hardware
                self.warmup_capture()
                
                return True
                
            except Exception as e:
                logging.warning(f"Failed to open camera: {e}")
                self.update_gui_status("Camera not connected", "red")
                Dcamapi.uninit()
                time.sleep(2)
        
        return False

    def main_loop(self):
        """Main camera operation loop"""
        while self.running:
            try:
                if self.capturing and not self.stop_requested.is_set():
                    self.capture_frame()
                else:
                    time.sleep(0.01)  # Short sleep to reduce CPU when idle
                    
            except Exception as e:
                logging.error(f"Error in camera loop: {e}")
                time.sleep(0.001)  # 1ms sleep to avoid tight loop on errors

    def capture_frame(self):
        """Capture a single frame"""
        if self.stop_requested.is_set():
            return
            
        timeout_milisec = 10  # 10ms is sufficient for high frame rates
        
        if not DCamLock.acquire_capture(timeout=0.005):  # Reduced to 5ms
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
                    frame_copy = np.copy(npBuf)
                    
                    DCamLock.release_capture()
                    
                    # Process frame with timing info
                    self.process_captured_frame(frame_copy, timestamp, framestamp)
                    return
                    
        finally:
            DCamLock.release_capture()

    def process_captured_frame(self, frame, timestamp, framestamp):
        """Process frame with timing information"""
        # Handle timestamp rollover (32-bit microseconds wraps at ~4295 seconds)
        raw_timestamp = timestamp.sec + timestamp.microsec / 1e6
        if raw_timestamp < self.last_raw_timestamp - 4000:
            self.timestamp_offset += 4294.967296
            logging.warning(f"Timestamp rollover at frame {self.frame_index}")
        self.last_raw_timestamp = raw_timestamp
        corrected_timestamp = raw_timestamp + self.timestamp_offset
        
        # Handle framestamp rollover (16-bit counter wraps at 65536)
        if framestamp < self.last_raw_framestamp - 60000:
            self.framestamp_offset += 65536
            logging.warning(f"Framestamp rollover at frame {self.frame_index}")
        self.last_raw_framestamp = framestamp
        corrected_framestamp = framestamp + self.framestamp_offset
        
        self.frame_count += 1
        
        # Queue frame for display - keep only the most recent frame
        try:
            # Clear the entire queue and only keep the newest frame
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
            # Now add the new frame
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        
        # Queue for saving if enabled
        if self.save_queue is not None:
            queue_size = self.save_queue.qsize()
            max_size = self.save_queue.maxsize
            
            # Apply backpressure if queue is getting dangerously full
            # Drop frames before we run out of memory
            if queue_size > max_size * 0.9:
                if not hasattr(self, '_backpressure_warned'):
                    self._backpressure_warned = True
                    logging.error(f"BACKPRESSURE: Save queue {queue_size}/{max_size} ({100*queue_size/max_size:.1f}% full) - "
                                 f"disk writes cannot keep up with capture rate! Dropping frames to prevent OOM.")
                # Drop this frame to prevent memory explosion
                if not hasattr(self, '_frames_dropped_backpressure'):
                    self._frames_dropped_backpressure = 0
                self._frames_dropped_backpressure += 1
                return  # Skip saving this frame
            elif queue_size > max_size * 0.8:
                # Warning level - still saving but close to limit
                if not hasattr(self, '_last_backpressure_warning'):
                    self._last_backpressure_warning = 0
                if time.time() - self._last_backpressure_warning > 5.0:
                    logging.warning(f"Save queue filling up: {queue_size}/{max_size} ({100*queue_size/max_size:.1f}% full) - "
                                   f"approaching backpressure threshold")
                    self._last_backpressure_warning = time.time()
            else:
                # Reset warning flags when queue drains
                if hasattr(self, '_backpressure_warned'):
                    delattr(self, '_backpressure_warned')
            
            try:
                self.save_queue.put_nowait((frame, corrected_timestamp, corrected_framestamp))
            except queue.Full:
                # This shouldn't happen with backpressure, but handle it anyway
                if not hasattr(self, '_frames_dropped_full'):
                    self._frames_dropped_full = 0
                self._frames_dropped_full += 1
                logging.error(f"Save queue full despite backpressure - dropped frame (total: {self._frames_dropped_full})")

        self.frame_index += 1

    def start_capture(self):
        """Start capture"""
        logging.info("Starting capture")
        
        self.stop_requested.clear()
        
        if not DCamLock.acquire_capture(timeout=3.0):
            logging.error("Failed to acquire lock")
            return False
            
        try:
            # Stop any existing capture
            if self.capturing:
                self._stop_capture_internal()

            if self.dcam is None:
                logging.error("Camera not initialized")
                return False
            
            # Allocate buffer
            if not self.dcam.buf_alloc(self.buffer_size):
                logging.error("Buffer allocation failed")
                return False

            # Initialize capture state
            self.capturing = True
            self.frame_index = 0
            self.frame_count = 0
            self.fps_calc_time = time.time()
            
            # Reset timestamp rollover tracking
            self.timestamp_offset = 0
            self.last_raw_timestamp = 0
            self.framestamp_offset = 0
            self.last_raw_framestamp = 0

            # Enable timestamp producer
            try:
                self.dcam.prop_setgetvalue(CAMERA_PARAMS['TIME_STAMP_PRODUCER'], 1)
            except Exception as e:
                logging.error(f"Error setting timestamp producer: {e}")

            # Record time immediately before starting camera
            time_before_cap_start = time.time()
            
            # Start capture
            if not self.dcam.cap_start():
                logging.error("Failed to start capture")
                self.capturing = False
                self.dcam.buf_release()
                return False

            # Record time immediately after cap_start returns
            time_after_cap_start = time.time()
            
            # Store these times
            self.time_before_cap_start = time_before_cap_start
            self.time_after_cap_start = time_after_cap_start
            
            logging.info(f"dcam.cap_start() took {(time_after_cap_start - time_before_cap_start)*1000:.2f}ms")
            logging.info("Capture started successfully")
            
            # Calculate expected PPS time if using external trigger
            if hasattr(self, 'trigger_source') and self.trigger_source == "External":
                # First integer second AFTER cap_start completed
                expected_pps_time = int(time_after_cap_start) + 1
                self.expected_pps_time = expected_pps_time
                logging.info(f"Expecting first PPS at {expected_pps_time:.6f} ({datetime.fromtimestamp(expected_pps_time, tz=timezone.utc).isoformat()})")
            
            return True
            
        finally:
            DCamLock.release_capture()

    def stop_capture(self):
        """Stop capture"""
        logging.info("Stopping capture")
        
        self.stop_requested.set()
        time.sleep(0.2)
        
        if DCamLock.acquire_capture(timeout=2.0):
            try:
                result = self._stop_capture_internal()
            finally:
                DCamLock.release_capture()
        else:
            result = self._stop_capture_internal(force=True)
        
        self.stop_requested.clear()
        return result

    def _stop_capture_internal(self, force=False):
        """Internal capture stop logic"""
        self.capturing = False
        
        if self.dcam is not None and not force:
            try:
                if not self.dcam.cap_stop():
                    logging.error(f"cap_stop failed: {self.dcam.lasterr()}")
                if not self.dcam.buf_release():
                    logging.error(f"buf_release failed: {self.dcam.lasterr()}")
                logging.info("Capture stopped cleanly")
            except Exception as e:
                logging.error(f"Error stopping capture: {e}")
        
        return True

    def set_defaults(self):
        """Set default camera parameters"""
        logging.info("Setting default camera parameters")
        defaults = {
            'READOUT_SPEED': 1.0,
            'EXPOSURE_TIME': 1.0,  # 1 second exposure time
            'TRIGGER_SOURCE': 2.0,  # 2=External (1=Internal, 3=Software)
            'TRIGGER_MODE': 6.0,    # 1=Normal, 6=Start
            'OUTPUT_TRIG_KIND_0': 3.0,    # 1=Ready, 2=Exposure, 3=Programmable, 4=Global
            'OUTPUT_TRIG_ACTIVE_0': 1.0,   # Enable output trigger
            'OUTPUT_TRIG_POLARITY_0': 1.0, # 1=Positive, 2=Negative
            'OUTPUT_TRIG_PERIOD_0': 10.0,  # Period in seconds
            'SENSOR_MODE': 1.0,
            'IMAGE_PIXEL_TYPE': 2.0,
            'DEFECTCORRECT_MODE': 1.0,  # 1=OFF, 2=ON
            'HOTPIXELCORRECT_LEVEL': 2.0  # 1=STANDARD, 2=MINIMUM, 3=AGGRESSIVE
        }
        for prop, value in defaults.items():
            self.set_property(prop, value)

    def warmup_capture(self):
        """Perform a quick warmup capture to initialize camera hardware
        
        This avoids the long 3+ second delay on first real capture by
        initializing the camera's internal state ahead of time.
        """
        logging.info("Performing warmup capture to initialize camera hardware...")
        
        if not DCamLock.acquire_capture(timeout=3.0):
            logging.warning("Could not acquire lock for warmup capture")
            return
        
        try:
            if self.dcam is None:
                return
            
            # Allocate minimal buffer (just 1 frame)
            if not self.dcam.buf_alloc(1):
                logging.warning("Warmup: Buffer allocation failed")
                return
            
            # Start capture (this is what takes 3+ seconds the first time)
            warmup_start = time.time()
            if not self.dcam.cap_start():
                logging.warning("Warmup: cap_start failed")
                self.dcam.buf_release()
                return
            
            warmup_duration = time.time() - warmup_start
            logging.info(f"Warmup: cap_start took {warmup_duration*1000:.2f}ms")
            
            # Immediately stop - we don't need to capture any frames
            time.sleep(0.01)  # Brief delay to let camera settle
            
            if not self.dcam.cap_stop():
                logging.warning("Warmup: cap_stop failed")
            
            if not self.dcam.buf_release():
                logging.warning("Warmup: buf_release failed")
            
            logging.info("Warmup capture complete - camera is now initialized")
            
        except Exception as e:
            logging.error(f"Error during warmup capture: {e}")
        finally:
            DCamLock.release_capture()

    def set_property(self, prop_name, value):
        """Set camera property"""
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
                
                # Log all available properties once for debugging
                if not hasattr(self, '_logged_all_props'):
                    self._logged_all_props = True
                    logging.info("=== ALL AVAILABLE CAMERA PROPERTIES ===")
                    for prop_name in sorted(self.shared_data.camera_params.keys()):
                        logging.info(f"  {prop_name}: {self.shared_data.camera_params[prop_name]}")
                    logging.info("=== END CAMERA PROPERTIES ===")
        finally:
            DCamLock.release_property()

    def cleanup(self):
        """Clean up resources with safe shutdown order"""
        # Prevent double cleanup
        if self.cleanup_done:
            logging.debug("Cleanup already completed, skipping")
            return
        
        self.cleanup_done = True
        logging.info("Starting camera thread cleanup...")
        
        # Step 1: Signal all loops to stop
        self.running = False
        self.stop_requested.set()
        
        # Report backpressure statistics if any frames were dropped
        if hasattr(self, '_frames_dropped_backpressure') and self._frames_dropped_backpressure > 0:
            total_frames = self.frame_index
            drop_rate = 100.0 * self._frames_dropped_backpressure / total_frames if total_frames > 0 else 0
            logging.warning(f"Camera thread backpressure statistics: {self._frames_dropped_backpressure} frames dropped "
                          f"out of {total_frames} captured ({drop_rate:.2f}% drop rate)")
            logging.warning("Backpressure drops indicate disk writes could not keep up with capture rate")
        
        if hasattr(self, '_frames_dropped_full') and self._frames_dropped_full > 0:
            logging.error(f"Queue full drops (should not happen): {self._frames_dropped_full} frames")
        
        # Step 2: Stop capture and wait for camera operations to complete
        if self.capturing:
            logging.info("Stopping capture...")
            try:
                self.stop_capture()
            except Exception as e:
                logging.error(f"Error stopping capture: {e}")
            
            # Give camera thread time to exit any wait loops
            time.sleep(0.5)
        
        # Step 3: Close camera device with lock protection
        if self.dcam is not None:
            logging.info("Closing camera device...")
            
            # Try to acquire lock, but proceed anyway if timeout
            lock_acquired = DCamLock.acquire_capture(timeout=2.0)
            
            try:
                self.dcam.dev_close()
                logging.info("Camera device closed successfully")
            except Exception as e:
                logging.error(f"Error closing camera device: {e}")
            finally:
                if lock_acquired:
                    DCamLock.release_capture()
                self.dcam = None
        
        # Step 4: Uninitialize DCAM API
        try:
            logging.info("Uninitializing DCAM API...")
            Dcamapi.uninit()
            logging.info("DCAM API uninitialized successfully")
        except Exception as e:
            logging.error(f"Error uninitializing DCAM API: {e}")
        
        logging.info("Camera thread cleanup complete")

    def update_gui_status(self, message, color):
        """Update GUI status message"""
        if self.gui_ref:
            self.gui_ref.update_status(message, color)

    def stop(self):
        """Stop thread"""
        self.running = False
        self.stop_requested.set()
        self.cleanup()


class SaveThread(threading.Thread):
    """Optimized save thread for high-speed FITS writing"""
    
    def __init__(self, save_queue, camera_thread, object_name, shared_data, batch_size, base_path=None, timing_info=None):
        super().__init__(name="SaveThread")
        self.save_queue = save_queue
        self.running = True
        self.camera_thread = camera_thread
        self.object_name = object_name
        self.batch_size = batch_size
        self.cube_index = 0
        self.shared_data = shared_data
        self.base_path = base_path or os.getcwd()  # Use provided path or current directory
        self.save_folder = "captures"
        self.timing_info = timing_info or {}  # Store timing information for FITS headers
        
        # First frame tracking
        self.first_frame_timestamp = None
        self.first_frame_logged = False
        
        # Preallocated buffers to avoid list append overhead
        self.frame_buffer = None
        self.timestamp_buffer = None
        self.framestamp_buffer = None
        self.buffer_index = 0
        
        # Performance monitoring
        self.total_frames_saved = 0
        self.total_frames_dropped = 0
        self.last_fps_time = time.time()
        self.frames_since_last_report = 0
        
        # Increase thread pool for parallel writes to maximize NVMe throughput
        # With 6000 MB/s NVMe and ~16 MB frames at 120 fps = 1920 MB/s
        # Use 6-8 workers to keep disk busy with multiple concurrent writes
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="FITSWriter")
        self.pending_writes = []
        
        # Memory monitoring
        self.max_pending_writes = 6  # Increased to keep more workers busy and handle variable write speeds
        
    def preallocate_buffers(self, frame_shape, frame_dtype):
        """Preallocate buffers based on frame size"""
        logging.info(f"Preallocating buffers for {self.batch_size} frames of shape {frame_shape}")
        self.frame_buffer = np.empty((self.batch_size, *frame_shape), dtype=frame_dtype)
        self.timestamp_buffer = np.empty(self.batch_size, dtype=np.float64)
        self.framestamp_buffer = np.empty(self.batch_size, dtype=np.int64)
        self.buffer_index = 0

    def run(self):
        """Main save thread loop"""
        try:
            logging.info("Save thread started")
            
            start_time_filename_str = time.strftime('%Y%m%d_%H%M%S')
            date_str = time.strftime('%Y_%m_%d')
            
            # Create save folder in the specified base path
            self.save_folder = os.path.join(self.base_path, f"captures_{date_str}")
            os.makedirs(self.save_folder, exist_ok=True)
            logging.info(f"Saving data to: {self.save_folder}")
            
            # Track consecutive drops for warning escalation
            consecutive_drops = 0
            last_queue_warning_time = 0

            while self.running or not self.save_queue.empty():
                try:
                    # Limit concurrent writes to prevent memory explosion
                    while len(self.pending_writes) >= self.max_pending_writes:
                        self.check_pending_writes()
                        time.sleep(0.001)  # Short sleep to avoid busy waiting
                    
                    # Pull frames from queue in batches for efficiency
                    frames_read = 0
                    max_batch_read = 100  # Read up to 100 frames at once
                    
                    while frames_read < max_batch_read and self.buffer_index < self.batch_size:
                        try:
                            frame, timestamp, framestamp = self.save_queue.get(timeout=0.001)
                            
                            # Detect and log first frame
                            if self.first_frame_timestamp is None:
                                # Record time immediately when first frame arrives
                                time_first_frame_arrived = time.time()
                                
                                # Store camera's relative timestamp
                                self.first_frame_timestamp = timestamp
                                self.timing_info['first_frame_timestamp'] = timestamp
                                self.timing_info['time_first_frame_arrived'] = time_first_frame_arrived
                                
                                # Get timing from camera_thread
                                if hasattr(self.camera_thread, 'time_before_cap_start'):
                                    self.timing_info['time_before_cap_start'] = self.camera_thread.time_before_cap_start
                                if hasattr(self.camera_thread, 'time_after_cap_start'):
                                    self.timing_info['time_after_cap_start'] = self.camera_thread.time_after_cap_start
                                if hasattr(self.camera_thread, 'expected_pps_time'):
                                    self.timing_info['expected_pps_time'] = self.camera_thread.expected_pps_time
                                
                                # Get readout time from camera
                                readout_time = None
                                if hasattr(self.camera_thread, 'dcam') and self.camera_thread.dcam is not None:
                                    try:
                                        # TIMING_READOUTTIME property ID is 0x00403010
                                        readout_time = self.camera_thread.dcam.prop_getvalue(0x00403010)
                                        if readout_time is not False:
                                            self.timing_info['readout_time'] = readout_time
                                    except:
                                        pass
                                
                                # Calculate estimated trigger arrival time
                                if 'exposure_time' in self.timing_info and readout_time is not None:
                                    # Frame arrived at T_FRAME1
                                    # Trigger arrived at: T_FRAME1 - exposure_time - readout_time
                                    exposure_time = self.timing_info['exposure_time']
                                    estimated_trigger_arrival = time_first_frame_arrived - exposure_time - readout_time
                                    self.timing_info['estimated_trigger_arrival'] = estimated_trigger_arrival
                                
                                if not self.first_frame_logged:
                                    self.first_frame_logged = True
                                    logging.info(f"First frame arrived at system time {time_first_frame_arrived:.6f} ({datetime.fromtimestamp(time_first_frame_arrived, tz=timezone.utc).isoformat()})")
                                    logging.info(f"First frame camera timestamp: {timestamp:.6f}s (camera-relative)")
                                    
                                    # Log exposure and readout times
                                    if 'exposure_time' in self.timing_info:
                                        logging.info(f"Exposure time: {self.timing_info['exposure_time']*1000:.3f}ms")
                                    if 'readout_time' in self.timing_info:
                                        logging.info(f"Readout time: {self.timing_info['readout_time']*1000:.3f}ms")
                                    
                                    # Log estimated trigger arrival
                                    if 'estimated_trigger_arrival' in self.timing_info:
                                        est_trigger = self.timing_info['estimated_trigger_arrival']
                                        logging.info(f"Estimated trigger arrival: {est_trigger:.6f} ({datetime.fromtimestamp(est_trigger, tz=timezone.utc).isoformat()})")
                                        
                                        # Compare to expected PPS
                                        if 'expected_pps_time' in self.timing_info:
                                            expected = self.timing_info['expected_pps_time']
                                            offset = est_trigger - expected
                                            logging.info(f"Trigger offset from expected PPS: {offset*1000:.2f}ms")
                                    
                                    # Log delays
                                    if 'time_before_cap_start' in self.timing_info and 'time_after_cap_start' in self.timing_info:
                                        cap_start_duration = self.timing_info['time_after_cap_start'] - self.timing_info['time_before_cap_start']
                                        time_from_cap_start = time_first_frame_arrived - self.timing_info['time_before_cap_start']
                                        logging.info(f"dcam.cap_start() duration: {cap_start_duration*1000:.2f}ms")
                                        logging.info(f"Time from cap_start call to first frame arrival: {time_from_cap_start*1000:.2f}ms")
                                    
                                    # Log total time from button press
                                    if 'trigger_sent_time' in self.timing_info:
                                        total_delay = time_first_frame_arrived - self.timing_info['trigger_sent_time']
                                        logging.info(f"Total time from button press to first frame: {total_delay*1000:.2f}ms")
                            
                            # Lazy buffer allocation on first frame
                            if self.frame_buffer is None:
                                self.preallocate_buffers(frame.shape, frame.dtype)
                            
                            # Copy directly into preallocated buffer (no list overhead)
                            self.frame_buffer[self.buffer_index] = frame
                            self.timestamp_buffer[self.buffer_index] = timestamp
                            self.framestamp_buffer[self.buffer_index] = framestamp
                            self.buffer_index += 1
                            frames_read += 1
                            self.frames_since_last_report += 1
                            consecutive_drops = 0
                            
                        except queue.Empty:
                            break
                    
                    # Monitor queue depth and warn if getting full
                    queue_size = self.save_queue.qsize()
                    if queue_size > self.save_queue.maxsize * 0.8:
                        current_time = time.time()
                        if current_time - last_queue_warning_time > 5.0:  # Warn every 5 seconds max
                            logging.warning(f"Save queue {queue_size}/{self.save_queue.maxsize} ({100*queue_size/self.save_queue.maxsize:.1f}% full) - disk may be falling behind!")
                            last_queue_warning_time = current_time
                    
                    # Write cube when buffer is full
                    if self.buffer_index >= self.batch_size:
                        self.write_cube_async(start_time_filename_str)
                        consecutive_drops = 0
                    
                    # Check for completed writes
                    self.check_pending_writes()
                    
                    # Report performance periodically
                    self.report_performance()
                        
                except Exception as e:
                    logging.error(f"Save thread error: {e}", exc_info=True)
                    consecutive_drops += 1
                    if consecutive_drops > 10:
                        logging.critical("Save thread experiencing repeated errors - capture may be unstable!")

            # Write remaining frames
            if self.buffer_index > 0:
                logging.info(f"Writing final partial cube with {self.buffer_index} frames")
                self.write_cube_async(start_time_filename_str, partial=True)
            
            # Wait for all pending writes
            logging.info(f"Waiting for {len(self.pending_writes)} pending writes to complete...")
            self.wait_for_pending_writes()
            
            # Final statistics
            logging.info(f"Save thread finished. Total frames saved: {self.total_frames_saved}, dropped: {self.total_frames_dropped}")
            if self.total_frames_dropped > 0:
                drop_rate = 100.0 * self.total_frames_dropped / (self.total_frames_saved + self.total_frames_dropped)
                logging.warning(f"Frame drop rate: {drop_rate:.2f}%")
                
        except Exception as e:
            logging.error(f"Fatal save thread error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def write_cube_async(self, start_time_filename_str, partial=False):
        """Write data cube asynchronously with zero-copy approach"""
        try:
            self.cube_index += 1
            filename = f"{self.object_name}_{start_time_filename_str}_cube{self.cube_index:03d}.fits"
            filepath = os.path.join(self.save_folder, filename)
            
            n_frames = self.buffer_index
            logging.info(f"Queuing cube {self.cube_index} ({n_frames} frames) for write")

            # Create views/slices instead of copies to save memory
            # These are lightweight and don't duplicate data
            if partial:
                # For partial cubes, we need to copy the relevant portion
                frames_to_write = self.frame_buffer[:n_frames].copy()
                timestamps_to_write = self.timestamp_buffer[:n_frames].copy()
                framestamps_to_write = self.framestamp_buffer[:n_frames].copy()
            else:
                # For full cubes, we can transfer ownership of the entire buffer
                # This is the zero-copy approach
                frames_to_write = self.frame_buffer
                timestamps_to_write = self.timestamp_buffer
                framestamps_to_write = self.framestamp_buffer
                
                # Allocate new buffers for next cube
                frame_shape = self.frame_buffer.shape[1:]
                frame_dtype = self.frame_buffer.dtype
                self.frame_buffer = np.empty((self.batch_size, *frame_shape), dtype=frame_dtype)
                self.timestamp_buffer = np.empty(self.batch_size, dtype=np.float64)
                self.framestamp_buffer = np.empty(self.batch_size, dtype=np.int64)
            
            # Get camera params snapshot
            with self.shared_data.lock:
                camera_params = dict(self.shared_data.camera_params)
            
            # Submit to thread pool
            write_start_time = time.time()
            future = self.executor.submit(
                self.write_fits_in_thread,
                filepath, frames_to_write, timestamps_to_write, framestamps_to_write,
                self.object_name, self.cube_index, camera_params, n_frames, write_start_time
            )
            
            self.pending_writes.append((filepath, future, n_frames))
            self.total_frames_saved += n_frames

            # Reset buffer index
            self.buffer_index = 0
            
            # Force garbage collection to free memory from completed writes
            import gc
            gc.collect()
            
        except Exception as e:
            logging.error(f"Write cube error: {e}", exc_info=True)

    def write_fits_in_thread(self, filepath, frames, timestamps, framestamps,
                             object_name, cube_index, camera_params, n_frames, write_start_time):
        """Write FITS file in thread with optimizations for speed"""
        try:
            start_time = time.time()
            
            # Frames is already a numpy array - no conversion needed!
            # Just use it directly or create a view if it's a partial cube
            data_cube = frames if len(frames) == n_frames else frames[:n_frames]
            
            # Create FITS HDUs
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['OBJECT'] = (object_name, 'Object name')
            primary_hdu.header['DATE-OBS'] = (datetime.now(timezone.utc).isoformat(), 'UTC date of observation')
            primary_hdu.header['CUBEIDX'] = (cube_index, 'Cube index number')
            primary_hdu.header['NFRAMES'] = (n_frames, 'Number of frames in cube')
            
            # Add timing information from capture start
            if 'button_press_time' in self.timing_info:
                primary_hdu.header['BTNPRESS'] = (self.timing_info['button_press_time'], 'Unix time when Start button pressed (s)')
                primary_hdu.header['BTNPISO'] = (datetime.fromtimestamp(self.timing_info['button_press_time'], tz=timezone.utc).isoformat(), 
                                                  'ISO time when Start button pressed (UTC)')
            
            # Time immediately before dcam.cap_start() call
            if 'time_before_cap_start' in self.timing_info:
                primary_hdu.header['T_BFCAPS'] = (self.timing_info['time_before_cap_start'], 'Unix time before dcam.cap_start() (s)')
                primary_hdu.header['BFCAPISO'] = (datetime.fromtimestamp(self.timing_info['time_before_cap_start'], tz=timezone.utc).isoformat(),
                                                   'ISO time before dcam.cap_start() (UTC)')
            
            # Time immediately after dcam.cap_start() returns
            if 'time_after_cap_start' in self.timing_info:
                primary_hdu.header['T_AFCAPS'] = (self.timing_info['time_after_cap_start'], 'Unix time after dcam.cap_start() (s)')
                primary_hdu.header['AFCAPISO'] = (datetime.fromtimestamp(self.timing_info['time_after_cap_start'], tz=timezone.utc).isoformat(),
                                                   'ISO time after dcam.cap_start() (UTC)')
            
            # Add expected PPS time (for external triggers)
            if 'expected_pps_time' in self.timing_info:
                primary_hdu.header['EXPPPS'] = (self.timing_info['expected_pps_time'], 'Expected PPS time (Unix s)')
                primary_hdu.header['EXPPISIS'] = (datetime.fromtimestamp(self.timing_info['expected_pps_time'], tz=timezone.utc).isoformat(),
                                                   'Expected PPS time (UTC)')
            
            # Time when first frame arrived at SaveThread
            if 'time_first_frame_arrived' in self.timing_info:
                primary_hdu.header['T_FRAME1'] = (self.timing_info['time_first_frame_arrived'], 'Unix time when first frame arrived (s)')
                primary_hdu.header['FR1ISO'] = (datetime.fromtimestamp(self.timing_info['time_first_frame_arrived'], tz=timezone.utc).isoformat(),
                                                 'ISO time when first frame arrived (UTC)')
            
            # Camera-relative timestamp of first frame
            if 'first_frame_timestamp' in self.timing_info:
                primary_hdu.header['CAMTS1'] = (self.timing_info['first_frame_timestamp'], 'First frame camera timestamp (s)')
                primary_hdu.header['COMMENT'] = 'CAMTS1 is camera-relative time, NOT Unix time'
                primary_hdu.header['COMMENT'] = 'Camera timestamps are seconds since camera started'
            
            # Exposure and readout times
            if 'exposure_time' in self.timing_info:
                primary_hdu.header['EXPTIME'] = (self.timing_info['exposure_time'], 'Exposure time (s)')
            if 'readout_time' in self.timing_info:
                primary_hdu.header['READTIME'] = (self.timing_info['readout_time'], 'Readout time (s)')
            
            # Estimated trigger arrival (frame arrival - exposure - readout)
            if 'estimated_trigger_arrival' in self.timing_info:
                primary_hdu.header['T_TRIG'] = (self.timing_info['estimated_trigger_arrival'], 'Estimated trigger arrival time (Unix s)')
                primary_hdu.header['TRIGISO'] = (datetime.fromtimestamp(self.timing_info['estimated_trigger_arrival'], tz=timezone.utc).isoformat(),
                                                  'Estimated trigger arrival (ISO UTC)')
                primary_hdu.header['COMMENT'] = 'T_TRIG = T_FRAME1 - EXPTIME - READTIME'
            
            # Store trigger source mode
            if 'trigger_source' in self.timing_info:
                primary_hdu.header['TRIGSRC'] = (self.timing_info['trigger_source'], 'Trigger source mode')

            # Use data_cube directly - it's already a numpy array
            image_hdu = fits.ImageHDU(data=data_cube)
            image_hdu.header['EXTNAME'] = 'DATA_CUBE'
            
            # Add camera parameters (abbreviated to reduce overhead)
            for key, value in camera_params.items():
                if len(key) > 8:  # FITS keyword limit
                    key = key[:8]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        image_hdu.header[key] = value
                    except:
                        pass

            # Create timestamp table HDU - use views of arrays
            timestamp_data = timestamps if len(timestamps) == n_frames else timestamps[:n_frames]
            framestamp_data = framestamps if len(framestamps) == n_frames else framestamps[:n_frames]
            
            col1 = fits.Column(name='TIMESTAMP', format='D', array=timestamp_data)
            col2 = fits.Column(name='FRAMESTAMP', format='K', array=framestamp_data)
            timestamp_hdu = fits.BinTableHDU.from_columns([col1, col2])
            timestamp_hdu.header['EXTNAME'] = 'TIMESTAMPS'
            timestamp_hdu.header['TUNIT1'] = ('seconds', 'Camera timestamp in seconds')
            timestamp_hdu.header['TUNIT2'] = ('count', 'Frame counter from camera')

            # Write file - this is where most time is spent
            hdulist = fits.HDUList([primary_hdu, image_hdu, timestamp_hdu])
            
            # Use overwrite=True to avoid checking for existing files (saves time)
            hdulist.writeto(filepath, overwrite=True)
            hdulist.close()
            
            # Calculate write performance
            write_time = time.time() - start_time
            queue_wait_time = start_time - write_start_time
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            write_speed_mbps = file_size_mb / write_time if write_time > 0 else 0
            
            logging.info(f"Wrote {filepath} ({file_size_mb:.1f} MB) in {write_time:.2f}s ({write_speed_mbps:.1f} MB/s, queue wait: {queue_wait_time:.2f}s)")
            
            # Explicitly delete large objects to help garbage collector
            del data_cube, hdulist, image_hdu, timestamp_hdu
            
            return True
            
        except Exception as e:
            logging.error(f"FITS write error for {filepath}: {e}", exc_info=True)
            return False

    def check_pending_writes(self):
        """Check status of pending writes and free memory"""
        completed = []
        for filepath, future, n_frames in self.pending_writes:
            if future.done():
                try:
                    success = future.result(timeout=0)
                    if not success:
                        logging.error(f"Failed writing: {filepath}")
                        self.total_frames_dropped += n_frames
                except Exception as e:
                    logging.error(f"Write error for {filepath}: {e}")
                    self.total_frames_dropped += n_frames
                completed.append((filepath, future, n_frames))
        
        # Remove completed writes
        for item in completed:
            self.pending_writes.remove(item)
        
        # Trigger GC if we just freed significant memory
        if len(completed) > 0:
            import gc
            gc.collect()

    def wait_for_pending_writes(self):
        """Wait for all pending writes to complete"""
        total_pending = len(self.pending_writes)
        for i, (filepath, future, n_frames) in enumerate(self.pending_writes, 1):
            try:
                logging.info(f"Waiting for write {i}/{total_pending}: {os.path.basename(filepath)}")
                success = future.result(timeout=60)
                if not success:
                    logging.error(f"Failed final write: {filepath}")
                    self.total_frames_dropped += n_frames
            except Exception as e:
                logging.error(f"Final write error for {filepath}: {e}")
                self.total_frames_dropped += n_frames
        
        self.pending_writes.clear()

    def report_performance(self):
        """Report save performance statistics dynamically based on cube size"""
        # Report every batch_size/5 frames (5 reports per cube)
        report_interval = max(self.batch_size // 5, 1)  # At least report every frame if batch_size < 5
        
        if self.frames_since_last_report >= report_interval:
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            save_fps = self.frames_since_last_report / elapsed if elapsed > 0 else 0
            queue_size = self.save_queue.qsize()
            queue_pct = 100.0 * queue_size / self.save_queue.maxsize if self.save_queue.maxsize > 0 else 0
            
            logging.info(f"Save thread: {save_fps:.1f} fps, queue: {queue_size} ({queue_pct:.1f}%), "
                        f"pending writes: {len(self.pending_writes)}, "
                        f"total saved: {self.total_frames_saved}, dropped: {self.total_frames_dropped}")
            
            self.frames_since_last_report = 0
            self.last_fps_time = current_time

    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up save thread...")
        self.executor.shutdown(wait=True)
        
        # Clear buffers
        self.frame_buffer = None
        self.timestamp_buffer = None
        self.framestamp_buffer = None
        import gc
        gc.collect()
        
        logging.info("Save thread cleanup complete")

    def stop(self):
        """Stop save thread"""
        logging.info("Stop requested for save thread")
        self.running = False


class SimpleConfigManager:
    """Simple configuration manager"""
    
    def __init__(self, config_file="camera_config.json"):
        self.config_file = config_file
    
    def save_config(self, gui_instance):
        """Save current GUI configuration"""
        try:
            config = {
                "exposure_time_ms": gui_instance.exposure_time_var.get(),
                "binning": gui_instance.binning_var.get(),
                "bit_depth": gui_instance.bit_depth_var.get(),
                "readout_speed": gui_instance.readout_speed_var.get(),
                "sensor_mode": gui_instance.sensor_mode_var.get(),
                "subarray_mode": gui_instance.subarray_mode_var.get(),
                "trigger_source": gui_instance.trigger_source_var.get(),
                "trigger_mode": gui_instance.trigger_mode_var.get(),
                "defect_correct": gui_instance.defect_correct_var.get(),
                "hot_pixel_level": gui_instance.hot_pixel_level_var.get(),
                "save_data": gui_instance.save_data_var.get(),
                "object_name": gui_instance.object_name_entry.get(),
                "output_path": gui_instance.output_path_var.get(),
                "frames_per_datacube": gui_instance.cube_size_var.get(),
                "min_count": gui_instance.min_val.get(),
                "max_count": gui_instance.max_val.get(),
                "scaling_type": gui_instance.scaling_type_var.get()
            }
            
            # Add subarray parameters
            for param in ["HPOS", "HSIZE", "VPOS", "VSIZE"]:
                if param in gui_instance.subarray_vars:
                    config[f"subarray_{param.lower()}"] = gui_instance.subarray_vars[param].get()
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self, gui_instance):
        """Load and apply saved configuration"""
        try:
            if not os.path.exists(self.config_file):
                return False
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Apply settings
            gui_instance.exposure_time_var.set(config.get("exposure_time_ms", 1000))  # 1 second default
            gui_instance.binning_var.set(config.get("binning", "1x1"))
            gui_instance.bit_depth_var.set(config.get("bit_depth", "16-bit"))
            gui_instance.readout_speed_var.set(config.get("readout_speed", "Ultra Quiet Mode"))
            gui_instance.sensor_mode_var.set(config.get("sensor_mode", "Standard"))
            gui_instance.subarray_mode_var.set(config.get("subarray_mode", "Off"))
            gui_instance.trigger_source_var.set(config.get("trigger_source", "External"))  # External default
            gui_instance.trigger_mode_var.set(config.get("trigger_mode", "Start"))
            gui_instance.defect_correct_var.set(config.get("defect_correct", "OFF"))  # OFF default
            gui_instance.hot_pixel_level_var.set(config.get("hot_pixel_level", "MINIMUM"))  # MINIMUM default
            
            gui_instance.save_data_var.set(config.get("save_data", False))
            gui_instance.object_name_entry.delete(0, tk.END)
            gui_instance.object_name_entry.insert(0, config.get("object_name", ""))
            gui_instance.output_path_var.set(config.get("output_path", "/data/cerberus"))  # /data/cerberus default
            gui_instance.cube_size_var.set(config.get("frames_per_datacube", 100))
            
            gui_instance.min_val.set(str(config.get("min_count", "0")))
            gui_instance.max_val.set(str(config.get("max_count", "60000")))
            gui_instance.scaling_type_var.set(config.get("scaling_type", "Linear"))
            
            # Apply subarray parameters
            subarray_params = {
                "HPOS": config.get("subarray_hpos", 0),
                "HSIZE": config.get("subarray_hsize", 4096),
                "VPOS": config.get("subarray_vpos", 0),
                "VSIZE": config.get("subarray_vsize", 2304)
            }
            for param, value in subarray_params.items():
                if param in gui_instance.subarray_vars:
                    gui_instance.subarray_vars[param].set(value)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            return False
    
    def config_exists(self):
        return os.path.exists(self.config_file)


class CameraGUI(tk.Tk):
    """Main GUI application"""
    
    def __init__(self, shared_data, camera_thread, frame_queue):
        super().__init__()
        self.shared_data = shared_data
        self.camera_thread = camera_thread
        self.frame_queue = frame_queue
        self.updating_camera_status = True
        self.updating_frame_display = True
        self.save_thread = None
        self.last_frame = None
        self.mouse_x = None
        self.mouse_y = None
        
        # Performance monitoring
        self.actual_display_count = 0
        self.last_display_time = time.time()
        
        # LUT cache for fast scaling
        self._lut_cache = {}
        self._lut_cache_key = None

        self.min_val = tk.StringVar(value="200")
        self.max_val = tk.StringVar(value="300")

        self.title("Simple Camera Control")
        self.geometry("1200x950")
        
        # Set larger default fonts for better readability
        default_font = ("TkDefaultFont", 11)
        self.option_add("*Font", default_font)
        self.option_add("*Label.Font", default_font)
        self.option_add("*Button.Font", default_font)
        self.option_add("*Entry.Font", default_font)
        self.option_add("*Checkbutton.Font", default_font)
        self.option_add("*Menubutton.Font", default_font)

        self.setup_gui()

    def setup_gui(self):
        """Set up GUI elements"""
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
        self.status_message = tk.Label(self.main_frame, text="", justify=tk.LEFT, anchor="center", 
                                       width=40, wraplength=400, fg="red", font=("Arial", 13, "bold"))
        self.status_message.grid(row=4, column=0, sticky='ew')
        
        # Performance display
        perf_frame = LabelFrame(self.main_frame, text="Performance Monitor", padx=5, pady=5)
        perf_frame.grid(row=1, column=0, sticky='ew')
        self.performance_label = tk.Label(perf_frame, text="FPS: -- | Display FPS: --", font=("Courier", 11))
        self.performance_label.pack()

        # Set up control sections
        self.setup_camera_controls()
        self.setup_camera_settings()
        self.setup_subarray_controls()
        self.setup_output_trigger_controls()
        self.setup_display_controls()
        
        # Start update loops
        self.after(2000, self.update_camera_status)
        self.after(8, self.update_frame_display)  # ~120 Hz max (8ms interval)
        self.after(1000, self.update_performance_monitor)

    def update_performance_monitor(self):
        """Monitor and display performance metrics"""
        # Calculate capture FPS
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
        
        # Calculate display FPS
        current_time = time.time()
        time_elapsed = current_time - self.last_display_time
        if time_elapsed > 0:
            display_fps = self.actual_display_count / time_elapsed
        else:
            display_fps = 0
        self.actual_display_count = 0
        self.last_display_time = current_time
        
        self.performance_label.config(
            text=f"Capture FPS: {capture_fps:.1f} | Display FPS: {display_fps:.1f}"
        )
        
        self.after(1000, self.update_performance_monitor)

    def setup_camera_controls(self):
        """Set up camera control widgets"""
        camera_controls_frame = LabelFrame(self.main_frame, text="Basic Controls", padx=5, pady=5)
        camera_controls_frame.grid(row=0, column=1)

        self.config_manager = SimpleConfigManager()
        
        # Restore settings button
        self.restore_config_button = Button(camera_controls_frame, text="Restore Settings", 
                                        command=self.restore_previous_config, bg='lightblue')
        self.restore_config_button.grid(row=0, column=0, columnspan=2, pady=5)
        
        if not self.config_manager.config_exists():
            self.restore_config_button.config(state='disabled', text='No Previous Settings')

        Label(camera_controls_frame, text="Exposure Time (ms):").grid(row=1, column=0)
        self.exposure_time_var = tk.DoubleVar(value=1000)  # 1000ms = 1 second
        self.exposure_time_var.trace_add("write", self.update_exposure_time)
        self.exposure_time_entry = Entry(camera_controls_frame, textvariable=self.exposure_time_var)
        self.exposure_time_entry.grid(row=1, column=1)

        self.start_button = Button(camera_controls_frame, text="Start Streaming", 
                                   command=self.start_capture, fg='green')
        self.start_button.grid(row=2, column=0)

        self.stop_button = Button(camera_controls_frame, text="Stop Streaming", 
                                  command=self.stop_capture, fg='red', state='disabled')
        self.stop_button.grid(row=2, column=1)

        self.save_data_var = tk.BooleanVar()
        self.save_data_checkbox = Checkbutton(camera_controls_frame, text="Save Data to Disk", 
                                            variable=self.save_data_var)
        self.save_data_checkbox.grid(row=3, column=0, columnspan=2)

        Label(camera_controls_frame, text="Object Name:").grid(row=4, column=0)
        self.object_name_entry = Entry(camera_controls_frame)
        self.object_name_entry.grid(row=4, column=1)

        Label(camera_controls_frame, text="Output Directory:").grid(row=5, column=0)
        output_path_frame = Frame(camera_controls_frame)
        output_path_frame.grid(row=5, column=1, sticky='ew')
        self.output_path_var = tk.StringVar(value="/data/cerberus")
        self.output_path_entry = Entry(output_path_frame, textvariable=self.output_path_var, width=20)
        self.output_path_entry.pack(side='left', fill='x', expand=True)
        self.browse_button = Button(output_path_frame, text="Browse...", command=self.browse_output_path)
        self.browse_button.pack(side='left', padx=(2, 0))

        Label(camera_controls_frame, text="Frames per Datacube:").grid(row=6, column=0)
        self.cube_size_var = tk.IntVar(value=100)
        self.cube_size_entry = Entry(camera_controls_frame, textvariable=self.cube_size_var)
        self.cube_size_entry.grid(row=6, column=1)

    def setup_camera_settings(self):
        """Set up camera settings widgets"""
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

        Label(camera_settings_frame, text="Trigger Source:").grid(row=4, column=0)
        self.trigger_source_var = StringVar(value="External")
        self.trigger_source_menu = OptionMenu(camera_settings_frame, self.trigger_source_var,
                                              "Internal", "External", "Software",
                                              command=self.change_trigger_source)
        self.trigger_source_menu.grid(row=4, column=1)

        Label(camera_settings_frame, text="Trigger Mode:").grid(row=5, column=0)
        self.trigger_mode_var = StringVar(value="Start")
        self.trigger_mode_menu = OptionMenu(camera_settings_frame, self.trigger_mode_var,
                                            "Normal", "Start",
                                            command=self.change_trigger_mode)
        self.trigger_mode_menu.grid(row=5, column=1)
        
        # Defect correction controls
        Label(camera_settings_frame, text="Defect Correction:").grid(row=6, column=0)
        self.defect_correct_var = StringVar(value="OFF")
        self.defect_correct_menu = OptionMenu(camera_settings_frame, self.defect_correct_var,
                                              "OFF", "ON",
                                              command=self.change_defect_correct)
        self.defect_correct_menu.grid(row=6, column=1)
        
        Label(camera_settings_frame, text="Hot Pixel Level:").grid(row=7, column=0)
        self.hot_pixel_level_var = StringVar(value="MINIMUM")
        self.hot_pixel_level_menu = OptionMenu(camera_settings_frame, self.hot_pixel_level_var,
                                               "STANDARD", "MINIMUM", "AGGRESSIVE",
                                               command=self.change_hot_pixel_level)
        self.hot_pixel_level_menu.grid(row=7, column=1)

    def setup_subarray_controls(self):
        """Set up subarray control widgets"""
        subarray_controls_frame = LabelFrame(self.main_frame, text="Subarray Controls", padx=5, pady=5)
        subarray_controls_frame.grid(row=3, column=1)

        Label(subarray_controls_frame, text="Subarray Mode:").grid(row=0, column=0, columnspan=2)
        self.subarray_mode_var = StringVar(value="Off")
        self.subarray_mode_menu = OptionMenu(subarray_controls_frame, self.subarray_mode_var, 
                                             "Off", "On", command=self.change_subarray_mode)
        self.subarray_mode_menu.grid(row=0, column=2, columnspan=2)

        subarray_params = [
            ("HPOS", 0, 1, 0),
            ("HSIZE", 4096, 1, 2),
            ("VPOS", 0, 2, 0),
            ("VSIZE", 2304, 2, 2)
        ]
        
        self.subarray_vars = {}
        self.subarray_entries = {}
        
        for param, default, row, col in subarray_params:
            Label(subarray_controls_frame, text=f"{param}:").grid(row=row, column=col)
            var = tk.IntVar(value=default)
            var.trace_add("write", self.update_subarray)
            self.subarray_vars[param] = var
            entry = Entry(subarray_controls_frame, textvariable=var, state='disabled', width=8)
            entry.grid(row=row, column=col+1)
            self.subarray_entries[param] = entry

        Label(subarray_controls_frame, text="Note: Values rounded to nearest factor of 4.").grid(
            row=5, column=0, columnspan=4)

    def setup_output_trigger_controls(self):
        """Set up output trigger controls"""
        output_trig_frame = LabelFrame(self.main_frame, text="Output Trigger (for syncing other equipment)", 
                                       padx=5, pady=5)
        output_trig_frame.grid(row=4, column=1, sticky='ew')

        Label(output_trig_frame, text="Trigger Kind:").grid(row=0, column=0)
        self.output_trig_kind_var = StringVar(value="Programmable")
        self.output_trig_kind_menu = OptionMenu(output_trig_frame, self.output_trig_kind_var,
                                               "Trigger Ready", "Exposure", "Programmable", "Global Exposure",
                                               command=self.change_output_trigger_kind)
        self.output_trig_kind_menu.grid(row=0, column=1)

        Label(output_trig_frame, text="Polarity:").grid(row=1, column=0)
        self.output_trig_polarity_var = StringVar(value="Positive")
        self.output_trig_polarity_menu = OptionMenu(output_trig_frame, self.output_trig_polarity_var,
                                                   "Positive", "Negative",
                                                   command=self.change_output_trigger_polarity)
        self.output_trig_polarity_menu.grid(row=1, column=1)

        Label(output_trig_frame, text="Period (s):").grid(row=2, column=0)
        self.output_trig_period_var = tk.DoubleVar(value=10.0)
        self.output_trig_period_var.trace_add("write", self.update_output_trigger_period)
        self.output_trig_period_entry = Entry(output_trig_frame,
                                              textvariable=self.output_trig_period_var, width=10)
        self.output_trig_period_entry.grid(row=2, column=1)
        
        Label(output_trig_frame, text="Use to sync external equipment", 
              font=("Arial", 8), fg="gray").grid(row=3, column=0, columnspan=2)

    def setup_display_controls(self):
        """Set up display control widgets"""
        display_controls_frame = LabelFrame(self.main_frame, text="Display Controls", padx=5, pady=5)
        display_controls_frame.grid(row=2, column=0)

        # Scaling type selection
        scaling_frame = Frame(display_controls_frame)
        scaling_frame.grid(row=0, column=0, columnspan=4, pady=5)
        
        Label(scaling_frame, text="Scaling:").pack(side='left', padx=2)
        
        self.scaling_type_var = tk.StringVar(value="Linear")
        self.linear_radio = tk.Radiobutton(scaling_frame, text="Linear", 
                                        variable=self.scaling_type_var, value="Linear",
                                        command=self.refresh_frame_display)
        self.linear_radio.pack(side='left', padx=5)
        
        self.log_radio = tk.Radiobutton(scaling_frame, text="Log", 
                                    variable=self.scaling_type_var, value="Log",
                                    command=self.refresh_frame_display)
        self.log_radio.pack(side='left', padx=5)

        # Manual scaling controls
        Label(display_controls_frame, text="Min Count:").grid(row=1, column=0)
        self.min_val.trace_add("write", self.refresh_frame_display)
        self.min_entry = Entry(display_controls_frame, textvariable=self.min_val, width=8)
        self.min_entry.grid(row=1, column=1)

        Label(display_controls_frame, text="Max Count:").grid(row=1, column=2)
        self.max_val.trace_add("write", self.refresh_frame_display)
        self.max_entry = Entry(display_controls_frame, textvariable=self.max_val, width=8)
        self.max_entry.grid(row=1, column=3)

        self.pixel_info_label = Label(display_controls_frame, text="Pixel (x,y): -- Value: --", 
                                      font=("Courier", 11))
        self.pixel_info_label.grid(row=2, column=0, columnspan=4)

    def restore_previous_config(self):
        """Restore previous configuration"""
        if self.camera_thread.capturing:
            self.update_status("Cannot restore during capture", "orange")
            return
        
        success = self.config_manager.load_config(self)
        if success:
            self.update_status("Previous settings restored", "green")
            logging.info("Configuration restored")
        else:
            self.update_status("Failed to restore settings", "red")

    def update_status(self, message, color="red"):
        """Update status message"""
        self.after(0, lambda: self.status_message.config(text=message, fg=color))

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
                    logging.error(f"Camera status error: {e}")
            
            threading.Thread(target=_update, daemon=True).start()
        
        self.after(2000, self.update_camera_status)

    def update_frame_display(self):
        """Update frame display"""
        if self.updating_frame_display:
            try:
                if not self.frame_queue.empty():
                    try:
                        frame = self.frame_queue.get_nowait()
                        self.last_frame = frame
                        self.actual_display_count += 1
                        self.process_frame(frame)
                    except queue.Empty:
                        pass
            except Exception as e:
                logging.error(f"Frame display error: {e}")
        
        self.after(8, self.update_frame_display)  # ~120 Hz display rate

    def refresh_frame_display(self, *_):
        """Refresh the frame display"""
        # Invalidate LUT cache since parameters changed
        self._lut_cache_key = None
        
        if self.last_frame is not None:
            self.process_frame(self.last_frame)

    def apply_scaling(self, data, min_val, max_val, scaling_type="Linear"):
        """Apply linear or logarithmic scaling using LUT for speed"""
        try:
            # Create cache key
            cache_key = (min_val, max_val, scaling_type, data.dtype)
            
            # Check if we can reuse cached LUT
            if cache_key != self._lut_cache_key:
                # Build new LUT
                if data.dtype == np.uint16:
                    input_range = 65536
                else:
                    input_range = 256
                
                lut_input = np.arange(input_range, dtype=np.float32)
                
                if scaling_type == "Log":
                    # Log LUT
                    offset = 1.0
                    clipped = np.clip(lut_input, min_val, max_val)
                    clipped = clipped - min_val + offset
                    log_range = np.log10(max_val - min_val + offset)
                    
                    if log_range > 0:
                        lut = np.log10(clipped) * (255.0 / log_range)
                    else:
                        lut = np.zeros_like(lut_input)
                else:
                    # Linear LUT
                    if max_val > min_val:
                        scale_factor = 255.0 / (max_val - min_val)
                        lut = (lut_input - min_val) * scale_factor
                    else:
                        lut = np.zeros_like(lut_input)
                
                # Clip and convert to uint8
                lut = np.clip(lut, 0, 255).astype(np.uint8)
                
                # Cache it
                self._lut_cache = lut
                self._lut_cache_key = cache_key
            
            # Apply LUT - this is extremely fast
            scaled = self._lut_cache[data]
            return scaled
            
        except Exception as e:
            logging.error(f"Error in LUT scaling: {e}")
            # Fast fallback without LUT
            if max_val > min_val:
                scale_factor = 255.0 / (max_val - min_val)
                return np.clip((data.astype(np.float32) - min_val) * scale_factor, 0, 255).astype(np.uint8)
            else:
                return np.zeros(data.shape, dtype=np.uint8)

    def process_frame(self, data):
        """Process and display a frame"""
        try:
            try:
                min_val = int(self.min_val.get())
                max_val = int(self.max_val.get())
            except:
                min_val, max_val = 0, 65535

            if max_val <= min_val:
                min_val = 0
                max_val = 65535 if data.dtype == np.uint16 else 255
            
            # Apply scaling
            scaling_type = self.scaling_type_var.get()
            scaled_data = self.apply_scaling(data, min_val, max_val, scaling_type)

            # Create window if needed (only once)
            if not hasattr(self, 'opencv_window_created'):
                cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
                cv2.setMouseCallback('Camera Preview', self.on_mouse_move)
                self.opencv_window_created = True

            # Update display - this is the fast path
            cv2.imshow('Camera Preview', scaled_data)
            cv2.waitKey(1)  # Minimal wait
            
            # Update pixel info less frequently to reduce overhead
            if hasattr(self, '_frame_display_count'):
                self._frame_display_count += 1
                if self._frame_display_count % 12 == 0:  # Only every 12th frame (~10 Hz at 120 FPS)
                    self.update_pixel_info(self.mouse_x, self.mouse_y)
            else:
                self._frame_display_count = 0
            
        except Exception as e:
            logging.error(f"Process frame error: {e}")
    
    def on_mouse_move(self, event, x, y, flags, param):
        """Handle mouse movement"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            self.update_pixel_info(x, y)
    
    def update_pixel_info(self, x, y):
        """Update pixel info label"""
        try:
            if self.last_frame is not None and x is not None and y is not None:
                height, width = self.last_frame.shape[:2]
                
                if 0 <= x < width and 0 <= y < height:
                    pixel_value = self.last_frame[y, x]
                    
                    if isinstance(pixel_value, (np.integer, int)):
                        info_text = f"Pixel: ({x:4d}, {y:4d}) | Value: {pixel_value:5d}"
                    else:
                        info_text = f"Pixel: ({x:4d}, {y:4d}) | Value: {pixel_value:8.1f}"
                else:
                    info_text = "Pixel: (---, ---) | Value: -----"
            else:
                info_text = "Pixel: (---, ---) | Value: -----"
                
            self.after(0, lambda: self.pixel_info_label.config(text=info_text))
            
        except Exception as e:
            logging.error(f"Error updating pixel info: {e}")

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

    def browse_output_path(self):
        """Browse for output directory"""
        from tkinter import filedialog
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_path_var.get()
        )
        if directory:
            self.output_path_var.set(directory)
            logging.info(f"Output directory set to: {directory}")
            self.update_status(f"Output: {os.path.basename(directory)}", "green")

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

    def change_trigger_source(self, selected_source):
        """Change trigger source"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        trigger_source_value = {"Internal": 1.0, "External": 2.0, "Software": 3.0}[selected_source]
        self.camera_thread.set_property('TRIGGER_SOURCE', trigger_source_value)
        self.update_status(f"Trigger source: {selected_source}", "green")

    def change_trigger_mode(self, selected_mode):
        """Change trigger mode"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        trigger_mode_value = {"Normal": 1.0, "Start": 6.0}[selected_mode]
        self.camera_thread.set_property('TRIGGER_MODE', trigger_mode_value)
        self.update_status(f"Trigger mode: {selected_mode}", "green")
    
    def change_defect_correct(self, selected_mode):
        """Change defect correction mode"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        defect_mode_value = {"OFF": 1.0, "ON": 2.0}[selected_mode]
        self.camera_thread.set_property('DEFECTCORRECT_MODE', defect_mode_value)
        self.update_status(f"Defect correction: {selected_mode}", "green")
    
    def change_hot_pixel_level(self, selected_level):
        """Change hot pixel correction level"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        level_value = {"STANDARD": 1.0, "MINIMUM": 2.0, "AGGRESSIVE": 3.0}[selected_level]
        self.camera_thread.set_property('HOTPIXELCORRECT_LEVEL', level_value)
        self.update_status(f"Hot pixel level: {selected_level}", "green")

    def change_output_trigger_kind(self, selected_kind):
        """Change output trigger kind"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        kind_value = {
            "Trigger Ready": 1.0,
            "Exposure": 2.0,
            "Programmable": 3.0,
            "Global Exposure": 4.0
        }[selected_kind]
        self.camera_thread.set_property('OUTPUT_TRIG_KIND_0', kind_value)

    def change_output_trigger_polarity(self, selected_polarity):
        """Change output trigger polarity"""
        if self.camera_thread.capturing:
            self.update_status("Cannot change during capture", "orange")
            return
        
        polarity_value = {"Positive": 1.0, "Negative": 2.0}[selected_polarity]
        self.camera_thread.set_property('OUTPUT_TRIG_POLARITY_0', polarity_value)

    def update_output_trigger_period(self, *_):
        """Update output trigger period"""
        if self.camera_thread.capturing:
            return
        try:
            period = float(self.output_trig_period_var.get())
            self.camera_thread.set_property('OUTPUT_TRIG_PERIOD_0', period)
        except ValueError:
            pass

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
                try:
                    value = float(round(float(var.get()) / 4) * 4)
                except tk.TclError:
                    continue
                self.camera_thread.set_property(f'SUBARRAY_{param}', value)
        except ValueError:
            logging.error("Invalid subarray parameters")

    def start_capture(self):
        """Start camera capture"""
        try:
            # Record button press time immediately
            button_press_time = time.time()
            
            self.update_status("Starting capture...", "blue")
            
            # Start capture immediately (no artificial delay)
            trigger_sent_time = time.time()
            logging.info(f"Button pressed at {button_press_time:.6f}, initializing camera at {trigger_sent_time:.6f}")
            
            # Store trigger info for later (will calculate expected PPS after camera ready)
            self.camera_thread.button_press_time = button_press_time
            self.camera_thread.trigger_sent_time = trigger_sent_time
            self.camera_thread.trigger_source = self.trigger_source_var.get()

            # Set up save thread if needed
            if self.save_data_var.get():
                save_queue = queue.Queue(maxsize=2000)  # Reduced for tighter memory control
                self.camera_thread.save_queue = save_queue
                object_name = self.object_name_entry.get() or "capture"
                output_path = self.output_path_var.get()
                
                # Validate output path
                if not os.path.isdir(output_path):
                    self.update_status(f"Invalid output path: {output_path}", "red")
                    return
                
                # Pass basic timing information to SaveThread
                # Expected PPS will be calculated after camera initialization completes
                timing_info = {
                    'button_press_time': button_press_time,
                    'trigger_sent_time': trigger_sent_time,
                    'trigger_source': self.trigger_source_var.get(),
                    'exposure_time': self.exposure_time_var.get() / 1000.0  # Convert ms to seconds
                }
                    
                self.save_thread = SaveThread(save_queue, self.camera_thread, object_name,
                                             self.shared_data, self.cube_size_var.get(), output_path, timing_info)
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
            
        except Exception as e:
            logging.error(f"Stop capture error: {e}")
            self.update_status(f"Error: {e}", "red")

    def disable_controls_during_capture(self):
        """Disable controls during capture"""
        controls = [
            self.exposure_time_entry, self.save_data_checkbox, 
            self.start_button, self.binning_menu,
            self.bit_depth_menu, self.readout_speed_menu, self.sensor_mode_menu,
            self.subarray_mode_menu, self.restore_config_button
        ]
        
        for widget in controls:
            widget.config(state='disabled')

        self.stop_button.config(state='normal')
        
        if self.subarray_mode_var.get() == "On":
            for entry in self.subarray_entries.values():
                entry.config(state='disabled')

    def enable_controls_after_capture(self):
        """Enable controls after capture"""
        controls = [
            self.exposure_time_entry, self.save_data_checkbox,
            self.start_button, self.binning_menu,
            self.bit_depth_menu, self.readout_speed_menu, self.sensor_mode_menu,
            self.subarray_mode_menu, self.restore_config_button
        ]
        
        for widget in controls:
            widget.config(state='normal')

        self.stop_button.config(state='disabled')

        if self.subarray_mode_var.get() == "On":
            for entry in self.subarray_entries.values():
                entry.config(state='normal')

    def on_close(self):
        """Handle application close with safe shutdown order"""
        try:
            logging.info("=" * 60)
            logging.info("APPLICATION SHUTDOWN INITIATED")
            logging.info("=" * 60)
            
            # Step 1: Save configuration
            if hasattr(self, 'config_manager'):
                try:
                    self.config_manager.save_config(self)
                    logging.info("✓ Configuration saved on exit")
                except Exception as e:
                    logging.error(f"Failed to save config: {e}")
            
            # Step 2: Stop GUI update loops FIRST (prevents race conditions)
            logging.info("Stopping GUI update loops...")
            self.updating_camera_status = False
            self.updating_frame_display = False
            time.sleep(0.3)  # Give update loops time to exit
            logging.info("✓ GUI update loops stopped")
            
            # Step 3: Close OpenCV windows BEFORE stopping camera
            # (prevents display updates while camera is shutting down)
            try:
                logging.info("Closing OpenCV windows...")
                cv2.destroyAllWindows()
                cv2.waitKey(100)  # Longer wait for proper cleanup
                logging.info("✓ OpenCV windows closed")
            except Exception as e:
                logging.error(f"Error closing OpenCV windows: {e}")
            
            # Step 4: Stop save thread (give it time to finish writing)
            if self.save_thread and self.save_thread.is_alive():
                logging.info("Stopping save thread (waiting up to 30s for file writes to complete)...")
                self.save_thread.stop()
                self.save_thread.join(timeout=30)
                if self.save_thread.is_alive():
                    logging.warning("⚠ Save thread did not stop within timeout")
                else:
                    logging.info("✓ Save thread stopped")
            
            # Step 5: Stop camera thread (give it time for proper DCAM cleanup)
            if self.camera_thread:
                logging.info("Stopping camera thread (waiting up to 10s)...")
                self.camera_thread.stop()
                self.camera_thread.join(timeout=10)
                if self.camera_thread.is_alive():
                    logging.warning("⚠ Camera thread did not stop within timeout")
                else:
                    logging.info("✓ Camera thread stopped")
            
            # Step 6: Destroy GUI
            logging.info("Destroying GUI...")
            self.quit()
            self.destroy()
            
            logging.info("=" * 60)
            logging.info("APPLICATION CLOSED SUCCESSFULLY")
            logging.info("=" * 60)
            
        except Exception as e:
            logging.error(f"Error during close: {e}", exc_info=True)
            # Force exit on error to prevent hang
            import sys
            logging.error("Forcing exit due to shutdown error")
            sys.exit(1)


def main():
    """Main application entry point"""
    try:
        logging.info("Starting Simple Camera Control")
        
        # Create shared resources
        shared_data = SharedData()
        frame_queue = queue.Queue(maxsize=5)
        
        # Create GUI
        app = CameraGUI(shared_data, None, frame_queue)
        
        # Create and start camera thread
        camera_thread = CameraThread(shared_data, frame_queue, app)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Set thread reference in GUI
        app.camera_thread = camera_thread
        
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