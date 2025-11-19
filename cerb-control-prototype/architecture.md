# Camera Control Software Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
│                           ↓                                  │
│                    [GUI Window]                              │
│         (Tkinter - buttons, sliders, settings)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────┴───────┐
                    │               │
                    ↓               ↓
        ┌──────────────────┐  ┌──────────────────┐
        │  Camera Thread   │  │   Save Thread    │
        │  (Captures)      │  │   (Writes FITS)  │
        └──────────────────┘  └──────────────────┘
                    ↓               ↑
                    │               │
                    └──→ [Queues] ──┘
                    
                    ↓
        ┌──────────────────┐
        │  OpenCV Window   │
        │  (Live Preview)  │
        └──────────────────┘
```

---

## The Three Main Threads

Your software runs **three independent threads** that work simultaneously:

### 1. **Main Thread (GUI)**
**What it does**: Shows buttons, handles clicks, updates displays

**Think of it as**: The "front desk" - it talks to you and coordinates everything

**Key responsibilities**:
- Display the control panel (buttons, sliders)
- Show status messages ("Capturing...", "Stopped", etc.)
- Update performance stats (FPS counters)
- Show live camera preview window

### 2. **Camera Thread**
**What it does**: Talks directly to the camera hardware, grabs frames

**Think of it as**: The "photographer" - constantly asking camera "got a new frame?"

**The continuous loop**:
```python
while running:
    if capturing:
        # Ask camera: "Do you have a new frame ready?"
        if camera.wait_for_frame(timeout=100ms):
            # Grab the frame data
            frame = camera.get_frame()
            timestamp = camera.get_timestamp()
            
            # Put frame in queue for display
            frame_queue.put(frame)
            
            # Put frame + timestamp in queue for saving
            if saving_enabled:
                save_queue.put((frame, timestamp))
    else:
        sleep(0.1 seconds)  # Idle, not capturing
```

**Key insight**: This thread runs as fast as the camera can produce frames (up to 100+ FPS). It **never** does slow operations like file writing - that would cause frame drops!

### 3. **Save Thread**
**What it does**: Writes frames to disk as FITS files

**Think of it as**: The "archivist" - collects 100 frames, bundles them, writes to disk

**The batching process**:
```python
frame_buffer = []

while saving:
    # Collect frames from queue
    frame, timestamp = save_queue.get()
    frame_buffer.append(frame)
    timestamp_buffer.append(timestamp)
    
    # When we have 100 frames...
    if len(frame_buffer) >= 100:
        # Stack into a 3D cube (100 frames × height × width)
        data_cube = stack(frame_buffer)
        
        # Write to FITS file (slow operation, but happens in background)
        write_fits("capture_cube001.fits", data_cube, timestamps)
        
        # Clear buffers for next batch
        frame_buffer.clear()
```

**Key insight**: Writing files is **slow** (takes ~100-500ms per cube). By doing this in a separate thread, the camera thread can keep capturing without interruption.

---

## The Communication System: Queues

**Queues** are like pipes between threads. Thread A puts data in one end, Thread B takes it out the other end.

```
Camera Thread  ──[frame data]──→  frame_queue  ──→  GUI Thread (display)
                                                       ↓
                                                   [Shows in OpenCV window]

Camera Thread  ──[frame + timestamp]──→  save_queue  ──→  Save Thread
                                                             ↓
                                                        [Writes FITS file]
```

**Why queues?**
- Threads run at different speeds (camera: 100 FPS, display: 10-120 FPS, saving: ~1 cube/sec)
- Queues act as **buffers** - if one thread is temporarily slow, frames pile up in the queue
- If queue gets too full, we **drop old frames** (better to show recent frame than old one)

---

## Data Flow: From Photons to Disk

Let's trace a single frame through the entire system:

### Step 1: **Photon hits camera sensor**
```
Light → Camera sensor → Camera electronics
```
- Camera accumulates photons during exposure time (e.g., 10ms)
- Converts to digital values (0-65535 for 16-bit)

### Step 2: **Camera Thread reads frame**
```python
# Camera thread's main loop
frame_data = camera.get_frame()  # Raw 16-bit array: 4096×2304 pixels
timestamp = camera.get_timestamp()  # Camera's internal clock (in seconds)
framestamp = camera.get_framestamp()  # Frame counter (0, 1, 2, ...)
```

**Important**: The camera has its own clock! Timestamps come from camera hardware, not your computer's clock. This is crucial for precision timing.

### Step 3: **Frame goes to display queue**
```python
# Only keep the most recent frame for display
while not frame_queue.empty():
    frame_queue.get_nowait()  # Throw away old frames
    
frame_queue.put_nowait(frame_data)  # Add new frame
```

**Why drop old frames?** 
- Display can only show ~10-120 frames/sec
- Camera might produce 100 frames/sec
- Better to skip frames and stay current than show stale data

### Step 4: **GUI displays frame** (every 8ms = 120 Hz max)
```python
# GUI thread's display loop (runs every 8ms)
if not frame_queue.empty():
    frame = frame_queue.get()
    
    # Scale from 16-bit (0-65535) to 8-bit (0-255) for display
    # Using LUT (lookup table) for speed
    display_frame = apply_scaling_LUT(frame, min_val, max_val)
    
    # Show in OpenCV window
    cv2.imshow('Camera Preview', display_frame)
```

**The LUT trick** (Lookup Table):
Instead of calculating `(pixel - min) * 255 / (max - min)` for 9 million pixels...
```python
# Pre-compute once:
lut = [(i - min) * 255 / (max - min) for i in range(65536)]

# Then for each frame, just look up:
display_frame = lut[frame]  # Super fast!
```

### Step 5: **Frame goes to save queue** (if saving enabled)
```python
if saving_enabled:
    save_queue.put((frame, timestamp, framestamp))
```

### Step 6: **Save thread batches frames**
```python
# Collect 100 frames
frame_buffer = [frame1, frame2, ..., frame100]
timestamp_buffer = [t1, t2, ..., t100]

# Stack into 3D cube
data_cube = numpy.array(frame_buffer)  # Shape: (100, 2304, 4096)
```

### Step 7: **Write FITS file**
```python
# Create FITS structure
fits_file = {
    'PRIMARY': {header_info},
    'DATA_CUBE': data_cube,  # The 100 images
    'TIMESTAMPS': {
        'TIMESTAMP': [t1, t2, ..., t100],
        'FRAMESTAMP': [0, 1, ..., 99]
    }
}

# Write to disk (slow, but happens in background thread)
fits_file.save('capture_cube001.fits')
```

---

## Thread Synchronization: Locks

**Problem**: What if the Camera Thread tries to read a camera setting while the GUI Thread is changing it?

**Solution**: **Locks** - like a bathroom door lock. Only one thread can hold the lock at a time.

```python
# GUI Thread wants to change exposure
def change_exposure(new_value):
    lock.acquire()  # Wait for camera to finish what it's doing
    camera.set_exposure(new_value)
    lock.release()  # Okay, camera thread can use camera again

# Camera Thread capturing frames
def capture_frame():
    lock.acquire()  # Wait if GUI is changing settings
    frame = camera.get_frame()
    lock.release()  # Done, GUI can change settings now
```

**We have different locks for different things**:
- `capture_lock` - For actually grabbing frames
- `property_lock` - For reading/changing camera settings
- `buffer_lock` - For allocating/releasing camera memory

---

## The Timestamp System

Your camera has a **hardware timestamp** that's separate from your computer's clock:

```
Frame 0: timestamp = 0.000000 sec, framestamp = 0
Frame 1: timestamp = 0.010000 sec, framestamp = 1  (10ms later)
Frame 2: timestamp = 0.020000 sec, framestamp = 2
...
Frame 65535: timestamp = 655.350000 sec, framestamp = 65535
Frame 65536: timestamp = 0.000000 sec, framestamp = 0  ← ROLLOVER!
```

**The rollover problem**:
- Timestamp is 32-bit microseconds → wraps at ~4295 seconds (~71 minutes)
- Framestamp is 16-bit counter → wraps at 65536 frames

**The solution**:
```python
# Detect rollover
if new_timestamp < old_timestamp - 4000:
    timestamp_offset += 4294.967296  # Add one full cycle
    
corrected_timestamp = new_timestamp + timestamp_offset
```

This lets you run for hours without losing timing accuracy!

---

## Performance Optimization: Why 120 Hz Display?

**The challenge**: 
- Camera produces 4096×2304 = 9,437,184 pixels per frame
- At 100 FPS = 943,718,400 pixels/second
- Each needs scaling, conversion, display

**Optimization 1: LUT-based scaling**
```
OLD WAY (slow):
    for each pixel:
        output = (pixel - min) * 255 / (max - min)  # Math for 9M pixels!

NEW WAY (fast):
    lut = pre_computed_lookup_table[65536 values]  # Do math once
    output = lut[pixels]  # Just array indexing - super fast!
```
**Speed improvement**: ~10-50× faster

**Optimization 2: Aggressive frame dropping**
```python
# Always show the NEWEST frame, throw away everything else
while frame_queue.qsize() > 0:
    old_frame = frame_queue.get()  # Discard
    
frame_queue.put(newest_frame)  # Show this one
```

**Optimization 3: Skip expensive operations**
```python
# Only update pixel info every 12th frame
if frame_count % 12 == 0:
    update_pixel_info()  # This is slow (Tkinter call)
```

---

## The Trigger System

Your camera can run in different modes:

### Internal Triggering (Default)
```
Camera: "I'll capture frames as fast as I can!"
Frame timing: Exposure time + readout time = ~10ms between frames
```

### External Triggering - Normal Mode
```
External Device → [Pulse] → Camera captures ONE frame
External Device → [Pulse] → Camera captures ONE frame
...

Use case: GPS PPS signal, laser trigger, etc.
```

### External Triggering - Start Mode
```
External Device → [Single Pulse] → Camera starts capturing continuously
                                     (runs until you hit STOP)

Use case: GPS-synchronized observations starting exactly at a specific time
```

### Output Triggers
```
Camera ────[Output Trigger]────→ External Device

Examples:
- "Exposure" trigger: Pulse during exposure → sync shutter, laser, LED
- "Programmable" trigger: Periodic pulses → trigger another camera
```

---

## Configuration Management

**Problem**: You set up 20 different parameters. Next time you open the software, you have to set them all again!

**Solution**: Auto-save to JSON file

```python
# On every change, save to config file
config = {
    "exposure_time_ms": 100,
    "binning": "1x1",
    "trigger_source": "Internal",
    "min_count": 200,
    "max_count": 300,
    # ... all your settings
}

save_json("camera_config.json", config)

# On startup, load settings
config = load_json("camera_config.json")
apply_settings(config)
```

The "Restore Settings" button reloads the last saved configuration.

---

## Error Handling & Recovery

### Problem 1: Camera stops responding
```python
# Watchdog timer
last_frame_time = current_time

# In main loop:
if current_time - last_frame_time > 5 seconds:
    logging.error("Camera stopped responding!")
    # Try to restart capture
    reset_camera()
```

### Problem 2: Secure Boot blocks drivers
```python
# During installation
if secure_boot_enabled:
    print("ERROR: Secure Boot blocks unsigned drivers")
    print("Solution: Disable Secure Boot in BIOS")
```

### Problem 3: Queue fills up (save thread too slow)
```python
if save_queue.qsize() > 10000:
    logging.warning("Save queue backing up - frames will be dropped!")
```

---

## Memory Management

**Challenge**: 100 FPS × 16-bit × 4096×2304 pixels = ~1.7 GB/second!

**Solutions**:

1. **Circular buffer in camera**:
```
Camera has 200 frame buffers in hardware:
[0][1][2]...[199][0][1]...  ← Reuses memory
        ↑
   Read pointer follows write pointer
```

2. **Copy frames immediately**:
```python
frame = camera.get_frame()  # This is a reference to camera memory
frame_copy = np.copy(frame)  # Make our own copy
# Camera can now reuse that buffer for next frame
```

3. **Batch writing**:
```python
# Don't write 100 separate files
# Write one file with 100 frames
data_cube = stack_100_frames()  # 3D array
write_fits(data_cube)  # One I/O operation
```

---

## Summary: The Big Picture

```
YOU (the user)
    ↓
[GUI Thread]
    ├─→ Clicks buttons → sends commands to Camera Thread
    ├─→ Reads frame_queue → displays in OpenCV window
    └─→ Updates status, FPS counters
    
[Camera Thread] ← Actually talks to camera hardware
    ├─→ Continuous loop: "Got frame? Got frame? Got frame?"
    ├─→ Puts frames in frame_queue (for display)
    └─→ Puts frames in save_queue (for saving)
    
[Save Thread]
    ├─→ Collects 100 frames from save_queue
    ├─→ Stacks into 3D cube
    └─→ Writes FITS file to disk
    
[Camera Hardware]
    ├─→ Has internal clock (timestamps)
    ├─→ Can be triggered externally
    └─→ Can output triggers to other equipment
```

**Key Insights**:
1. **Three threads** = three things happening simultaneously
2. **Queues** = how threads communicate without blocking each other
3. **Locks** = prevent threads from interfering with each other
4. **LUT scaling** = why display is so fast
5. **Batching** = why saving doesn't slow down capture
6. **Hardware timestamps** = why timing is microsecond-accurate

---

## Common Operations Explained

### "I click Start Streaming"
```
1. GUI thread: start_capture() called
2. GUI thread → Camera thread: "Start capturing!"
3. Camera thread: 
   - Allocates 200 frame buffers in camera memory
   - Enables timestamp producer
   - Starts hardware capture
   - Enters fast loop: wait_for_frame() → get_frame() → queue
4. GUI thread: Updates button to "Stop Streaming", disables settings
5. Camera thread now produces ~100 frames/second
6. GUI thread displays ~10-120 frames/second
7. Save thread (if enabled) writes ~1 cube/second
```

### "I change exposure time"
```
1. GUI thread: You type new value
2. GUI thread: "Hey Camera thread, change exposure!"
3. GUI thread: Acquires property_lock (waits if camera busy)
4. GUI thread: camera.set_exposure(new_value)
5. GUI thread: Releases property_lock
6. Camera thread: Next frames use new exposure time
```

### "Frames appear in OpenCV window"
```
Every 8ms (120 Hz):
1. GUI timer triggers: update_frame_display()
2. Check frame_queue: Is there a new frame?
3. If yes:
   - Get frame from queue
   - Apply LUT scaling (16-bit → 8-bit)
   - cv2.imshow(scaled_frame)
4. Schedule next update in 8ms
```

This architecture lets you:
- **Capture** at 100+ FPS
- **Display** smoothly at up to 120 FPS  
- **Save** continuously to disk
- **Never drop frames** (unless disk is too slow)

All running simultaneously on your 24-core Threadripper! 🚀