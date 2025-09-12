import ctypes
import sys
import subprocess
from ctypes import byref, c_int, c_uint32, c_void_p
from astropy.time import Time

# Load the Meinberg shared library
libmbg = ctypes.CDLL('libmbgdevio.so')

# Define necessary structures
class PCPS_HR_TIME(ctypes.Structure):
    _fields_ = [
        ("tstamp_sec", c_uint32),
        ("tstamp_frac", c_uint32),
        ("signal", c_uint32),
        ("status", c_uint32)
    ]

class PCPS_UCAP_ENTRIES(ctypes.Structure):
    _fields_ = [
        ("used", c_uint32),
        ("max", c_uint32)
    ]

# Function prototypes
libmbg.mbg_find_devices.restype = c_int
libmbg.mbg_open_device.argtypes = [c_int]
libmbg.mbg_open_device.restype = c_void_p
libmbg.mbg_get_ucap_entries.argtypes = [c_void_p, ctypes.POINTER(PCPS_UCAP_ENTRIES)]
libmbg.mbg_get_ucap_entries.restype = c_int
libmbg.mbg_get_ucap_event.argtypes = [c_void_p, ctypes.POINTER(PCPS_HR_TIME)]
libmbg.mbg_get_ucap_event.restype = c_int
libmbg.mbg_close_device.argtypes = [c_void_p]
libmbg.mbg_close_device.restype = c_int
libmbg.mbg_clr_ucap_buff.argtypes = [c_void_p]
libmbg.mbg_clr_ucap_buff.restype = c_int

def get_first_timestamp():
    # Find the number of devices
    num_devices = libmbg.mbg_find_devices()
    if num_devices == 0:
        print("No supported Meinberg device found")
        return None

    # Open the first device
    device_handle = libmbg.mbg_open_device(0)
    if not device_handle:
        print("Failed to open Meinberg device")
        return None

    # Check the buffer for entries
    entries = PCPS_UCAP_ENTRIES()
    rc = libmbg.mbg_get_ucap_entries(device_handle, byref(entries))
    if rc != 0:
        print(f"Error reading buffer entries: {rc}")
        #libmbg.mbg_close_device(device_handle)
        return None

    if entries.used == 0:
        print("No timestamps available in buffer")
        #libmbg.mbg_close_device(device_handle)
        return None
    

    # Retrieve the first timestamp from the buffer
    ucap_event = PCPS_HR_TIME()
    print(ucap_event)
    rc = libmbg.mbg_get_ucap_event(device_handle, byref(ucap_event))
    #libmbg.mbg_close_device(device_handle)

    if rc == 0:
        # Convert UNIX time + fractional seconds to Astropy Time format
        unix_time = ucap_event.tstamp_sec + ucap_event.tstamp_frac / 2**32
        astropy_time = Time(unix_time, format='unix',precision=9)
        print(astropy_time.isot)
        return astropy_time
    else:
        print(f"Error retrieving timestamp: {rc}")
        return None

def clear_buffer():
    # Find the number of devices
    num_devices = libmbg.mbg_find_devices()
    if num_devices == 0:
        print("No supported Meinberg device found")
        return False

    # Open the first device
    device_handle = libmbg.mbg_open_device(0)
    if not device_handle:
        print("Failed to open Meinberg device")
        return False

    # Clear the buffer
    rc = libmbg.mbg_clr_ucap_buff(device_handle)
    if rc != 0:
        print(f"Error clearing buffer: {rc}")
        return False

    print("Buffer cleared successfully")
    return True

# Example usage
if __name__ == "__main__":
    timestamp = get_first_timestamp()
    if timestamp:
       print(f"First timestamp in Astropy format: {timestamp.iso}")
    # Clear the buffer
    clear_buffer()

