#!/usr/bin/env python3
"""
Las Campanas Observatory - Telescope Control System GUI
A simple GUI to display telescope telemetry data from the TCS.
Based on socket communication pattern similar to als_engineering_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import socket
import threading
from queue import Queue, Empty
from datetime import datetime
import time
import re

def _timestamp():
    return datetime.now().isoformat(timespec='seconds')


class TCSData:
    """Class to hold parsed TCS data"""
    def __init__(self, request, response, timestamp):
        self.request = request
        self.response = response  
        self.timestamp = timestamp
        self.parsed_data = self._parse_response(request, response)
        
    def _parse_response(self, request, response):
        """Parse TCS response based on request type"""
        try:
            parts = response.strip().split()
            
            if request == "datetime":
                if len(parts) >= 3:
                    return {
                        'date': parts[0],
                        'time': parts[1], 
                        'sidereal_time': parts[2]
                    }
                    
            elif request == "telpos":
                if len(parts) >= 6:
                    return {
                        'ra': parts[0],
                        'dec': parts[1],
                        'equinox': parts[2],
                        'hour_angle': parts[3],
                        'airmass': parts[4],
                        'rotator_angle': parts[5]
                    }
                    
            elif request == "teldata":
                if len(parts) >= 10:
                    return {
                        'roi': parts[0],
                        'guider_status': parts[1],
                        'mount_guider_motion': parts[2],
                        'mount_motion': parts[3],
                        'azimuth': parts[4],
                        'elevation': parts[5],
                        'zenith': parts[6],
                        'parallactic_angle': parts[7],
                        'dome_azimuth': parts[8],
                        'dome_status': parts[9]
                    }
            
            return None
            
        except Exception as e:
            print(f"Error parsing {request} response: {e}")
            return None


class TCSHandler:
    """Handles socket read/write and queues for GUI"""
    def __init__(self, ip_address, port):
        self._socket = None
        self._read_thread = None
        self._write_thread = None
        self._data_buffer = b''
        
        self.read_queue = Queue()
        self.write_queue = Queue()
        self._running = False
        
        self.ip_address = ip_address
        self.port = port
        
    def connect(self):
        """Connect to TCS"""
        try:
            self._socket = socket.create_connection((self.ip_address, self.port), timeout=5)
            self._socket.settimeout(1)
            self._running = True
            
            self._read_thread = threading.Thread(target=self._read_worker, daemon=True)
            self._write_thread = threading.Thread(target=self._write_worker, daemon=True)
            
            self._read_thread.start()
            self._write_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from TCS"""
        self._running = False
        
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
            
    def _read_worker(self):
        """Read data from TCS socket"""
        while self._running:
            try:
                data = self._socket.recv(1024)
                if data == b'':
                    print('TCS closed connection')
                    break
                    
                self._data_buffer += data
                
                # Split on newline characters
                lines = self._data_buffer.split(b'\n')
                
                # Process complete lines
                while len(lines) > 1:
                    line = lines[0].strip()
                    if line:
                        self.read_queue.put(line.decode('ascii', errors='ignore'))
                    lines = lines[1:]
                    
                # Keep remaining partial line in buffer
                self._data_buffer = lines[0]
                
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"Read error: {e}")
                break
                
    def _write_worker(self):
        """Write commands to TCS socket"""
        while self._running:
            try:
                message = self.write_queue.get(timeout=1)
                if self._socket:
                    self._socket.sendall(f"{message}\n".encode('ascii'))
                    
            except Empty:
                continue
            except Exception as e:
                if self._running:
                    print(f"Write error: {e}")
                break
                
    def send_command(self, command):
        """Send command to TCS"""
        if self._running:
            self.write_queue.put(command)


class TCSWorker:
    """Manages TCS communication and data processing"""
    def __init__(self, handler):
        self._handler = handler
        self.gui_queue = Queue()
        self._running = False
        
        # Current telescope data
        self.current_data = {
            'ra': '--:--:--.--',
            'dec': '±--:--:--.--', 
            'elevation': '--.----°',
            'azimuth': '---.---°',
            'airmass': '-.---',
            'rotator_angle': '--.----°',
            'hour_angle': '--:--:--',
            'parallactic_angle': '---.----°',
            'sidereal_time': '--:--:--',
            'date': '----/--/--',
            'time': '--:--:--',
            'dome_status': 'UNKNOWN',
            'guider_status': 'UNKNOWN',
            'mount_status': 'UNKNOWN',
            'rotator_status': 'UNKNOWN',
            'roi': 'UNKNOWN',
            'connection_status': 'Disconnected'
        }
        
        self._reader_thread = None
        self._status_thread = None
        
    def start(self):
        """Start worker threads"""
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_listener, daemon=True)
        self._status_thread = threading.Thread(target=self._request_data, daemon=True) 
        
        self._reader_thread.start()
        self._status_thread.start()
        
    def stop(self):
        """Stop worker threads"""
        self._running = False
        self._handler.disconnect()
        
    def send_command(self, command):
        """Send command to TCS"""
        self._handler.send_command(command)
        self.gui_queue.put(('log', f"{_timestamp()} TX: {command}"))
        
    def _read_listener(self):
        """Listen for responses from TCS"""
        while self._running:
            try:
                response = self._handler.read_queue.get(timeout=1)
                
                # Log the response
                self.gui_queue.put(('log', f"{_timestamp()} RX: {response}"))
                
                # Try to parse the response and determine what type it is
                self._process_response(response)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Read listener error: {e}")
                
    def _process_response(self, response):
        """Process TCS response based on official TCS documentation format"""
        try:
            parts = response.strip().split()
            if not parts:
                return
                
            print(f"Processing response with {len(parts)} parts: {parts[:3]}...")  # Debug
                
            # Based on TCS documentation Table 2:
            # datetime: dateobs telut telst (3 parts)
            # telpos: telra teldc telep telha telam rotangle (6 parts)  
            # teldata: roi telguide gdrmountmv mountmv telaz telel zd telpa teldm dmstat (10 parts)
            
            if len(parts) == 3 and '-' in parts[0] and ':' in parts[1] and ':' in parts[2]:
                # datetime response: "2025-09-10 13:50:25 08:27:03"
                self._update_datetime_data(parts)
                self.gui_queue.put(('data_update', 'datetime'))
                print("Updated datetime data")
                    
            elif len(parts) == 6 and ':' in parts[0] and ':' in parts[1]:
                # telpos response: "08:26:02.36 -28:58:34.3 2000.00  -00:00:01  1.000  172.2522"
                self._update_position_data(parts)
                self.gui_queue.put(('data_update', 'telpos'))
                print("Updated position data")
                    
            elif len(parts) == 10 and parts[0].isdigit():
                # teldata response: "3 00 000 0111  168.0006  89.9280   0.0720 -017.7478  045       0"
                self._update_teldata(parts)
                self.gui_queue.put(('data_update', 'teldata'))
                print("Updated teldata")
                
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Response was: {response}")
                
    def _update_datetime_data(self, parts):
        """Update datetime data: 2025-09-10 13:50:25 08:27:03"""
        try:
            self.current_data['date'] = parts[0]
            self.current_data['time'] = parts[1]
            self.current_data['sidereal_time'] = parts[2]
            print(f"DateTime updated: {parts[0]} {parts[1]} {parts[2]}")
        except Exception as e:
            print(f"Error updating datetime: {e}")
            
    def _update_position_data(self, parts):
        """Update position data: 08:26:02.36 -28:58:34.3 2000.00  -00:00:01  1.000  172.2522"""
        try:
            if len(parts) >= 6:
                self.current_data['ra'] = parts[0]  # telra
                self.current_data['dec'] = parts[1]  # teldc
                # parts[2] = telep (equinox)
                self.current_data['hour_angle'] = parts[3]  # telha
                self.current_data['airmass'] = parts[4]  # telam
                self.current_data['rotator_angle'] = f"{parts[5]}°"  # rotangle
                print(f"Position updated: RA={parts[0]}, DEC={parts[1]}, Airmass={parts[4]}")
        except Exception as e:
            print(f"Error updating position: {e}")
            
    def _update_teldata(self, parts):
        """Update telescope data: 3 00 000 0111  168.0006  89.9280   0.0720 -017.7478  045       0"""
        try:
            if len(parts) >= 10:
                # Based on TCS documentation Table 2:
                # teldata: roi telguide gdrmountmv mountmv telaz telel zd telpa teldm dmstat
                roi = parts[0]  # Rotator of interest
                telguide = parts[1]  # Guider status  
                gdrmountmv = parts[2]  # Mount and guider motion
                mountmv = parts[3]  # Mount and rotator motion
                telaz = parts[4]  # Azimuth
                telel = parts[5]  # Elevation
                zd = parts[6]  # Zenith distance
                telpa = parts[7]  # Parallactic angle
                teldm = parts[8]  # Dome azimuth
                dmstat = parts[9]  # Dome status
                
                # Update display values
                self.current_data['azimuth'] = f"{telaz}°"
                self.current_data['elevation'] = f"{telel}°"
                self.current_data['parallactic_angle'] = f"{telpa}°"
                
                # Parse ROI (Table 6)
                roi_map = {'0': 'NASW', '1': 'NASE', '2': 'CASS', '3': 'AUX1', '4': 'AUX2', '5': 'AUX3'}
                self.current_data['roi'] = roi_map.get(roi, f"ROI_{roi}")
                
                # Parse dome status (Table 2)
                dome_status_map = {'0': 'CLOSED', '1': 'OPEN', '-1': 'UNKNOWN'}
                self.current_data['dome_status'] = dome_status_map.get(dmstat, f"STATUS_{dmstat}")
                
                # Parse telguide (Table 2: ab where a=tracking, b=active_guider)
                if len(telguide) >= 2:
                    tracking = telguide[0] == '1'
                    active_guider = int(telguide[1]) if telguide[1].isdigit() else 0
                    
                    if tracking:
                        guider_names = {0: 'NOT_GUIDING', 1: 'GUIDER_1', 2: 'GUIDER_2'}
                        self.current_data['guider_status'] = f"TRACKING - {guider_names.get(active_guider, 'UNKNOWN')}"
                    else:
                        self.current_data['guider_status'] = 'NOT TRACKING'
                else:
                    self.current_data['guider_status'] = 'UNKNOWN'
                    
                # Parse gdrmountmv (Table 2: abc where a=mount, b=guider1, c=guider2)
                if len(gdrmountmv) >= 3:
                    mount_status = int(gdrmountmv[0])
                    mount_status_map = {0: 'NOT_MOVING', 1: 'MOVING', 2: 'REJECTED', 3: 'LIMIT', 9: 'FAILURE'}
                    self.current_data['mount_status'] = mount_status_map.get(mount_status, f"STATUS_{mount_status}")
                else:
                    self.current_data['mount_status'] = 'UNKNOWN'
                    
                # Parse mountmv (Table 2: abcd where a=slewing, b=closed_loop, c=in_position, d=in_position_time)
                if len(mountmv) >= 4:
                    slewing = mountmv[0] == '1'
                    closed_loop = mountmv[1] == '1'
                    in_position = mountmv[2] == '1'
                    in_position_time = mountmv[3] == '1'
                    
                    if slewing:
                        self.current_data['rotator_status'] = 'SLEWING'
                    elif in_position_time:
                        self.current_data['rotator_status'] = 'IN POSITION'
                    elif in_position:
                        self.current_data['rotator_status'] = 'MOVING TO POSITION'
                    elif closed_loop:
                        self.current_data['rotator_status'] = 'CLOSED LOOP'
                    else:
                        self.current_data['rotator_status'] = 'OPEN LOOP'
                else:
                    self.current_data['rotator_status'] = 'UNKNOWN'
                    
                print(f"Teldata updated: Az={telaz}°, El={telel}°, Dome={self.current_data['dome_status']}")
                            
        except Exception as e:
            print(f"Error updating teldata: {e}")
            print(f"Parts were: {parts}")
                
    def _request_data(self):
        """Periodically request telescope data"""
        request_cycle = ['datetime', 'telpos', 'teldata']
        cycle_index = 0
        
        while self._running:
            if self._handler._running:
                command = request_cycle[cycle_index]
                self._handler.send_command(command)
                self.gui_queue.put(('log', f"{_timestamp()} TX: {command}"))
                cycle_index = (cycle_index + 1) % len(request_cycle)
                
            time.sleep(2)  # Request data every 2 seconds


class TCSGUI:
    """Main TCS GUI application"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Las Campanas Observatory - TCS Monitor')
        self.root.geometry('1200x800')
        self.root.configure(bg='#0c1445')
        
        # TCS connection settings
        self.tcs_ip = tk.StringVar(value="200.28.147.28")
        self.tcs_port = tk.StringVar(value="5800")
        
        # Initialize TCS components
        self.handler = None
        self.worker = None
        
        self.is_connected = False
        self.auto_update = False
        
        # Offset parameters
        self.offset_ns = tk.StringVar(value="5.0")  # North/South offset in arcseconds
        self.offset_ew = tk.StringVar(value="5.0")  # East/West offset in arcseconds
        
        self.setup_gui()
        self.update_display()
        
        # Handle window closing
        self.root.protocol('WM_DELETE_WINDOW', self.on_closing)
        
    def setup_gui(self):
        """Setup GUI components"""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#0c1445')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="Las Campanas Observatory\nTelescope Control System Monitor",
                              bg='#0c1445', fg='#00ff41',
                              font=('Courier', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Connection frame
        self.setup_connection_frame(main_frame)
        
        # Create notebook for data display
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Data tabs
        self.setup_coordinates_tab()
        self.setup_position_tab()
        self.setup_status_tab()
        self.setup_offsets_tab()
        self.setup_log_tab()
        
        # Control frame
        self.setup_control_frame(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
        
    def setup_connection_frame(self, parent):
        """Setup connection controls"""
        conn_frame = tk.LabelFrame(parent, text="Connection Settings", 
                                  bg='#1a1a2e', fg='#00ff41',
                                  font=('Courier', 10, 'bold'))
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # IP Address
        tk.Label(conn_frame, text="TCS IP:", bg='#1a1a2e', fg='#ffffff').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ip_entry = tk.Entry(conn_frame, textvariable=self.tcs_ip, width=15)
        ip_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Port
        tk.Label(conn_frame, text="Port:", bg='#1a1a2e', fg='#ffffff').grid(row=0, column=2, sticky='w', padx=5, pady=5)
        port_entry = tk.Entry(conn_frame, textvariable=self.tcs_port, width=8)
        port_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Connect button
        self.connect_btn = tk.Button(conn_frame, text="Connect",
                                    command=self.toggle_connection,
                                    bg='#00ff41', fg='#000000',
                                    font=('Courier', 10, 'bold'))
        self.connect_btn.grid(row=0, column=4, padx=10, pady=5)
        
    def setup_coordinates_tab(self):
        """Setup coordinates display tab"""
        coords_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(coords_frame, text="Coordinates")
        
        self.ra_label = self.create_data_row(coords_frame, "Right Ascension:", "--:--:--.--", 0)
        self.dec_label = self.create_data_row(coords_frame, "Declination:", "±--:--:--.--", 1)
        self.ha_label = self.create_data_row(coords_frame, "Hour Angle:", "--:--:--", 2)
        self.sidereal_label = self.create_data_row(coords_frame, "Sidereal Time:", "--:--:--", 3)
        self.date_label = self.create_data_row(coords_frame, "Date:", "----/--/--", 4)
        self.time_label = self.create_data_row(coords_frame, "Time:", "--:--:--", 5)
        
    def setup_position_tab(self):
        """Setup position display tab"""
        position_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(position_frame, text="Position")
        
        self.elevation_label = self.create_data_row(position_frame, "Elevation:", "--.----°", 0)
        self.azimuth_label = self.create_data_row(position_frame, "Azimuth:", "---.---°", 1)
        self.airmass_label = self.create_data_row(position_frame, "Airmass:", "-.---", 2)
        self.rotator_label = self.create_data_row(position_frame, "Rotator Angle:", "--.----°", 3)
        self.parallactic_label = self.create_data_row(position_frame, "Parallactic Angle:", "---.----°", 4)
        
    def setup_status_tab(self):
        """Setup status display tab"""
        status_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(status_frame, text="Status")
        
        self.dome_status_label = self.create_data_row(status_frame, "Dome Status:", "UNKNOWN", 0)
        self.guider_status_label = self.create_data_row(status_frame, "Guider Status:", "UNKNOWN", 1)
        self.mount_status_label = self.create_data_row(status_frame, "Mount Status:", "UNKNOWN", 2)
        self.rotator_status_label = self.create_data_row(status_frame, "Rotator Status:", "UNKNOWN", 3)
        self.roi_label = self.create_data_row(status_frame, "Rotator of Interest:", "UNKNOWN", 4)
        
    def setup_offsets_tab(self):
        """Setup coordinate offsets tab"""
        offset_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(offset_frame, text="Offsets")
        
        # Title
        tk.Label(offset_frame, text="Telescope Coordinate Offsets", 
                bg='#1a1a2e', fg='#00ff41',
                font=('Courier', 14, 'bold')).pack(pady=20)
        
        # Offset amount controls
        controls_frame = tk.Frame(offset_frame, bg='#1a1a2e')
        controls_frame.pack(pady=20)
        
        tk.Label(controls_frame, text="N/S Offset (arcsec):", bg='#1a1a2e', fg='#ffffff',
                font=('Courier', 11)).grid(row=0, column=0, padx=10, pady=5, sticky='e')
        tk.Entry(controls_frame, textvariable=self.offset_ns, width=10,
                font=('Courier', 11)).grid(row=0, column=1, padx=10, pady=5)
                
        tk.Label(controls_frame, text="E/W Offset (arcsec):", bg='#1a1a2e', fg='#ffffff',
                font=('Courier', 11)).grid(row=1, column=0, padx=10, pady=5, sticky='e')
        tk.Entry(controls_frame, textvariable=self.offset_ew, width=10,
                font=('Courier', 11)).grid(row=1, column=1, padx=10, pady=5)
        
        # Directional buttons in cross pattern
        buttons_frame = tk.Frame(offset_frame, bg='#1a1a2e')
        buttons_frame.pack(pady=30)
        
        # North button
        north_btn = tk.Button(buttons_frame, text="NORTH",
                             command=lambda: self.send_offset('north'),
                             bg='#00ff41', fg='#000000',
                             font=('Courier', 12, 'bold'), width=8)
        north_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # West and East buttons
        west_btn = tk.Button(buttons_frame, text="WEST",
                            command=lambda: self.send_offset('west'),
                            bg='#00ff41', fg='#000000',
                            font=('Courier', 12, 'bold'), width=8)
        west_btn.grid(row=1, column=0, padx=5, pady=5)
        
        east_btn = tk.Button(buttons_frame, text="EAST",
                            command=lambda: self.send_offset('east'),
                            bg='#00ff41', fg='#000000',
                            font=('Courier', 12, 'bold'), width=8)
        east_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # South button
        south_btn = tk.Button(buttons_frame, text="SOUTH",
                             command=lambda: self.send_offset('south'),
                             bg='#00ff41', fg='#000000',
                             font=('Courier', 12, 'bold'), width=8)
        south_btn.grid(row=2, column=1, padx=5, pady=5)
        
        # Instructions
        instructions = tk.Label(offset_frame, 
                               text="Click directional buttons to offset telescope\n" +
                                    "Offsets are applied relative to current position\n" +
                                    "Uses TCS commands: ofra, ofdc, offp",
                               bg='#1a1a2e', fg='#ffffff',
                               font=('Courier', 9), justify=tk.CENTER)
        instructions.pack(pady=20)
        
    def setup_log_tab(self):
        """Setup communication log tab"""
        log_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(log_frame, text="Communication Log")
        
        # Create text widget with scrollbar
        log_container = tk.Frame(log_frame, bg='#1a1a2e')
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(log_container, bg='#000000', fg='#00ff41',
                               font=('Courier', 9), state='disabled')
        scrollbar = tk.Scrollbar(log_container, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_data_row(self, parent, label_text, value_text, row):
        """Create a data display row"""
        tk.Label(parent, text=label_text, bg='#1a1a2e', fg='#ffffff',
                font=('Courier', 11, 'bold')).grid(row=row, column=0, sticky='w', padx=10, pady=8)
        
        value_label = tk.Label(parent, text=value_text, bg='#000000', fg='#00ff41',
                              font=('Courier', 12), width=20, anchor='center', relief='sunken')
        value_label.grid(row=row, column=1, padx=10, pady=8, sticky='ew')
        
        parent.grid_columnconfigure(1, weight=1)
        return value_label
        
    def setup_control_frame(self, parent):
        """Setup control buttons"""
        control_frame = tk.Frame(parent, bg='#0c1445')
        control_frame.pack(fill=tk.X, pady=10)
        
        self.update_btn = tk.Button(control_frame, text="Start Auto Update",
                                   command=self.toggle_updates,
                                   bg='#00ff41', fg='#000000',
                                   font=('Courier', 10, 'bold'))
        self.update_btn.pack(side=tk.LEFT, padx=5)
        
        refresh_btn = tk.Button(control_frame, text="Refresh Now",
                               command=self.refresh_data,
                               bg='#00ff41', fg='#000000',
                               font=('Courier', 10, 'bold'))
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
    def setup_status_bar(self, parent):
        """Setup status bar"""
        status_frame = tk.Frame(parent, bg='#0c1445')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.connection_status = tk.Label(status_frame, text="● Disconnected",
                                         bg='#0c1445', fg='#ff4444',
                                         font=('Courier', 10))
        self.connection_status.pack(side=tk.LEFT)
        
        self.last_update = tk.Label(status_frame, text="Last Update: Never",
                                   bg='#0c1445', fg='#ffffff',
                                   font=('Courier', 10))
        self.last_update.pack(side=tk.RIGHT)
        
    def toggle_connection(self):
        """Toggle TCS connection"""
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()
            
    def connect(self):
        """Connect to TCS"""
        try:
            ip = self.tcs_ip.get().strip()
            port = int(self.tcs_port.get().strip())
            
            if not ip or not port:
                messagebox.showerror("Error", "Please enter valid IP and port")
                return
                
            self.handler = TCSHandler(ip, port)
            if self.handler.connect():
                self.worker = TCSWorker(self.handler)
                self.is_connected = True
                self.worker.start()
                self.connect_btn.config(text="Disconnect")
                self.connection_status.config(text="● Connected", fg='#00ff41')
                self.log_message(f"Connected to TCS at {ip}:{port}")
            else:
                messagebox.showerror("Connection Error", f"Failed to connect to TCS at {ip}:{port}")
        except ValueError:
            messagebox.showerror("Error", "Port must be a valid number")
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")
            
    def disconnect(self):
        """Disconnect from TCS"""
        if self.auto_update:
            self.toggle_updates()
            
        if self.worker:
            self.worker.stop()
            
        self.is_connected = False
        self.connect_btn.config(text="Connect")
        self.connection_status.config(text="● Disconnected", fg='#ff4444')
        self.log_message("Disconnected from TCS")
        
    def toggle_updates(self):
        """Toggle automatic updates"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to TCS first")
            return
            
        self.auto_update = not self.auto_update
        self.update_btn.config(text="Stop Auto Update" if self.auto_update else "Start Auto Update")
        
    def refresh_data(self):
        """Manual refresh"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to TCS first")
            return
            
        # Send manual requests
        for cmd in ['datetime', 'telpos', 'teldata']:
            self.worker.send_command(cmd)
            
    def send_offset(self, direction):
        """Send coordinate offset command"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to TCS first")
            return
            
        try:
            ns_offset = float(self.offset_ns.get())
            ew_offset = float(self.offset_ew.get())
            
            # Calculate RA and DEC offsets based on direction
            # North = +DEC, South = -DEC, East = -RA, West = +RA
            if direction == 'north':
                ra_offset = 0.0
                dec_offset = ns_offset
            elif direction == 'south':
                ra_offset = 0.0
                dec_offset = -ns_offset
            elif direction == 'east':
                ra_offset = -ew_offset
                dec_offset = 0.0
            elif direction == 'west':
                ra_offset = ew_offset
                dec_offset = 0.0
            else:
                return
                
            # Send TCS offset commands (Table 13 in documentation)
            self.worker.send_command(f"ofra {ra_offset}")
            self.worker.send_command(f"ofdc {dec_offset}")
            # wait a second
            time.sleep(1)
            self.worker.send_command("offp ")  # Execute the offset
            
            self.log_message(f"Offset command sent: {direction.upper()} "
                           f"(RA: {ra_offset:+.1f}\", DEC: {dec_offset:+.1f}\")")
            
        except ValueError:
            messagebox.showerror("Error", "Offset values must be valid numbers")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send offset: {str(e)}")
            
    def log_message(self, message):
        """Add message to log"""
        if hasattr(self, 'log_text'):
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, f"{_timestamp()}: {message}\n")
            self.log_text.config(state='disabled')
            self.log_text.see(tk.END)
        
    def update_display(self):
        """Update GUI display with new data"""
        # Process messages from worker
        if self.worker:
            while True:
                try:
                    msg_type, data = self.worker.gui_queue.get_nowait()
                    
                    if msg_type == 'data_update':
                        self.update_telescope_data()
                        self.last_update.config(text=f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
                        print(f"GUI updated with {data} data")  # Debug print
                        
                    elif msg_type == 'log':
                        if hasattr(self, 'log_text'):
                            self.log_text.config(state='normal')
                            self.log_text.insert(tk.END, f"{data}\n")
                            self.log_text.config(state='disabled')
                            self.log_text.see(tk.END)
                            
                except Empty:
                    break
                except Exception as e:
                    print(f"Error in update_display: {e}")
                
        # Schedule next update
        self.root.after(100, self.update_display)
        
    def update_telescope_data(self):
        """Update telescope data display"""
        if not self.worker:
            return
            
        data = self.worker.current_data
        
        # Update coordinates tab
        self.ra_label.config(text=data['ra'])
        self.dec_label.config(text=data['dec'])
        self.ha_label.config(text=data['hour_angle'])
        self.sidereal_label.config(text=data['sidereal_time'])
        self.date_label.config(text=data['date'])
        self.time_label.config(text=data['time'])
        
        # Update position tab
        self.elevation_label.config(text=data['elevation'])
        self.azimuth_label.config(text=data['azimuth'])
        self.airmass_label.config(text=data['airmass'])
        self.rotator_label.config(text=data['rotator_angle'])
        self.parallactic_label.config(text=data['parallactic_angle'])
        
        # Update status tab
        self.dome_status_label.config(text=data['dome_status'])
        self.guider_status_label.config(text=data['guider_status'])
        self.mount_status_label.config(text=data['mount_status'])
        self.rotator_status_label.config(text=data['rotator_status'])
        self.roi_label.config(text=data['roi'])
        
    def on_closing(self):
        """Handle window closing"""
        if self.is_connected:
            self.disconnect()
        self.root.destroy()
        
    def run(self):
        """Run the GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()


def main():
    """Main application entry point"""
    app = TCSGUI()
    app.run()


if __name__ == "__main__":
    main()
