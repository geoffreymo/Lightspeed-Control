#!/usr/bin/env python3
"""
Las Campanas Observatory - Telescope Control System GUI
Enhanced with offset debugging capabilities
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
import socket
import threading
from queue import Queue, Empty
from datetime import datetime
import time
import re

def _timestamp():
    return datetime.now().isoformat(timespec='seconds')


class TCSHandler:
    """Handles socket read/write and queues for GUI"""
    def __init__(self, ip_address, port):
        self._socket = None
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
            
    def send_single_command(self, command, timeout=2.0):
        """Send a single command and get response using new connection"""
        try:
            # Create new connection for this command
            sock = socket.create_connection((self.ip_address, self.port), timeout=5)
            sock.settimeout(timeout)
            
            # Send command
            sock.sendall(f"{command}\n".encode('ascii'))
            
            # Read response
            response = b''
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    data = sock.recv(1024)
                    if not data:
                        break
                    response += data
                    if b'\n' in response:
                        break
                except socket.timeout:
                    break
            
            sock.close()
            
            if response:
                return response.decode('ascii', errors='ignore').strip()
            return None
            
        except Exception as e:
            print(f"Command '{command}' error: {e}")
            return None


class TCSGUI:
    """Main TCS GUI application with debugging features"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Las Campanas Observatory - TCS Monitor & Debug')
        self.root.geometry('1400x900')
        self.root.configure(bg='#0c1445')
        
        # TCS connection settings
        self.tcs_ip = tk.StringVar(value="200.28.147.59")
        self.tcs_port = tk.StringVar(value="5800")
        
        # Initialize TCS handler
        self.handler = TCSHandler("200.28.147.59", 5800)
        self.is_connected = False
        
        # Offset parameters
        self.offset_ns = tk.StringVar(value="5.0")
        self.offset_ew = tk.StringVar(value="5.0")
        
        # Store position before offset for comparison
        self.pre_offset_position = None
        
        self.setup_gui()
        
        # Handle window closing
        self.root.protocol('WM_DELETE_WINDOW', self.on_closing)
        
    def setup_gui(self):
        """Setup GUI components"""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container with two columns
        main_frame = tk.Frame(self.root, bg='#0c1445')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - original GUI
        left_frame = tk.Frame(main_frame, bg='#0c1445')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right side - debug panel
        right_frame = tk.Frame(main_frame, bg='#0c1445')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Title
        title_label = tk.Label(left_frame, 
                              text="Las Campanas Observatory\nTelescope Control System Monitor",
                              bg='#0c1445', fg='#00ff41',
                              font=('Courier', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Connection frame
        self.setup_connection_frame(left_frame)
        
        # Enhanced offset controls with debugging
        self.setup_offset_debug_frame(left_frame)
        
        # Debug panel on right side
        self.setup_debug_panel(right_frame)
        
        # Status bar
        self.setup_status_bar(left_frame)
        
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
        
    def setup_offset_debug_frame(self, parent):
        """Setup enhanced offset controls with debugging"""
        offset_frame = tk.LabelFrame(parent, text="Offset Testing & Debug", 
                                    bg='#1a1a2e', fg='#00ff41',
                                    font=('Courier', 12, 'bold'))
        offset_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Current position display
        pos_frame = tk.Frame(offset_frame, bg='#1a1a2e')
        pos_frame.pack(pady=10)
        
        tk.Label(pos_frame, text="Current Position:", bg='#1a1a2e', fg='#00ff41',
                font=('Courier', 11, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)
        
        tk.Label(pos_frame, text="RA:", bg='#1a1a2e', fg='#ffffff',
                font=('Courier', 10)).grid(row=1, column=0, sticky='e', padx=5)
        self.ra_display = tk.Label(pos_frame, text="--:--:--", bg='#000000', fg='#00ff41',
                                   font=('Courier', 10), width=15)
        self.ra_display.grid(row=1, column=1, padx=5)
        
        tk.Label(pos_frame, text="Dec:", bg='#1a1a2e', fg='#ffffff',
                font=('Courier', 10)).grid(row=2, column=0, sticky='e', padx=5)
        self.dec_display = tk.Label(pos_frame, text="--:--:--", bg='#000000', fg='#00ff41',
                                    font=('Courier', 10), width=15)
        self.dec_display.grid(row=2, column=1, padx=5)
        
        # Offset controls
        controls_frame = tk.Frame(offset_frame, bg='#1a1a2e')
        controls_frame.pack(pady=10)
        
        tk.Label(controls_frame, text="N/S Offset (arcsec):", bg='#1a1a2e', fg='#ffffff',
                font=('Courier', 10)).grid(row=0, column=0, padx=10, pady=5, sticky='e')
        tk.Entry(controls_frame, textvariable=self.offset_ns, width=10,
                font=('Courier', 10)).grid(row=0, column=1, padx=10, pady=5)
                
        tk.Label(controls_frame, text="E/W Offset (arcsec):", bg='#1a1a2e', fg='#ffffff',
                font=('Courier', 10)).grid(row=1, column=0, padx=10, pady=5, sticky='e')
        tk.Entry(controls_frame, textvariable=self.offset_ew, width=10,
                font=('Courier', 10)).grid(row=1, column=1, padx=10, pady=5)
        
        # Test buttons
        test_frame = tk.Frame(offset_frame, bg='#1a1a2e')
        test_frame.pack(pady=10)
        
        tk.Button(test_frame, text="Test Single Offset",
                 command=self.test_single_offset,
                 bg='#00ff41', fg='#000000',
                 font=('Courier', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        
        tk.Button(test_frame, text="Test Offset Sequence",
                 command=self.test_offset_sequence,
                 bg='#00ff41', fg='#000000',
                 font=('Courier', 10, 'bold')).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Button(test_frame, text="Test All Commands",
                 command=self.test_all_commands,
                 bg='#ff9900', fg='#000000',
                 font=('Courier', 10, 'bold')).grid(row=1, column=0, padx=5, pady=5)
        
        tk.Button(test_frame, text="Update Position",
                 command=self.update_position,
                 bg='#00aaff', fg='#000000',
                 font=('Courier', 10, 'bold')).grid(row=1, column=1, padx=5, pady=5)
        
    def setup_debug_panel(self, parent):
        """Setup debug output panel"""
        debug_frame = tk.LabelFrame(parent, text="Debug Output", 
                                   bg='#1a1a2e', fg='#00ff41',
                                   font=('Courier', 12, 'bold'))
        debug_frame.pack(fill=tk.BOTH, expand=True)
        
        # Debug text area
        self.debug_text = scrolledtext.ScrolledText(debug_frame, 
                                                    bg='#000000', fg='#00ff41',
                                                    font=('Courier', 9),
                                                    height=30, width=60)
        self.debug_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Clear button
        tk.Button(debug_frame, text="Clear Debug",
                 command=lambda: self.debug_text.delete(1.0, tk.END),
                 bg='#ff4444', fg='#ffffff',
                 font=('Courier', 10)).pack(pady=5)
        
    def setup_status_bar(self, parent):
        """Setup status bar"""
        status_frame = tk.Frame(parent, bg='#0c1445')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.connection_status = tk.Label(status_frame, text="● Disconnected",
                                         bg='#0c1445', fg='#ff4444',
                                         font=('Courier', 10))
        self.connection_status.pack(side=tk.LEFT)
        
        self.last_update = tk.Label(status_frame, text="",
                                   bg='#0c1445', fg='#ffffff',
                                   font=('Courier', 10))
        self.last_update.pack(side=tk.RIGHT)
        
    def log_debug(self, message, color='#00ff41'):
        """Add message to debug output"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.debug_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.debug_text.see(tk.END)
        
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
            
            self.handler = TCSHandler(ip, port)
            if self.handler.connect():
                self.is_connected = True
                self.connect_btn.config(text="Disconnect")
                self.connection_status.config(text="● Connected", fg='#00ff41')
                self.log_debug(f"Connected to TCS at {ip}:{port}")
                self.update_position()
            else:
                messagebox.showerror("Connection Error", f"Failed to connect to TCS at {ip}:{port}")
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")
            
    def disconnect(self):
        """Disconnect from TCS"""
        if self.handler:
            self.handler.disconnect()
        self.is_connected = False
        self.connect_btn.config(text="Connect")
        self.connection_status.config(text="● Disconnected", fg='#ff4444')
        self.log_debug("Disconnected from TCS")
        
    def update_position(self):
        """Update current telescope position"""
        if not self.handler:
            return
            
        response = self.handler.send_single_command("telpos")
        if response:
            parts = response.split()
            if len(parts) >= 2:
                self.ra_display.config(text=parts[0])
                self.dec_display.config(text=parts[1])
                self.last_update.config(text=f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            self.log_debug(f"Position: {response}")
        else:
            self.log_debug("No response from telpos command", '#ff4444')
            
    def test_single_offset(self):
        """Test a single offset with detailed debugging"""
        if not self.handler:
            messagebox.showwarning("Warning", "Please connect to TCS first")
            return
            
        try:
            ns_offset = float(self.offset_ns.get())
            ew_offset = float(self.offset_ew.get())
            
            self.log_debug("="*50)
            self.log_debug("STARTING SINGLE OFFSET TEST")
            self.log_debug(f"Target offsets: NS={ns_offset}\" EW={ew_offset}\"")
            
            # Get initial position
            self.log_debug("Getting initial position...")
            initial_pos = self.handler.send_single_command("telpos")
            self.log_debug(f"Initial position: {initial_pos}")
            
            # Send RA offset
            if ew_offset != 0:
                self.log_debug(f"Sending: ofra {ew_offset}")
                response = self.handler.send_single_command(f"ofra {ew_offset}")
                self.log_debug(f"Response: '{response}'")
                time.sleep(0.5)
            
            # Send Dec offset  
            if ns_offset != 0:
                self.log_debug(f"Sending: ofdc {ns_offset}")
                response = self.handler.send_single_command(f"ofdc {ns_offset}")
                self.log_debug(f"Response: '{response}'")
                time.sleep(0.5)
            
            # Execute offset
            self.log_debug("Sending: offp")
            response = self.handler.send_single_command("offp")
            self.log_debug(f"Response: '{response}'")
            
            # Wait for telescope to move
            self.log_debug("Waiting 3 seconds for telescope to move...")
            time.sleep(3)
            
            # Get final position
            self.log_debug("Getting final position...")
            final_pos = self.handler.send_single_command("telpos")
            self.log_debug(f"Final position: {final_pos}")
            
            # Compare positions
            if initial_pos and final_pos:
                if initial_pos != final_pos:
                    self.log_debug("SUCCESS: Position changed!", '#00ff00')
                else:
                    self.log_debug("WARNING: Position unchanged", '#ffaa00')
            
            self.log_debug("TEST COMPLETE")
            self.log_debug("="*50)
            
            # Update display
            self.update_position()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid offset values")
        except Exception as e:
            self.log_debug(f"Error: {e}", '#ff4444')
            
    def test_offset_sequence(self):
        """Test a sequence of offsets"""
        if not self.handler:
            messagebox.showwarning("Warning", "Please connect to TCS first")
            return
            
        self.log_debug("="*50)
        self.log_debug("STARTING OFFSET SEQUENCE TEST")
        
        # Test sequence
        offsets = [
            (5, 0, "5\" East"),
            (0, 5, "5\" North"),
            (-5, 0, "5\" West"),
            (0, -5, "5\" South"),
            (0, 0, "Return to origin")
        ]
        
        for ew, ns, description in offsets:
            self.log_debug(f"\nTesting: {description}")
            
            # Get position before
            pos_before = self.handler.send_single_command("telpos")
            self.log_debug(f"Before: {pos_before}")
            
            # Send offsets
            if ew != 0:
                cmd = f"ofra {ew}"
                self.log_debug(f"Send: {cmd}")
                resp = self.handler.send_single_command(cmd)
                self.log_debug(f"Response: '{resp}'")
                time.sleep(0.2)
                
            if ns != 0:
                cmd = f"ofdc {ns}"
                self.log_debug(f"Send: {cmd}")
                resp = self.handler.send_single_command(cmd)
                self.log_debug(f"Response: '{resp}'")
                time.sleep(0.2)
            
            # Execute
            self.log_debug("Send: offp")
            resp = self.handler.send_single_command("offp")
            self.log_debug(f"Response: '{resp}'")
            
            # Wait and check
            time.sleep(2)
            pos_after = self.handler.send_single_command("telpos")
            self.log_debug(f"After: {pos_after}")
            
            if pos_before != pos_after:
                self.log_debug("Position changed ✓", '#00ff00')
            else:
                self.log_debug("Position unchanged ✗", '#ffaa00')
                
        self.log_debug("\nSEQUENCE TEST COMPLETE")
        self.log_debug("="*50)
        self.update_position()
        
    def test_all_commands(self):
        """Test various TCS commands to understand protocol"""
        if not self.handler:
            messagebox.showwarning("Warning", "Please connect to TCS first")
            return
            
        self.log_debug("="*50)
        self.log_debug("TESTING ALL TCS COMMANDS")
        
        test_commands = [
            # Basic queries
            ("datetime", "Date/time query"),
            ("telpos", "Position query"),
            ("teldata", "Telescope data query"),
            
            # Offset commands with variations
            ("ofra 5", "RA offset 5\""),
            ("ofra 5.0", "RA offset 5.0\""),
            ("ofra 5.00", "RA offset 5.00\""),
            ("ofdc 5", "Dec offset 5\""),
            ("offp", "Execute offset"),
            
            # Try to find offset status
            ("ofst", "Offset status?"),
            ("offset", "Offset info?"),
            
            # Try different offset formats
            ("offset 5 5", "Compound offset?"),
            ("offsets 5 5", "Alternative compound?"),
            
            # Help commands
            ("help", "Help"),
            ("?", "Help (?)"),
            ("commands", "Command list?"),
        ]
        
        for cmd, description in test_commands:
            self.log_debug(f"\n{description}:")
            self.log_debug(f"  Send: '{cmd}'")
            response = self.handler.send_single_command(cmd, timeout=1.0)
            if response:
                # Truncate long responses
                if len(response) > 100:
                    response = response[:100] + "..."
                self.log_debug(f"  Response: '{response}'", '#00ff41')
            else:
                self.log_debug(f"  Response: None/Timeout", '#ffaa00')
            time.sleep(0.5)
            
        self.log_debug("\nALL COMMANDS TEST COMPLETE")
        self.log_debug("="*50)
        
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