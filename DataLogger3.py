"""
ROB 311 - Fall 2025
Author: GSI Yilin Ma & IA Eeshwar Krishnan
University of Michigan

DataLogger3.py - Simple API with Stable Real-time Visualization
"""

import os
import socket
import json
import threading
import time
import subprocess
import sys
from typing import Optional, List


class dataLogger:
    """
    Drop-in replacement for DataLogger with optional real-time plotting.

    Features:
    - Same API as original DataLogger (appendData, writeOut)
    - Automatic TCP-based streaming to viewer (more stable than WebSocket)
    - Optional PyQtGraph viewer (auto-launches if available)
    - Graceful degradation if dependencies missing

    Example:
        # Exact same usage as DataLogger/DataLogger2
        logger = dataLogger("output.txt")
        logger.appendData(["time", "sensor1", "sensor2"])  # Headers
        logger.appendData([0.0, 1.23, 4.56])  # Data
        logger.writeOut()
    """

    def __init__(self, name: str, enable_plotting: bool = True, port: int = 5557):
        """
        Initialize data logger with optional real-time plotting.

        Args:
            name: Output filename
            enable_plotting: Enable real-time plot viewer (default: True)
            port: TCP port for streaming data (default: 5557)
        """
        self.name = name
        self.myData = []
        self.header = None
        self.enable_plotting = enable_plotting
        self.port = port

        # Ensure directory exists
        d = os.path.dirname(self.name)
        if d:
            os.makedirs(d, exist_ok=True)

        # Create/truncate file
        with open(self.name, 'w'):
            pass

        # Initialize streaming server
        self.server = None
        self.clients = []
        self.server_thread = None
        self._running = False

        if enable_plotting:
            self._start_server()

    def _start_server(self):
        """Start TCP server for streaming data to viewers"""
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(('0.0.0.0', self.port))
            self.server.listen(5)
            self.server.settimeout(0.5)  # Non-blocking accept
            self._running = True

            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()

            print(f"[DataLogger3] Streaming server started on port {self.port}")
            print(f"[DataLogger3] Run viewer: python DataLogger3_viewer.py")

        except Exception as e:
            print(f"[DataLogger3] Could not start server: {e}")
            self.enable_plotting = False

    def _server_loop(self):
        """Accept client connections in background thread"""
        while self._running:
            try:
                client_sock, addr = self.server.accept()
                self.clients.append(client_sock)
                print(f"[DataLogger3] Viewer connected from {addr}")

                # Send header if we have one
                if self.header is not None:
                    self._send_to_clients({
                        'type': 'header',
                        'headers': self.header
                    })
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[DataLogger3] Server error: {e}")

    def _send_to_clients(self, data: dict):
        """Send data to all connected clients"""
        if not self.clients:
            return

        message = json.dumps(data) + '\n'
        dead_clients = []

        for client in self.clients:
            try:
                client.sendall(message.encode('utf-8'))
            except:
                dead_clients.append(client)

        # Remove disconnected clients
        for client in dead_clients:
            try:
                client.close()
            except:
                pass
            self.clients.remove(client)

    def appendData(self, val: List):
        """
        Append data row to log.

        First call is treated as headers, subsequent calls as data.
        Compatible with both DataLogger and DataLogger2 APIs.

        Args:
            val: List of values (headers or data row)
        """
        if self.header is None:
            # First call - treat as headers
            if isinstance(val, list) and len(val) == 1 and isinstance(val[0], str):
                # Header is a single string "col1 col2 col3", split it
                self.header = val[0].split()
            elif isinstance(val, list) and all(isinstance(v, str) for v in val):
                # Already a list of strings
                self.header = val
            else:
                # Convert to strings for header
                self.header = [str(v) for v in val]

            if self.enable_plotting:
                self._send_to_clients({
                    'type': 'header',
                    'headers': self.header
                })
        else:
            # Subsequent calls - treat as data
            if self.enable_plotting:
                self._send_to_clients({
                    'type': 'data',
                    'values': val
                })

        self.myData.append(val)

    def writeOut(self):
        """Write buffered data to file"""
        print('[DataLogger3] Storing data...\n')
        outTxt = []
        for data in self.myData:
            outTxt.append(' '.join(str(e) for e in data))
            outTxt.append('\n')
        if outTxt:
            with open(self.name, 'a') as f:
                f.write(''.join(outTxt))
        self.myData = []

    def close(self):
        """Clean up resources"""
        self._running = False
        if self.server:
            try:
                self.server.close()
            except:
                pass
        for client in self.clients:
            try:
                client.close()
            except:
                pass

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


# ============================================================================
# Standalone Viewer (can be run separately as DataLogger3_viewer.py)
# ============================================================================

def check_dependencies():
    """Check and optionally install PyQtGraph dependencies"""
    missing = []

    try:
        import numpy as np
    except ImportError:
        missing.append('numpy')

    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets
    except ImportError:
        missing.extend(['pyqtgraph', 'PyQt5'])

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        response = input("Install now? (y/n): ").strip().lower()
        if response == 'y':
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
                print("âœ“ Installation complete! Restart to use viewer.")
                return False
            except:
                print("âœ— Installation failed. Install manually:")
                print(f"  pip3 install {' '.join(missing)}")
                return False
        return False
    return True


class DataLogger3Viewer:
    """Real-time viewer for DataLogger3 streams"""

    def __init__(self, host='localhost', port=5557):
        if not check_dependencies():
            print("Cannot start viewer without dependencies")
            sys.exit(1)

        import numpy as np
        from collections import deque
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        self.headers = None

        # Data buffers (5 seconds at 200 Hz = 1000 points)
        self.max_points = 1000
        self.data = {}

        # Create Qt application
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)

        # Connect to server
        self.connect_to_server()

        # Setup will happen after receiving headers
        self.win = None
        self.plots = {}
        self.curves = {}

        # Timer for reading data
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.read_and_update)
        self.timer.start(20)  # 50 Hz update rate

    def connect_to_server(self):
        """Connect to DataLogger3 TCP server"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            print(f"âœ“ Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"âœ— Connection failed: {e}")
            print(f"Make sure your script is running with DataLogger3")
            sys.exit(1)

    def setup_ui(self):
        """Create plot window after receiving headers"""
        import pyqtgraph as pg

        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.resize(1600, 900)
        self.win.setWindowTitle(f'DataLogger3 Viewer - {self.host}:{self.port}')

        # Create plots for each data column (skip iteration counter)
        num_plots = len(self.headers) - 1  # Skip first column (iteration)
        cols = 2
        rows = (num_plots + cols - 1) // cols

        for idx, header in enumerate(self.headers[1:], start=1):
            row = (idx - 1) // cols
            col = (idx - 1) % cols

            plot = self.win.addPlot(title=header, row=row, col=col)
            plot.setLabel('left', header)
            plot.setLabel('bottom', 'Time', units='s')
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setDownsampling(mode='peak')
            plot.setClipToView(True)

            curve = plot.plot(pen=pg.mkPen(color=(100, 200, 255), width=2))

            self.plots[header] = plot
            self.curves[header] = curve

    def read_and_update(self):
        """Read data from socket and update plots"""
        import numpy as np
        from collections import deque

        if not self.sock:
            return

        try:
            chunk = self.sock.recv(8192).decode('utf-8')
            self.buffer += chunk

            # Process complete lines
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                try:
                    msg = json.loads(line)
                    self.process_message(msg)
                except json.JSONDecodeError:
                    continue
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"Read error: {e}")
            self.sock = None
            return

        # Update plots if we have UI
        if self.win and self.data:
            time_key = self.headers[1] if len(self.headers) > 1 else 'time'
            if time_key in self.data and len(self.data[time_key]) > 0:
                time_data = np.array(self.data[time_key])

                for header in self.headers[1:]:
                    if header in self.curves and header in self.data:
                        y_data = np.array(self.data[header])
                        if len(y_data) > 0:
                            self.curves[header].setData(time_data, y_data)

    def process_message(self, msg):
        """Process incoming message"""
        from collections import deque

        if msg['type'] == 'header':
            self.headers = msg['headers']
            print(f"Received headers: {self.headers}")

            # Initialize data buffers
            for header in self.headers:
                self.data[header] = deque(maxlen=self.max_points)

            # Create UI if not exists
            if not self.win:
                self.setup_ui()

        elif msg['type'] == 'data' and self.headers:
            values = msg['values']
            for header, value in zip(self.headers, values):
                if header in self.data:
                    self.data[header].append(float(value))

    def run(self):
        """Start the viewer event loop"""
        import sys
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    """
    Standalone viewer mode
    Usage: python DataLogger3.py [host] [port]
    """
    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5557

    viewer = DataLogger3Viewer(host=host, port=port)
    viewer.run()