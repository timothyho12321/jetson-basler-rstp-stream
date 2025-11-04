import cv2
import numpy as np
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from frame_manager import FrameManager

# Global shared generators - SINGLE instance for ALL clients
class GlobalStream1Generator:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self.start_time = time.time()
        self.global_frame_count = 0
        self.lock = threading.Lock()
    
    def get_frame(self):
        """Get frame with GLOBAL frame count - same for ALL clients"""
        with self.lock:
            frame_count = self.global_frame_count
            self.global_frame_count += 1
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue background for stream1
        
        # Calculate time from SHARED start time
        elapsed = time.time() - self.start_time
        current_sgt = self._get_sgt_time()
        
        # IDENTICAL overlay for ALL clients
        cv2.putText(frame, "STREAM 1", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {current_sgt}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        # # IDENTICAL animation for ALL clients (based on GLOBAL frame count)
        # pulse_size = 10 + int(5 * np.sin(frame_count * 0.1))
        # cv2.circle(frame, (600, 40), pulse_size, (255, 255, 255), -1)
        
        #angle = frame_count * 5 % 360
        #rad = np.radians(angle)
        #end_x = int(320 + 50 * np.cos(rad))
        #end_y = int(240 + 50 * np.sin(rad))
        #cv2.line(frame, (320, 240), (end_x, end_y), (255, 255, 255), 2)
        
        return frame
    
    def _get_sgt_time(self):
        utc_time = time.gmtime()
        sgt_timestamp = time.mktime(utc_time) + (8 * 3600)
        return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(sgt_timestamp))

class GlobalStream2Generator:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self.start_time = time.time()
        self.global_frame_count = 0
        self.lock = threading.Lock()
    
    def get_frame(self):
        """Get frame with GLOBAL frame count - same for ALL clients"""
        with self.lock:
            frame_count = self.global_frame_count
            self.global_frame_count += 1
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Red background for stream2
        
        elapsed = time.time() - self.start_time
        current_sgt = self._get_sgt_time()
        
        # IDENTICAL overlay for ALL clients
        cv2.putText(frame, "STREAM 2", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {current_sgt}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
  
        
        # IDENTICAL animation for ALL clients (based on GLOBAL frame count)
        #pulse_size = 10 + int(5 * np.sin(frame_count * 0.1))
        #cv2.circle(frame, (600, 40), pulse_size, (255, 255, 255), -1)
        
        #angle = frame_count * 5 % 360
        #rad = np.radians(angle)
        #end_x = int(320 + 50 * np.cos(rad))
        #end_y = int(240 + 50 * np.sin(rad))
        #cv2.line(frame, (320, 240), (end_x, end_y), (255, 255, 255), 2)
        
        return frame
    
    def _get_sgt_time(self):
        utc_time = time.gmtime()
        sgt_timestamp = time.mktime(utc_time) + (8 * 3600)
        return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(sgt_timestamp))

class DualStreamHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, frame_manager=None, **kwargs):
        # Use provided frame_manager or create new one
        if frame_manager is not None:
            self.frame_manager = frame_manager
        else:
            self.frame_manager = FrameManager()
        
        # Initialize generators
        self.stream1_generator = GlobalStream1Generator()
        self.stream2_generator = GlobalStream2Generator()
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        pass
    
    def get_sgt_time(self):
        utc_time = time.gmtime()
        sgt_timestamp = time.mktime(utc_time) + (8 * 3600)
        return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(sgt_timestamp))
    
    def do_GET(self):
        if self.path == '/stream1':
            self.handle_stream1()
        elif self.path == '/stream2':
            self.handle_stream2()
        elif self.path == '/both':
            self.handle_both_streams()
        elif self.path == '/':
            self.handle_root()
        elif self.path == '/status':
            self.handle_status()
        else:
            self.send_error(404)
    
    def handle_stream1(self):
        """Handle stream1 - ALL clients see IDENTICAL frames"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()
        
        client_frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Get frame from external app via frame_manager
                external_frame, frame_time, external_count = self.frame_manager.get_frame1()
                
                if external_frame is not None:
                    # Use external frame from YOLO detection
                    frame = external_frame
                else:
                    # Fallback to generated frame
                    frame = self.stream1_generator.get_frame()
                
                # Encode to JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
                success, jpeg_data = cv2.imencode('.jpg', frame, encode_params)
                
                if success:
                    # Send frame to client
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg_data)))
                    self.end_headers()
                    self.wfile.write(jpeg_data.tobytes())
                    self.wfile.write(b'\r\n')
                
                client_frame_count += 1
                
                # Log performance
                if client_frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = client_frame_count / elapsed
                    print(f"?? Stream1 Client: {client_frame_count} frames, {fps:.1f} FPS")
                
                time.sleep(0.033)  # ~30 FPS
                
        except (BrokenPipeError, ConnectionResetError):
            print("?? Stream1 client disconnected")
    
    def handle_stream2(self):
        """Handle stream2 - ALL clients see IDENTICAL frames"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()
        
        client_frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Get frame from external app via frame_manager
                external_frame, frame_time, external_count = self.frame_manager.get_frame2()
                
                if external_frame is not None:
                    # Use external frame
                    frame = external_frame
                else:
                    # Fallback to generated frame
                    frame = self.stream2_generator.get_frame()
                
                # Encode to JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
                success, jpeg_data = cv2.imencode('.jpg', frame, encode_params)
                
                if success:
                    # Send frame to client
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg_data)))
                    self.end_headers()
                    self.wfile.write(jpeg_data.tobytes())
                    self.wfile.write(b'\r\n')
                
                client_frame_count += 1
                
                # Log performance
                if client_frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = client_frame_count / elapsed
                    print(f"?? Stream2 Client: {client_frame_count} frames, {fps:.1f} FPS")
                
                time.sleep(0.033)  # ~30 FPS
                
        except (BrokenPipeError, ConnectionResetError):
            print("?? Stream2 client disconnected")

    def handle_both_streams(self):
        """Handle combined stream showing both streams side by side"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()
        
        client_frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Get frames from external app or generators
                frame1, time1, count1 = self.frame_manager.get_frame1()
                if frame1 is None:
                    frame1 = self.stream1_generator.get_frame()
                
                frame2, time2, count2 = self.frame_manager.get_frame2()
                if frame2 is None:
                    frame2 = self.stream2_generator.get_frame()
                
                if frame1 is not None and frame2 is not None:
                    # Create combined frame (side by side)
                    combined = np.hstack([frame1, frame2])
                    
                    # Add overlay
                    self.add_combined_overlay(combined, client_frame_count, count1, count2)
                    
                    # Encode and send
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
                    success, jpeg_data = cv2.imencode('.jpg', combined, encode_params)
                    
                    if success:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg_data)))
                        self.end_headers()
                        self.wfile.write(jpeg_data.tobytes())
                        self.wfile.write(b'\r\n')
                
                client_frame_count += 1
                
                if client_frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    fps = client_frame_count / elapsed
                    print(f" Combined Client: {client_frame_count} frames, {fps:.1f} FPS")
                
                time.sleep(0.06)
                
        except (BrokenPipeError, ConnectionResetError):
            print(" Combined stream client disconnected")
    
    def add_combined_overlay(self, combined_frame, client_frame_count, count1, count2):
        """Add overlay for combined stream"""
        current_sgt = self.get_sgt_time()
        
        cv2.putText(combined_frame, "DUAL STREAM ", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_frame, f"Time: {current_sgt}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined_frame, f"Client Frames: {client_frame_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add divider line
        height = combined_frame.shape[0]
        cv2.line(combined_frame, (640, 0), (640, height), (255, 255, 255), 2)
        cv2.putText(combined_frame, "STREAM 1", (250, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.putText(combined_frame, "STREAM 2", (900, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
    
    def handle_root(self):
        """Serve HTML page"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        current_sgt = self.get_sgt_time()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flotech AI Camera test</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #0a0a0a; color: #fff; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: #111; padding: 20px; border-radius: 10px; }}
                .header {{ text-align: center; margin-bottom: 20px; background: #222; padding: 15px; border-radius: 10px; }}
                .streams {{ display: flex; gap: 20px; flex-wrap: wrap; }}
                .stream-box {{ flex: 1; min-width: 400px; background: #1a1a1a; padding: 15px; border-radius: 10px; }}
                .stream-title {{ text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .stream1-title {{ color: #00aaff; }}
                .stream2-title {{ color: #ffaa00; }}
                img {{ max-width: 100%; border: 2px solid; border-radius: 5px; }}
                .stream1-img {{ border-color: #00aaff; }}
                .stream2-img {{ border-color: #ffaa00; }}
                .note {{ background: #333; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 12px; color: #0f0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1></h1>
                    <div style="color: #0f0;">{current_sgt}</div>
                    <div style="color: #0ff; font-size: 14px; margin-top: 10px;">
                        Flotech AI Camera Test
                    </div>
                </div>
                
                <div class="streams">
                    <div class="stream-box">
                        <div class="stream-title stream1-title">Stream 1</div>
                        <img src="/stream1" alt="camera1" class="stream1-img">
                    </div>
                    
                    <div class="stream-box">
                        <div class="stream-title stream2-title">Stream 2</div>
                        <img src="/stream2" alt="Test Stream" class="stream2-img">
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 20px; color: #888;">
                    <a href="/both" style="color: #0ff; font-size: 16px; margin: 0 10px;">?? View Combined Stream</a>
                    <a href="/status" style="color: #0ff; font-size: 16px; margin: 0 10px;"> API Status</a>
                </div>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))
    
    def handle_status(self):
        """API endpoint for stream status"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        frame1_info = self.frame_manager.get_frame_info()['frame1']
        frame2_info = self.frame_manager.get_frame_info()['frame2']
        
        status = {
            'timestamp': time.time(),
            'sgt_time': self.get_sgt_time(),
            'streams': {
                'stream1': {
                    'url': '/stream1',
                    'description': 'Stream 1',
                    'external_frames': frame1_info['frame_count'],
                    'has_frame': frame1_info['has_frame']
                },
                'stream2': {
                    'url': '/stream2', 
                    'description': 'Stream 2',
                    'external_frames': frame2_info['frame_count'],
                    'has_frame': frame2_info['has_frame']
                }
            }
        }
        self.wfile.write(str(status).encode('utf-8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

# No main function here - only importable components
