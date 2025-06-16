import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "thread_type;0|threads;1"
os.environ["FFMPEG_THREADS"] = "1"

import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Resampling
import cv2
import numpy as np
import time
import threading
from collections import deque
from flask import Flask, render_template, Response, request, jsonify
import base64
import io

# Flask app
app = Flask(__name__)

# ---------- CONFIGURATION ----------
# Input sources (choose one)
WEBCAM_INDEX = 0
VIDEO_FILE_PATH = './TestData/Video/TinyWildLife.mp4'  # Set to None to use webcam
MODEL_PATH = './modelPts/optimized_model_v2.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Live streaming settings
TARGET_FPS = 30
BUFFER_SIZE = 5
SKIP_FRAMES = 1
DISPLAY_SCALE = 0.8  # Reduced for web streaming
QUALITY = 80  # JPEG quality for streaming (1-100)

# WebM/H264 streaming settings (much better than JPEG)
USE_H264 = True  # Use H264 instead of JPEG for better performance
H264_BITRATE = 1000000  # 1Mbps
H264_GOP = 30

# Flask settings
HOST = '0.0.0.0'  # Allow external connections
PORT = 8384
DEBUG = False
# -----------------------------------

class FlaskSwiftSRGAN:
    def __init__(self):
        self.model = None
        self.cap = None
        self.frame_buffer = deque(maxlen=BUFFER_SIZE)
        self.processed_buffer = deque(maxlen=BUFFER_SIZE)
        self.processing = False
        self.running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Performance tracking
        self.processing_times = deque(maxlen=30)
        
        # Stream control
        self.show_original = False
        self.stream_active = False
        
        # Current frames for streaming
        self.current_original = None
        self.current_processed = None
        self.current_combined = None
        
    def load_model(self):
        """Load the SRGAN model"""
        try:
            self.model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
            self.model.eval()
            print(f"Model loaded from {MODEL_PATH} on {DEVICE}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
    def initialize_camera(self):
        """Initialize camera or video input"""
        try:
            if VIDEO_FILE_PATH and os.path.exists(VIDEO_FILE_PATH):
                self.cap = cv2.VideoCapture(VIDEO_FILE_PATH)
                print(f"Using video file: {VIDEO_FILE_PATH}")
            else:
                self.cap = cv2.VideoCapture(WEBCAM_INDEX)
                print(f"Using webcam index: {WEBCAM_INDEX}")
                
            if not self.cap.isOpened():
                raise Exception("Could not open video source")
            # Set camera properties for webcam only
            #if not (VIDEO_FILE_PATH and os.path.exists(VIDEO_FILE_PATH)):
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Input resolution: {actual_width}x{actual_height} at {actual_fps:.1f} FPS")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
        
    def frame_to_tensor(self, frame):
        """Convert OpenCV frame to tensor"""
        
        '''
        PIL image and transform approach
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
        pil_image = Image.fromarray(frame_rgb)
        
        transform = transforms.Compose([
            transforms.Resize(512, interpolation=Resampling.BICUBIC),
            transforms.ToTensor()
        ])
        
        return transform(pil_image)
        '''

        height, width = frame.shape[:2]
        if width > 512 or height > 512:
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Direct numpy to tensor conversion (faster than PIL)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
        
        return frame_tensor


    def tensor_to_frame(self, tensor):
        """Convert tensor back to OpenCV frame"""

        '''
        PIL image and transform approach
        transform_output = transforms.ToPILImage()
        pil_image = transform_output(tensor.clamp(0, 1))
        frame_rgb = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr
        '''

        # Direct tensor to numpy conversion
        tensor_cpu = tensor.cpu().clamp(0, 1)
        frame_np = (tensor_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        return frame_bgr
        
    def process_frame(self, frame):
        """Process a single frame through SRGAN"""
        try:
            start_time = time.time()
            
            frame_tensor = self.frame_to_tensor(frame).unsqueeze(0)
            
            with torch.no_grad():
                frame_tensor = frame_tensor.to(DEVICE)
                output_tensor = self.model(frame_tensor).cpu().squeeze(0)
            
            processed_frame = self.tensor_to_frame(output_tensor)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return processed_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
            
    def frame_capture_thread(self):
        """Thread for capturing frames from video source"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if VIDEO_FILE_PATH and os.path.exists(VIDEO_FILE_PATH):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Failed to capture frame")
                    break
                    
            if len(self.frame_buffer) < BUFFER_SIZE:
                self.frame_buffer.append(frame.copy())
                
            time.sleep(1.0 / TARGET_FPS / 2)
            
    def frame_processing_thread(self):
        """Thread for processing frames"""
        while self.running:
            if self.frame_buffer and self.stream_active:
                frame = self.frame_buffer.popleft()
                
                self.frame_count += 1
                if self.frame_count % SKIP_FRAMES != 0:
                    continue
                    
                processed_frame = self.process_frame(frame)
                
                # Store current frames for streaming
                #DISPLAY_SCALE = 0.7
                self.current_original = self.add_info_overlay(frame.copy(), "Original")
                #DISPLAY_SCALE = 1.0
                self.current_processed = self.add_info_overlay(processed_frame.copy(), "SRGAN Enhanced")
                
                # Create combined view
                if self.current_original is not None and self.current_processed is not None:
                    # Resize for web display
                    #DISPLAY_SCALE = 1.5
                    orig_resized = self.resize_for_display(self.current_original)
                    #DISPLAY_SCALE = 0.5
                    proc_resized = self.resize_for_display(self.current_processed)
                    
                    '''
                    # Ensure both frames have the same dimensions
                    height = min(orig_resized.shape[0], proc_resized.shape[0])
                    orig_resized = orig_resized[:height, :]
                    proc_resized = proc_resized[:height, :]
                    '''

                    max_height = max(orig_resized.shape[0], proc_resized.shape[0])
                    pad_height = max_height - orig_resized.shape[0]
                    pad_top = pad_height // 2
                    pad_bottom = pad_height - pad_top
                    padding = ((pad_top, pad_bottom), (0, 0), (0, 0))
                    orig_resized = np.pad(orig_resized, padding, mode='constant', constant_values=0)

                    self.current_combined = np.hstack([orig_resized, proc_resized])
                    
            else:
                time.sleep(0.001)
                
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
            
    def add_info_overlay(self, frame, title=""):
        """Add performance information to frame"""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        processing_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        info_lines = [
            f"{title}",
            f"FPS: {self.current_fps}",
            f"Process FPS: {processing_fps:.1f}",
            f"Buffer: {len(self.processed_buffer)}/{BUFFER_SIZE}",
            f"Process Time: {avg_processing_time*1000:.1f}ms"
        ]
        
        # Draw background
        overlay = frame.copy()
        if title == "Original":
            cv2.rectangle(overlay, (10, 10), (220, 130), (0, 0, 0), -1)
        else:
            cv2.rectangle(overlay, (10, 10), (320, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        for i, line in enumerate(info_lines):
            if title == "Original":
                y_pos = 30 + i * 20
                cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                y_pos = 50 + i * 30
                cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_SCALE, (0, 255, 0), 1)
            
        return frame
        
    def resize_for_display(self, frame):
        """Resize frame for web display"""
        if DISPLAY_SCALE != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * DISPLAY_SCALE)
            new_height = int(height * DISPLAY_SCALE)
            return cv2.resize(frame, (new_width, new_height))
        return frame
        
    def start_processing(self):
        """Start the video processing threads"""
        if self.running:
            return False
            
        if not self.load_model():
            return False
            
        if not self.initialize_camera():
            return False
            
        self.running = True
        self.stream_active = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self.frame_capture_thread)
        self.processing_thread = threading.Thread(target=self.frame_processing_thread)
        
        self.capture_thread.daemon = True
        self.processing_thread.daemon = True
        
        self.capture_thread.start()
        self.processing_thread.start()
        
        print("Processing started")
        return True
        
    def stop_processing(self):
        """Stop the video processing"""
        self.running = False
        self.stream_active = False
        
        if self.cap:
            self.cap.release()
        
        print("Processing stopped")
        
    def get_frame_jpeg(self, frame_type='combined'):
        """Convert frame to JPEG bytes for streaming"""
        try:
            if frame_type == 'original' and self.current_original is not None:
                frame = self.resize_for_display(self.current_original)
            elif frame_type == 'processed' and self.current_processed is not None:
                frame = self.resize_for_display(self.current_processed)
            elif frame_type == 'combined' and self.current_combined is not None:
                frame = self.current_combined
            else:
                # Return placeholder frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Video", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            '''Simple Encoding
            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
            result, encimg = cv2.imencode('.jpg', frame, encode_param)
            '''

            # Optimized encoding
            if USE_H264 and frame.shape[0] > 0 and frame.shape[1] > 0:
                # Use optimized JPEG with better settings
                encode_param = [
                    int(cv2.IMWRITE_JPEG_QUALITY), QUALITY,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
                    int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1
                ]
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
            
            result, encimg = cv2.imencode('.jpg', frame, encode_param)
            
            if result:
                return encimg.tobytes()
            else:
                return None
                
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None

# Global processor instance
processor = FlaskSwiftSRGAN()

def generate_frames(stream_type='combined'):
    """Generate frames for video streaming"""
    while True:
        if processor.stream_active:
            frame_bytes = processor.get_frame_jpeg(stream_type)
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                processor.calculate_fps()
        time.sleep(1.0 / TARGET_FPS)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed/<stream_type>')
def video_feed(stream_type='combined'):
    """Video streaming route"""
    return Response(generate_frames(stream_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Start video processing"""
    success = processor.start_processing()
    return jsonify({'success': success, 'message': 'Processing started' if success else 'Failed to start processing'})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    """Stop video processing"""
    processor.stop_processing()
    return jsonify({'success': True, 'message': 'Processing stopped'})

@app.route('/toggle_view', methods=['POST'])
def toggle_view():
    """Toggle between original and processed view"""
    processor.show_original = not processor.show_original
    return jsonify({'success': True, 'show_original': processor.show_original})

@app.route('/status')
def status():
    """Get current status"""
    return jsonify({
        'running': processor.running,
        'stream_active': processor.stream_active,
        'fps': processor.current_fps,
        'device': DEVICE,
        'model_loaded': processor.model is not None
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Swift-SRGAN Flask Video Streaming Server")
    print("=" * 50)
    print(f"Device: {DEVICE}") 
    print(f"Model: {MODEL_PATH}")
    print(f"Video source: {'File - ' + VIDEO_FILE_PATH if VIDEO_FILE_PATH and os.path.exists(VIDEO_FILE_PATH) else 'Webcam'}")
    print(f"Server will run on http://{HOST}:{PORT}")
    print("=" * 50)
    
    try:
        app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        processor.stop_processing()
    except Exception as e:
        print(f"Error running server: {e}")
        processor.stop_processing()