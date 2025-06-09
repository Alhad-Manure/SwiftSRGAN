import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "thread_type;0|threads;1"
os.environ["FFMPEG_THREADS"] = "1"

import torch
#print(torch.version.cuda)
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Resampling
import cv2
import numpy as np
import time
import threading
from collections import deque



#from models import Generator

# ---------- CONFIGURATION ----------
# Input sources (choose one)
# Use 0 for default webcam, 1 for second camera, etc.
WEBCAM_INDEX = 0
# Set to video file path, or None to use webcam 
#VIDEO_FILE_PATH = None
VIDEO_FILE_PATH = './TestData/Video/En_WCE_record_0003_0000.mp4'

#MODEL_PATH = r'D:\\Mtech\\Sem2\\Mini_Project\\SRGAN\\Trainings\\Training2\\Results\\Models\\netG_4x_epoch245.pth.tar'
MODEL_PATH = './modelPts/optimized_model78.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Live streaming settings
TARGET_FPS = 30           # Target output FPS
BUFFER_SIZE = 5           # Frame buffer size for smooth playback
SKIP_FRAMES = 1           # Process every N frames (1 = process all, 2 = skip every other frame)
DISPLAY_SCALE = 0.4      # Scale factor for display window (1.0 = full size)
SHOW_ORIGINAL = False      # Show original video alongside processed
# -----------------------------------

class LiveSwiftSRGAN:
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
        
    def load_model(self):
        # Load the SRGAN model
        '''
        self.model = Generator()
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(DEVICE)
        '''

        self.model = torch.jit.load(MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Model loaded from {MODEL_PATH} on {DEVICE}")
        self.model.eval()
        # For debugging purposes
        #print(f"Model loaded successfully on {DEVICE}")
        
    def initialize_camera(self):
        """Initialize camera or video input"""
        if VIDEO_FILE_PATH:
            self.cap = cv2.VideoCapture(VIDEO_FILE_PATH, cv2.CAP_GSTREAMER)
            # For debugging purposes
            #print(f"Using video file: {VIDEO_FILE_PATH}")
        else:
            self.cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_GSTREAMER)
            # For debugging purposes
            #print(f"Using webcam index: {WEBCAM_INDEX}")
            
        if not self.cap.isOpened():
            raise Exception("Could not open video source")
            
        # Set camera properties for better performance
        # Only for webcam
        if not VIDEO_FILE_PATH:  
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            
        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Input resolution: {actual_width}x{actual_height} at {actual_fps:.1f} FPS")
        
    def frame_to_tensor(self, frame):
        """Convert OpenCV frame to tensor"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize(512, interpolation=Resampling.BICUBIC),
            transforms.ToTensor()
        ])
        
        return transform(pil_image)
        
    def tensot_to_frame(self, tensor):
        """Convert tensor back to OpenCV frame"""
        # Convert tensor to PIL Image
        transform_output = transforms.ToPILImage()
        pil_image = transform_output(tensor.clamp(0, 1))
        
        # Convert PIL to numpy array (RGB)
        frame_rgb = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
        
    def process_frame(self, frame):
        """Process a single frame through SRGAN"""
        try:
            start_time = time.time()
            
            # Preprocess
            frame_tensor = self.frame_to_tensor(frame).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                frame_tensor = frame_tensor.to(DEVICE)
                output_tensor = self.model(frame_tensor).cpu().squeeze(0)
            
            # Postprocess
            processed_frame = self.tensot_to_frame(output_tensor)
            
            # Track processing time
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
                if VIDEO_FILE_PATH and os.path.exists(VIDEO_FILE_PATH):  # Restart video file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Failed to capture frame")
                    break
                    
            # Add to buffer
            if len(self.frame_buffer) < BUFFER_SIZE:
                self.frame_buffer.append(frame.copy())
                
            time.sleep(1.0 / TARGET_FPS / 2)  # Control capture rate
            
    def frame_processing_thread(self):
        """Thread for processing frames"""
        while self.running:
            if self.frame_buffer:
                # Get frame from buffer
                frame = self.frame_buffer.popleft()
                
                # Skip frames if configured
                self.frame_count += 1
                if self.frame_count % SKIP_FRAMES != 0:
                    continue
                    
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Add to processed buffer
                if len(self.processed_buffer) < BUFFER_SIZE:
                    self.processed_buffer.append({
                        'original': frame,
                        'processed': processed_frame
                    })
                    
            else:
                time.sleep(0.001)  # Small delay when no frames available
                
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
            
    def draw_info_overlay(self, frame, title=""):
        """Draw performance information on frame"""
        # Calculate average processing time
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        processing_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        # Prepare info text
        info_lines = [
            f"{title}",
            f"Display FPS: {self.current_fps}",
            f"Processing FPS: {processing_fps:.1f}",
            f"Buffer: {len(self.processed_buffer)}/{BUFFER_SIZE}",
            f"Avg Process Time: {avg_processing_time*1000:.1f}ms"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
        return frame
        
    def resize_for_display(self, frame):
        """Resize frame for display"""
        if DISPLAY_SCALE != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * DISPLAY_SCALE)
            new_height = int(height * DISPLAY_SCALE)
            return cv2.resize(frame, (new_width, new_height))
        return frame
        
    def apply_live_processing(self):
        """live processing loop"""
        try:
            # for debugging purpose
            #print("Starting live SRGAN processing...")
            print("Press 'q' to quit, 's' to toggle original view")
            
            self.running = True
            
            # Start capture and processing threads
            capture_thread = threading.Thread(target=self.frame_capture_thread)
            processing_thread = threading.Thread(target=self.frame_processing_thread)
            
            capture_thread.daemon = True
            processing_thread.daemon = True
            
            capture_thread.start()
            processing_thread.start()
            
            show_original = SHOW_ORIGINAL
            
            while self.running:
                if self.processed_buffer:
                    # Get processed frame
                    frame_data = self.processed_buffer.popleft()
                    original_frame = frame_data['original']
                    processed_frame = frame_data['processed']
                    
                    # Add info overlay
                    display_original = self.draw_info_overlay(original_frame.copy(), "Original")
                    display_processed = self.draw_info_overlay(processed_frame.copy(), "SRGAN Enhanced")
                    
                    # Resize for display
                    display_original = self.resize_for_display(display_original)
                    display_processed = self.resize_for_display(display_processed)
                    
                    if show_original:
                        # Show both side by side
                        combined = np.hstack([display_original, display_processed])
                        cv2.imshow('SRGAN Live Processing - Original | Enhanced', combined)
                    else:
                        # Show only processed
                        cv2.imshow('SRGAN Live Processing - Enhanced Only', display_processed)
                    
                    self.calculate_fps()
                    
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    show_original = not show_original
                    cv2.destroyAllWindows()
                    print(f"Toggled original view: {'ON' if show_original else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            self.running = False
            
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            print("Live processing stopped")
            
    def apply(self):
        # Initialize and start processing
        try:
            self.load_model()
            self.initialize_camera()
            self.apply_live_processing()
            
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    print("=" * 50)
    print("Swift-SRGAN Live Streaming")
    print("=" * 50)
    
    # Check if model exists

    if not MODEL_PATH:
        print("Error: MODEL_PATH is not set. Please update the configuration section.")
        return

    '''
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in the configuration section.")
        return
        '''
    
    # For debugging purposes
    #print(f"Model: {MODEL_PATH}")
    #print(f"Device: {DEVICE}")
    
    if VIDEO_FILE_PATH:
        print(f"Input: Video file - {VIDEO_FILE_PATH}")
    else:
        print(f"Input: Webcam index {WEBCAM_INDEX}")
        
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Skip frames: {SKIP_FRAMES} (process every {SKIP_FRAMES} frame)")
    print(f"Display scale: {DISPLAY_SCALE}")
    print()
    
    # Apply Swift-SRGAN
    live_swift_srgan = LiveSwiftSRGAN()
    live_swift_srgan.apply()

if __name__ == "__main__":
    main()