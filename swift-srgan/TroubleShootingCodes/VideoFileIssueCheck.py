import cv2
import os
import subprocess
import sys

def debug_video_file(video_path):
    """Comprehensive debugging for video file issues on Linux"""
    
    print("=== VIDEO FILE DEBUGGING ===")
    print(f"Video path: {video_path}")
    print()
    
    # 1. Basic file checks
    print("1. BASIC FILE CHECKS:")
    print(f"   File exists: {os.path.exists(video_path)}")
    print(f"   File is file: {os.path.isfile(video_path)}")
    print(f"   File readable: {os.access(video_path, os.R_OK)}")
    print(f"   File size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")
    print()
    
    # 2. File permissions
    print("2. FILE PERMISSIONS:")
    if os.path.exists(video_path):
        stat_info = os.stat(video_path)
        permissions = oct(stat_info.st_mode)[-3:]
        print(f"   Permissions: {permissions}")
        print(f"   Owner readable: {bool(stat_info.st_mode & 0o400)}")
        print(f"   Group readable: {bool(stat_info.st_mode & 0o040)}")
        print(f"   Other readable: {bool(stat_info.st_mode & 0o004)}")
    print()
    
    # 3. OpenCV build info
    print("3. OPENCV BUILD INFO:")
    build_info = cv2.getBuildInformation()
    
    # Check for important video-related components
    video_components = ['FFMPEG', 'GStreamer', 'Video I/O']
    for component in video_components:
        if component in build_info:
            print(f"   ✓ {component} support detected")
        else:
            print(f"   ✗ {component} support not clearly detected")
    
    # Check backend
    backends = cv2.videoio_registry.getBackends()
    print(f"   Available backends: {backends}")
    print()
    
    # 4. Try different VideoCapture methods
    print("4. TESTING DIFFERENT OPENCV METHODS:")
    
    methods = [
        ("Default", lambda path: cv2.VideoCapture(path)),
        ("CAP_FFMPEG", lambda path: cv2.VideoCapture(path, cv2.CAP_FFMPEG)),
        ("CAP_GSTREAMER", lambda path: cv2.VideoCapture(path, cv2.CAP_GSTREAMER)),
    ]
    
    working_method = None
    for name, method in methods:
        try:
            cap = method(video_path)
            is_opened = cap.isOpened()
            print(f"   {name}: {'✓ SUCCESS' if is_opened else '✗ FAILED'}")
            
            if is_opened and working_method is None:
                working_method = (name, method)
                # Try to read a frame
                ret, frame = cap.read()
                print(f"   {name}: Frame read {'✓ SUCCESS' if ret else '✗ FAILED'}")
                if ret:
                    print(f"   {name}: Frame shape {frame.shape}")
            
            cap.release()
        except Exception as e:
            print(f"   {name}: ✗ ERROR - {e}")
    print()
    
    # 5. FFmpeg check
    print("5. FFMPEG/CODEC CHECK:")
    try:
        result = subprocess.run(['ffmpeg', '-i', video_path], 
                              capture_output=True, text=True)
        if "Invalid data found" in result.stderr:
            print("   ✗ Video file appears corrupted")
        elif "No such file" in result.stderr:
            print("   ✗ FFmpeg cannot find the file")
        else:
            print("   ✓ FFmpeg can read the file")
            # Extract codec info
            if "Video:" in result.stderr:
                video_line = [line for line in result.stderr.split('\n') if 'Video:' in line]
                if video_line:
                    print(f"   Video codec info: {video_line[0].strip()}")
    except FileNotFoundError:
        print("   ✗ FFmpeg not installed")
    except Exception as e:
        print(f"   ✗ FFmpeg check failed: {e}")
    print()
    
    # 6. Recommendations
    print("6. RECOMMENDATIONS:")
    if not os.path.exists(video_path):
        print("   • File doesn't exist - check the path")
    elif not os.access(video_path, os.R_OK):
        print(f"   • Fix permissions: chmod +r '{video_path}'")
    elif working_method:
        print(f"   • Use {working_method[0]} method")
    else:
        print("   • Try converting video: ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4")
        print("   • Install OpenCV with full codec support: pip uninstall opencv-python && pip install opencv-contrib-python")
        print("   • Install system codecs: sudo apt-get install ubuntu-restricted-extras")
    
    return working_method

def fixed_initialize_camera(video_file_path=None, webcam_index=0):
    """Fixed version that tries multiple methods"""
    
    if video_file_path and os.path.exists(video_file_path):
        print(f"Attempting to open video file: {video_file_path}")
        
        # Debug the file first
        working_method = debug_video_file(video_file_path)
        
        if working_method:
            name, method = working_method
            print(f"\nUsing {name} method...")
            cap = method(video_file_path)
            if cap.isOpened():
                return cap
        
        # Try all backends systematically
        backends_to_try = [
            cv2.CAP_FFMPEG,
            cv2.CAP_GSTREAMER,
            cv2.CAP_V4L2,
            cv2.CAP_ANY
        ]
        
        for backend in backends_to_try:
            try:
                print(f"Trying backend: {backend}")
                cap = cv2.VideoCapture(video_file_path, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
                        print(f"SUCCESS with backend {backend}")
                        return cap
                cap.release()
            except Exception as e:
                print(f"Backend {backend} failed: {e}")
        
        raise Exception(f"Could not open video file with any method: {video_file_path}")
    
    else:
        # Webcam fallback
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        return cap

def install_missing_codecs():
    """Provide installation commands for missing codecs"""
    print("\n=== CODEC INSTALLATION COMMANDS ===")
    print("Try these commands to install missing video codecs:\n")
    
    print("# Update OpenCV with full codec support:")
    print("pip uninstall opencv-python opencv-contrib-python")
    print("pip install opencv-contrib-python")
    print()
    
    print("# Install system codecs (Ubuntu/Debian):")
    print("sudo apt update")
    print("sudo apt install ubuntu-restricted-extras")
    print("sudo apt install ffmpeg")
    print("sudo apt install libavcodec-dev libavformat-dev libswscale-dev")
    print("sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev")
    print()
    
    print("# For CentOS/RHEL:")
    print("sudo yum install epel-release")
    print("sudo yum install ffmpeg ffmpeg-devel")
    print()
    
    print("# Convert problematic video:")
    print("ffmpeg -i input.mp4 -c:v libx264 -c:a aac -strict experimental output.mp4")

# Example usage
if __name__ == "__main__":
    # Replace with your actual video path
    video_path = './TestData/Video/En_WCE_record_0003_0000.mp4'
    
    try:
        cap = fixed_initialize_camera(video_path)
        print("✓ Video opened successfully!")
        
        # Read a few frames to verify
        for i in range(3):
            ret, frame = cap.read()
            if ret:
                print(f"Frame {i+1}: {frame.shape}")
            else:
                print(f"Could not read frame {i+1}")
                break
        
        cap.release()
        
    except Exception as e:
        print(f"✗ Failed to open video: {e}")
        install_missing_codecs()