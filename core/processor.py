import cv2
import time
import os
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
import numpy as np

CODEC_MAP = {
    "H.264 (MP4)": ("mp4v", ".mp4"),
    "H.264 (MOV)": ("avc1", ".mov"),
    "ProRes (Standard)": ("apcn", ".mov"),
    "ProRes (HQ)": ("apch", ".mov"),
    "DNxHD (MOV)": ("AVdn", ".mov")
}

class VideoProcessorThread(QThread):
    """
    Worker thread that reads video frames in a batch, applies filters,
    and writes them to an output file.
    """
    progress = pyqtSignal(int, float, str)  # frame_idx, percentage, eta_string
    completed = pyqtSignal(str, bool)       # output_path, success
    error = pyqtSignal(str)                 # error_msg
    
    def __init__(self, input_path, output_path, codec_name, filter_engine, params):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.codec_name = codec_name
        self.filter_engine = filter_engine
        self.params = params
        self._is_running = True
        
    def stop(self):
        self._is_running = False
        
    def run(self):
        try:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.error.emit(f"Could not open input video: {self.input_path}")
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if total_frames <= 0 or fps <= 0:
                self.error.emit("Invalid video metadata (frame count or frame rate is zero)")
                cap.release()
                return

            # Retrieve codec parameters
            fourcc_str, ext = CODEC_MAP.get(self.codec_name, ("mp4v", ".mp4"))
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            
            # Ensure the output path matches the codec extension
            base, _ = os.path.splitext(self.output_path)
            actual_output_path = base + ext
            
            # Initialize VideoWriter
            writer = cv2.VideoWriter(actual_output_path, fourcc, fps, (width, height))
            use_fallback = False
            
            if not writer.isOpened():
                # Fallback to standard MP4 H.264
                print(f"Failed to open with codec {fourcc_str}, falling back to standard H.264 (mp4v)...")
                fourcc_str = "mp4v"
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                actual_output_path = base + ".mp4"
                writer = cv2.VideoWriter(actual_output_path, fourcc, fps, (width, height))
                use_fallback = True
                if not writer.isOpened():
                    self.error.emit("Failed to initialize video writer with both primary and fallback codecs.")
                    cap.release()
                    return
            
            start_time = time.time()
            frame_idx = 0
            
            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process the frame
                processed_frame = self.filter_engine.process_frame(frame, self.params)
                
                # Write frame
                writer.write(processed_frame)
                
                frame_idx += 1
                
                # Progress and ETA calculations
                elapsed = time.time() - start_time
                percentage = (frame_idx / total_frames) * 100.0
                
                # Calculate speed and ETA
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                remaining_frames = total_frames - frame_idx
                eta_seconds = remaining_frames / current_fps if current_fps > 0 else 0
                
                # Format ETA
                if eta_seconds > 60:
                    eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                else:
                    eta_str = f"{int(eta_seconds)}s"
                    
                self.progress.emit(frame_idx, percentage, eta_str)
                
            writer.release()
            cap.release()
            
            if self._is_running:
                self.completed.emit(actual_output_path, use_fallback)
            else:
                # If stopped mid-run, clean up partially written file
                if os.path.exists(actual_output_path):
                    try:
                        os.remove(actual_output_path)
                    except:
                        pass
                        
        except Exception as e:
            self.error.emit(str(e))


class WebcamThread(QThread):
    """
    Worker thread to capture live frames from the webcam, apply the current filters,
    and emit the frame as BGR numpy arrays to be displayed in the UI.
    """
    frame_ready = pyqtSignal(np.ndarray, np.ndarray)  # processed BGR, original BGR frame
    error = pyqtSignal(str)
    
    def __init__(self, filter_engine, params_getter):
        super().__init__()
        self.filter_engine = filter_engine
        self.params_getter = params_getter  # Callback function to retrieve latest sliders state
        self._is_running = True
        self.camera_idx = 0
        
    def stop(self):
        self._is_running = False
        
    def run(self):
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            self.error.emit("Could not access webcam (index 0)")
            return
            
        # Optional optimization for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Flip horizontally to act like a mirror for natural look
            frame = cv2.flip(frame, 1)
            
            # Fetch parameters dynamically (so sliders update live)
            params = self.params_getter()
            
            # Process the frame
            processed = self.filter_engine.process_frame(frame, params)
            
            # Emit copies to avoid memory segment faults in Qt event queue
            self.frame_ready.emit(processed.copy(), frame)
            
            # Yield CPU slice
            time.sleep(0.01)
            
        cap.release()



class VideoReader:
    """
    Helper class to manage OpenCV VideoCapture and facilitate direct seeking/scrubbing.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_index = -1
        
    def read_frame(self, frame_idx):
        """
        Seek to and read the frame at frame_idx. Returns BGR frame.
        """
        if not self.cap.isOpened():
            return None
            
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
            
        # Seeking in OpenCV: set property POS_FRAMES
        # If it's the next frame, we don't need to seek, just read (which is faster)
        if frame_idx != self.current_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
        ret, frame = self.cap.read()
        if ret:
            self.current_index = frame_idx
            return frame
        return None
        
    def release(self):
        if self.cap.isOpened():
            self.cap.release()
