import cv2
import time
import os
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import queue
from threading import Thread

CODEC_MAP = {
    "H.264 (MP4)": ("mp4v", ".mp4"),
    "H.264 (MOV)": ("avc1", ".mov"),
    "ProRes (Standard)": ("apcn", ".mov"),
    "ProRes (HQ)": ("apch", ".mov"),
    "DNxHD (MOV)": ("AVdn", ".mov"),
    "ProRes (4444 Alpha)": ("ap4h", ".mov")
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
        errors = []
        
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
            is_prores_alpha = (self.codec_name == "ProRes (4444 Alpha)")
            base, _ = os.path.splitext(self.output_path)
            
            if is_prores_alpha:
                import av
                actual_output_path = base + ".mov"
                container = av.open(actual_output_path, mode='w')
                stream = container.add_stream('prores_ks', rate=fps)
                stream.width = width
                stream.height = height
                stream.pix_fmt = 'yuva444p10le'
                stream.options = {'profile': '4'}  # ProRes 4444 profile
                writer = None
                use_fallback = False
            else:
                fourcc_str, ext = CODEC_MAP.get(self.codec_name, ("mp4v", ".mp4"))
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
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
            
            # Thread-safe queues
            raw_queue = queue.Queue(maxsize=32)
            processed_queue = queue.Queue(maxsize=32)
            
            params_copy = self.params.copy()
            if is_prores_alpha:
                params_copy['export_alpha'] = True

            def reader_worker():
                nonlocal errors
                try:
                    frame_idx = 0
                    while self._is_running:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                        
                        placed = False
                        while self._is_running:
                            try:
                                raw_queue.put((frame_idx, frame), timeout=0.1)
                                placed = True
                                break
                            except queue.Full:
                                continue
                        if not placed:
                            break
                        frame_idx += 1
                except Exception as e:
                    errors.append(f"Reader Thread error: {str(e)}")
                    self._is_running = False
                finally:
                    try:
                        raw_queue.put((None, None), timeout=1.0)
                    except queue.Full:
                        pass

            def processor_worker():
                nonlocal errors, params_copy
                try:
                    while self._is_running:
                        frame_idx, frame = None, None
                        got_item = False
                        while self._is_running:
                            try:
                                frame_idx, frame = raw_queue.get(timeout=0.1)
                                got_item = True
                                break
                            except queue.Empty:
                                continue
                        
                        if not got_item or frame is None:
                            break
                            
                        # Process the frame
                        processed_frame = self.filter_engine.process_frame(frame, params_copy)
                        
                        placed = False
                        while self._is_running:
                            try:
                                processed_queue.put((frame_idx, processed_frame), timeout=0.1)
                                placed = True
                                break
                            except queue.Full:
                                continue
                        if not placed:
                            break
                except Exception as e:
                    errors.append(f"Processor Thread error: {str(e)}")
                    self._is_running = False
                finally:
                    try:
                        processed_queue.put((None, None), timeout=1.0)
                    except queue.Full:
                        pass

            def writer_worker():
                nonlocal errors, is_prores_alpha, stream, container, writer, use_fallback, actual_output_path
                try:
                    written_count = 0
                    while self._is_running:
                        frame_idx, processed_frame = None, None
                        got_item = False
                        while self._is_running:
                            try:
                                frame_idx, processed_frame = processed_queue.get(timeout=0.1)
                                got_item = True
                                break
                            except queue.Empty:
                                continue
                                
                        if not got_item or processed_frame is None:
                            break
                            
                        # Write frame
                        if is_prores_alpha:
                            frame_rgba = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2RGBA)
                            av_frame = av.VideoFrame.from_ndarray(frame_rgba, format='rgba')
                            for packet in stream.encode(av_frame):
                                container.mux(packet)
                        else:
                            writer.write(processed_frame)
                            
                        written_count += 1
                        
                        # Progress and ETA calculations
                        elapsed = time.time() - start_time
                        percentage = (written_count / total_frames) * 100.0
                        
                        # Calculate speed and ETA
                        current_fps = written_count / elapsed if elapsed > 0 else 0
                        remaining_frames = total_frames - written_count
                        eta_seconds = remaining_frames / current_fps if current_fps > 0 else 0
                        
                        # Format ETA
                        if eta_seconds > 60:
                            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                        else:
                            eta_str = f"{int(eta_seconds)}s"
                            
                        self.progress.emit(written_count, percentage, eta_str)
                except Exception as e:
                    errors.append(f"Writer Thread error: {str(e)}")
                    self._is_running = False

            # Start threads
            t_reader = Thread(target=reader_worker, daemon=True)
            t_processor = Thread(target=processor_worker, daemon=True)
            t_writer = Thread(target=writer_worker, daemon=True)
            
            t_reader.start()
            t_processor.start()
            t_writer.start()
            
            # Wait for all threads to complete
            while t_reader.is_alive() or t_processor.is_alive() or t_writer.is_alive():
                t_reader.join(0.05)
                t_processor.join(0.05)
                t_writer.join(0.05)
                if errors:
                    self._is_running = False
            
            # Close/release
            if is_prores_alpha:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
            else:
                writer.release()
            cap.release()
            
            if errors:
                raise Exception("\n".join(errors))
                
            if self._is_running:
                self.completed.emit(actual_output_path, use_fallback)
            else:
                # Cleanup if stopped
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
