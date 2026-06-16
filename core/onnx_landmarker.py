import os
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort

class ONNXFaceMeshDetector:
    """
    Alternative landmark inference backend utilizing ONNX Runtime
    with DirectML (on Windows) or CPU execution provider.
    Uses YuNet for face detection and MediaPipe FaceMesh for landmarking.
    """
    def __init__(self):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.detector_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
        self.landmarker_path = os.path.join(model_dir, "face_landmark_detector.onnx")
        
        # Download models if they don't exist
        self._check_and_download_models()
        
        # Setup execution providers (prefer DirectML for GPU acceleration on Windows)
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        print(f"Initializing ONNX FaceMesh Landmarker with providers: {providers}")
        
        try:
            self.landmarker_session = ort.InferenceSession(self.landmarker_path, providers=providers)
            active_providers = self.landmarker_session.get_providers()
            print(f"ONNX active execution providers: {active_providers}")
        except Exception as e:
            print(f"ONNX DirectML initialization failed, falling back to CPU: {e}")
            self.landmarker_session = ort.InferenceSession(self.landmarker_path, providers=['CPUExecutionProvider'])
            
        # Initialize OpenCV YuNet face detector
        self.detector = cv2.FaceDetectorYN.create(self.detector_path, "", (320, 320))

    def _check_and_download_models(self):
        # 1. OpenCV YuNet Face Detector ONNX model
        if not os.path.exists(self.detector_path):
            print("Downloading YuNet Face Detector ONNX model...")
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            try:
                urllib.request.urlretrieve(url, self.detector_path)
                print("YuNet face detector ONNX model downloaded successfully!")
            except Exception as e:
                print(f"Failed to download YuNet model: {e}")
                
        # 2. Qualcomm MediaPipe Face Landmark ONNX model & data weights
        data_path = os.path.join(os.path.dirname(self.landmarker_path), "face_landmark_detector.data")
        if not os.path.exists(self.landmarker_path) or not os.path.exists(data_path):
            print("Downloading and extracting MediaPipe Face Landmark ONNX model & data...")
            url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mediapipe_face/releases/v0.55.0/mediapipe_face-onnx-float.zip"
            try:
                import zipfile
                import io
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    zip_data = response.read()
                
                with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                    # Extract face_landmark_detector.onnx
                    onnx_content = z.read("mediapipe_face-onnx-float/face_landmark_detector.onnx")
                    with open(self.landmarker_path, "wb") as f:
                        f.write(onnx_content)
                    
                    # Extract face_landmark_detector.data
                    data_content = z.read("mediapipe_face-onnx-float/face_landmark_detector.data")
                    with open(data_path, "wb") as f:
                        f.write(data_content)
                print("MediaPipe Face Landmark ONNX model and data extracted successfully!")
            except Exception as e:
                print(f"Failed to download/extract Face Landmark model: {e}")

    def detect(self, image):
        """
        Runs face detection and landmarking on the input BGR image.
        Returns a list of landmark arrays (shape: [num_faces, 468, 2]).
        """
        h, w = image.shape[:2]
        
        # Optimize face detection speed by scaling the image down for YuNet
        max_det_dim = 320
        if max(h, w) > max_det_dim:
            scale = max_det_dim / float(max(h, w))
            det_w = int(w * scale)
            det_h = int(h * scale)
            det_img = cv2.resize(image, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            det_w, det_h = w, h
            det_img = image
            
        # 1. Run face detection
        self.detector.setInputSize((det_w, det_h))
        _, faces = self.detector.detect(det_img)
        
        if faces is None:
            return []
            
        landmarks_list = []
        
        # Process at most 2 faces (similar to self.landmarker setup)
        for face in faces[:2]:
            # Bounding box coords (scale back to original resolution)
            box_x = int(face[0] / scale)
            box_y = int(face[1] / scale)
            box_w = int(face[2] / scale)
            box_h = int(face[3] / scale)
            
            # Apply padding margin (mediapipe expects slightly expanded crop region)
            pad_x = int(box_w * 0.15)
            pad_y = int(box_h * 0.15)
            
            crop_x = max(0, box_x - pad_x)
            crop_y = max(0, box_y - pad_y)
            crop_w = min(w - crop_x, box_w + 2 * pad_x)
            crop_h = min(h - crop_y, box_h + 2 * pad_y)
            
            if crop_w <= 10 or crop_h <= 10:
                continue
                
            crop_img = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # 2. Preprocess crop for landmarker (Qualcomm model expects 192x192 RGB input, shape NCHW)
            resized = cv2.resize(crop_img, (192, 192))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0.0, 1.0] and format to (1, 3, 192, 192)
            input_tensor = rgb.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # 3. Run landmark model inference
            input_name = self.landmarker_session.get_inputs()[0].name
            outputs = self.landmarker_session.run(None, {input_name: input_tensor})
            
            # Outputs shape is ('scores', 'landmarks'), landmarks shape (1, 468, 3)
            # Coordinates are normalized relative to the 192x192 crop image
            landmarks = outputs[1][0]
            
            # 4. Map back to original image coordinate space
            mapped_landmarks = []
            for landmark in landmarks:
                # Qualcomm model outputs landmarks already normalized in [0.0, 1.0]
                norm_x = landmark[0]
                norm_y = landmark[1]
                
                lm_x = crop_x + norm_x * crop_w
                lm_y = crop_y + norm_y * crop_h
                mapped_landmarks.append((float(lm_x), float(lm_y)))
                
            landmarks_list.append(np.array(mapped_landmarks, dtype=np.float32))
            
        return landmarks_list
