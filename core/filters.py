import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request

# Static lists of MediaPipe Face Mesh landmark indices
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW_INDICES = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

class BeautyFilterEngine:
    def __init__(self):
        # Locate/download the face landmarker model file
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(model_dir, "face_landmarker.task")
        
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe Face Landmarker task model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("Model downloaded successfully!")
            except Exception as e:
                print(f"Failed to download Face Landmarker model: {e}")
                
        # Initialize Face Landmarker task
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=2
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Temporal landmark tracking history for raw and warped passes (anti-jitter)
        self.face_history_raw = {}
        self.face_history_warped = {}
        self.next_id_raw = 0
        self.next_id_warped = 0

    def process_frame(self, image, params, preview_width=None):
        """
        Process a single image frame (BGR format) and apply the beauty filters.
        params: dict of filter parameters, each in range 0.0 to 1.0:
            - 'skin_smoothing'
            - 'blush_warmth'
            - 'skin_brightening'
            - 'eye_enhancement'
            - 'undereye_lighten'
            - 'nose_reduce'
            - 'cheeks_reduce'
            - 'forehead_reduce'
            - 'color_look' (string, e.g. "None", "Teal & Orange")
            - 'look_intensity' (0.0 to 1.0)
        """
        # Downscale for performance if preview_width is specified
        h, w = image.shape[:2]
        if preview_width is not None and w > preview_width:
            scale = preview_width / float(w)
            new_h = int(h * scale)
            image = cv2.resize(image, (preview_width, new_h), interpolation=cv2.INTER_AREA)
            
        processed = image.copy()
        h, w = processed.shape[:2]
        
        export_alpha = params.get('export_alpha', False)
        if export_alpha:
            accum_alpha = np.zeros((h, w), dtype=np.float32)
            accum_color = np.zeros((h, w, 3), dtype=np.float32)
            accumulators = (accum_color, accum_alpha)
        else:
            accumulators = None
            
        # Initialize temporal history tracking maps for current frame
        self.new_history_raw = {}
        self.new_history_warped = {}
        
        # Convert OpenCV BGR image to RGB order and wrap in mp.Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Run synchronous landmark inference
        results = self.landmarker.detect(mp_image)
        
        if not results.face_landmarks:
            if export_alpha:
                self.face_history_raw = {}
                self.face_history_warped = {}
                return np.zeros((h, w, 4), dtype=np.uint8)
            # Still apply color looks if no face is detected
            color_look = params.get('color_look', 'None')
            if color_look != 'None':
                processed = self.apply_color_filter(processed, color_look, params.get('look_intensity', 1.0))
            # Clean up history if tracking is completely lost
            self.face_history_raw = {}
            self.face_history_warped = {}
            return processed
            
        # Smooth raw landmarks coordinates
        raw_faces_coords = []
        for face_landmarks in results.face_landmarks:
            coords = self.get_smoothed_coords(face_landmarks, w, h, channel='raw')
            raw_faces_coords.append(coords)
            
        # 1. Apply face reshaping warps (applied directly to the image coordinates)
        # We only warp if any warp strength is above 0
        nose_r = params.get('nose_reduce', 0.0)
        cheeks_r = params.get('cheeks_reduce', 0.0)
        forehead_r = params.get('forehead_reduce', 0.0)
        eye_e = params.get('eye_enlarge', 0.0)
        lips_p = params.get('lips_plump', 0.0)
        
        if nose_r > 0.001 or cheeks_r > 0.001 or forehead_r > 0.001 or eye_e > 0.001 or lips_p > 0.001:
            processed = self.apply_reshape_warps(processed, raw_faces_coords, nose_r, cheeks_r, forehead_r, eye_e, lips_p)
            
            # Re-detect landmarks on warped image to ensure exact filter overlays
            warped_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            warped_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=warped_rgb)
            results = self.landmarker.detect(warped_mp)
            if not results.face_landmarks:
                if export_alpha:
                    self.face_history_raw = self.new_history_raw
                    self.face_history_warped = {}
                    return np.zeros((h, w, 4), dtype=np.uint8)
                # Fallback: color look and return
                color_look = params.get('color_look', 'None')
                if color_look != 'None':
                    processed = self.apply_color_filter(processed, color_look, params.get('look_intensity', 1.0))
                # Sync raw tracking history only before return
                self.face_history_raw = self.new_history_raw
                self.face_history_warped = {}
                return processed
                
        # Smooth warped landmarks coordinates
        warped_faces_coords = []
        for face_landmarks in results.face_landmarks:
            coords = self.get_smoothed_coords(face_landmarks, w, h, channel='warped')
            warped_faces_coords.append(coords)
            
        # Apply filters for each detected face using smoothed coordinates
        for coords in warped_faces_coords:
            # 2. Generate skin mask (covers entire skin including nose)
            skin_mask = self.get_skin_mask(processed, coords)
            
            # 3. Apply skin brightening
            if params.get('skin_brightening', 0.0) > 0.001:
                processed = self.apply_skin_brightening(processed, skin_mask, params['skin_brightening'], accumulators=accumulators)
            
            # 4. Apply skin smoothing (with high-pass texture recovery)
            if params.get('skin_smoothing', 0.0) > 0.001:
                texture_rec = params.get('skin_texture_recovery', 0.0)
                processed = self.apply_skin_smoothing(processed, skin_mask, params['skin_smoothing'], texture_rec, accumulators=accumulators)
                
            # 5. Apply blush / warmth to cheeks
            if params.get('blush_warmth', 0.0) > 0.001:
                processed = self.apply_blush(processed, coords, params['blush_warmth'], accumulators=accumulators)
                
            # 6. Apply under-eye lighten
            if params.get('undereye_lighten', 0.0) > 0.001:
                processed = self.apply_undereye_lighten(processed, coords, None, params['undereye_lighten'], accumulators=accumulators)
                
            # 7. Apply eye enhancement (contrast & clarity)
            if params.get('eye_enhancement', 0.0) > 0.001:
                processed = self.apply_eye_enhancement(processed, coords, params['eye_enhancement'], accumulators=accumulators)
                
            # 7.5. Apply lips color makeup tint (lipstick overlay)
            lip_shade = params.get('lipstick_shade', 'None')
            lip_strength = params.get('lipstick_strength', 0.0)
            if lip_shade != 'None' and lip_strength > 0.001:
                processed = self.apply_lipstick(processed, coords, lip_strength, lip_shade, accumulators=accumulators)
                
            # 7.6. Apply eye color makeup tint (colored contact lenses)
            eye_color = params.get('eye_color_shade', 'Natural')
            eye_color_strength = params.get('eye_color_strength', 0.0)
            if eye_color != 'Natural' and eye_color_strength > 0.001:
                processed = self.apply_eye_color(processed, coords, eye_color_strength, eye_color, accumulators=accumulators)
        
        if export_alpha:
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            safe_alpha = np.where(accum_alpha > 0.0001, accum_alpha, 1.0)
            safe_alpha_3d = np.expand_dims(safe_alpha, axis=2)
            bgra_rgb = accum_color / safe_alpha_3d
            
            bgra[:, :, :3] = np.clip(bgra_rgb, 0, 255).astype(np.uint8)
            bgra[:, :, 3] = np.clip(accum_alpha * 255.0, 0, 255).astype(np.uint8)
            
            # Update history tracking logs
            self.face_history_raw = self.new_history_raw
            self.face_history_warped = self.new_history_warped
            return bgra

        # 8. Apply color looks
        color_look = params.get('color_look', 'None')
        if color_look != 'None':
            processed = self.apply_color_filter(processed, color_look, params.get('look_intensity', 1.0))
            
        # Update history tracking logs
        self.face_history_raw = self.new_history_raw
        self.face_history_warped = self.new_history_warped
                
        return processed

    def get_smoothed_coords(self, face_landmarks, w, h, channel='raw'):
        """
        Track and smooth face landmarks over time using centroid-matching and
        an Exponential Moving Average (EMA) to eliminate frame-to-frame wiggles.
        """
        # Convert landmarks to pixel coordinates
        coords = np.array([(float(l.x * w), float(l.y * h)) for l in face_landmarks], dtype=np.float32)
        centroid = np.mean(coords, axis=0)
        
        # Calculate face scale
        face_width = np.linalg.norm(coords[234] - coords[454])
        if face_width < 10.0:
            face_width = 10.0
            
        # Pick trackers mapping based on channel
        history = self.face_history_raw if channel == 'raw' else self.face_history_warped
        new_history = self.new_history_raw if channel == 'raw' else self.new_history_warped
        
        # Match current face to historical track based on centroid distance
        match_id = None
        min_dist = float('inf')
        threshold = face_width * 0.18  # Reset tracking if moved past 18% of face scale
        
        for face_id, (hist_centroid, hist_coords) in history.items():
            dist = np.linalg.norm(centroid - hist_centroid)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                match_id = face_id
                
        # EMA weight (0.55 current frame, 0.45 history) balances response vs stability
        alpha = 0.55
        
        if match_id is not None:
            prev_coords = history[match_id][1]
            smoothed_coords = alpha * coords + (1.0 - alpha) * prev_coords
            smoothed_centroid = np.mean(smoothed_coords, axis=0)
            new_history[match_id] = (smoothed_centroid, smoothed_coords)
            return smoothed_coords.astype(np.int32)
        else:
            # Assign new track ID
            next_id = self.next_id_raw if channel == 'raw' else self.next_id_warped
            if channel == 'raw':
                self.next_id_raw += 1
            else:
                self.next_id_warped += 1
            new_history[next_id] = (centroid, coords)
            return coords.astype(np.int32)

    def get_skin_mask(self, image, coords):
        h, w = image.shape[:2]
        
        # 1. Face Oval Mask
        oval_coords = coords[FACE_OVAL_INDICES]
        hull = cv2.convexHull(oval_coords)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull, 255)
        
        # 2. Subtraction mask for eyes, eyebrows, and lips (Note: nostrils are left inside the skin mask)
        exclude_mask = np.zeros((h, w), dtype=np.uint8)
        
        def fill_features(indices, color=255):
            pts = coords[indices]
            hull_pts = cv2.convexHull(pts)
            cv2.fillConvexPoly(exclude_mask, hull_pts, color)
                
        # Exclude regions
        fill_features(LEFT_EYE_INDICES)
        fill_features(RIGHT_EYE_INDICES)
        fill_features(LEFT_EYEBROW_INDICES)
        fill_features(RIGHT_EYEBROW_INDICES)
        fill_features(LIPS_INDICES)
        
        # Landmarked face area minus features
        final_skin_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(exclude_mask))
        
        # Soften and feather the skin mask to prevent hard edges
        feather_size = int(max(w, h) * 0.012) | 1
        if feather_size < 3:
            feather_size = 3
        feathered_mask = cv2.GaussianBlur(final_skin_mask, (feather_size, feather_size), 0) / 255.0
        
        return feathered_mask

    def apply_skin_smoothing(self, image, skin_mask, strength, texture_recovery=0.0, accumulators=None):
        d = int(5 + 10 * strength)
        sigma_color = int(10 + 110 * strength)
        sigma_space = int(10 + 110 * strength)
        
        smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        mask_3d = np.expand_dims(skin_mask, axis=2)
        blend_factor = mask_3d * strength * 0.92
        
        smoothed_output = (image * (1.0 - blend_factor) + smoothed * blend_factor)
        
        if texture_recovery > 0.001:
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            detail = image.astype(np.float32) - blurred.astype(np.float32)
            reinjected_detail = detail * blend_factor * texture_recovery
            final_output = smoothed_output.astype(np.float32) + reinjected_detail
            output = np.clip(final_output, 0, 255).astype(np.uint8)
            target_color = smoothed.astype(np.float32) + detail * texture_recovery
        else:
            output = smoothed_output.astype(np.uint8)
            target_color = smoothed.astype(np.float32)
            
        if accumulators is not None:
            accum_color, accum_alpha = accumulators
            blend_1d = skin_mask * strength * 0.92
            accum_color[:] = accum_color * (1.0 - blend_factor) + target_color * blend_factor
            accum_alpha[:] = accum_alpha * (1.0 - blend_1d) + blend_1d
            
        return output

    def apply_skin_brightening(self, image, skin_mask, strength, accumulators=None):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        
        lift = int(25 * strength)
        Y_float = Y.astype(np.float32) + lift * skin_mask
        Y_new = np.clip(Y_float, 0, 255).astype(np.uint8)
        
        brightened = cv2.cvtColor(cv2.merge([Y_new, Cr, Cb]), cv2.COLOR_YCrCb2BGR)
        
        if accumulators is not None:
            accum_color, accum_alpha = accumulators
            mask_3d = np.expand_dims(skin_mask, axis=2)
            accum_color[:] = accum_color * (1.0 - mask_3d) + brightened.astype(np.float32) * mask_3d
            accum_alpha[:] = accum_alpha * (1.0 - skin_mask) + skin_mask
            
        return brightened

    def apply_blush(self, image, coords, strength, accumulators=None):
        h, w = image.shape[:2]
        
        left_cheek = coords[117]
        right_cheek = coords[346]
        
        face_width = np.linalg.norm(coords[234] - coords[454])
        blush_radius = int(face_width * 0.16)
        if blush_radius < 5:
            blush_radius = 5
            
        cheek_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(cheek_mask, tuple(left_cheek), blush_radius, 1.0, -1)
        cv2.circle(cheek_mask, tuple(right_cheek), blush_radius, 1.0, -1)
        
        blur_ksize = int(blush_radius * 1.5) | 1
        cheek_mask = cv2.GaussianBlur(cheek_mask, (blur_ksize, blur_ksize), 0)
        
        blush_color = np.array([130, 120, 250], dtype=np.float32)
        
        cheek_mask_3d = np.expand_dims(cheek_mask, axis=2)
        blend_factor = cheek_mask_3d * strength * 0.35
        
        output = (image * (1.0 - blend_factor) + blush_color * blend_factor).astype(np.uint8)
        
        if accumulators is not None:
            accum_color, accum_alpha = accumulators
            blend_1d = cheek_mask * strength * 0.35
            accum_color[:] = accum_color * (1.0 - cheek_mask_3d) + blush_color * blend_factor
            accum_alpha[:] = accum_alpha * (1.0 - blend_1d) + blend_1d
            
        return output

    def apply_undereye_lighten(self, image, coords, raw_landmarks, strength, accumulators=None):
        h, w = image.shape[:2]
        
        left_eye_height = np.linalg.norm(coords[159] - coords[145])
        right_eye_height = np.linalg.norm(coords[386] - coords[374])
        avg_eye_height = (left_eye_height + right_eye_height) / 2.0
        
        offset_y = int(avg_eye_height * 1.4)
        
        left_eye_bottom = [133, 155, 154, 153, 145, 144, 163, 33]
        right_eye_bottom = [362, 382, 381, 380, 374, 373, 390, 263]
        
        undereye_mask = np.zeros((h, w), dtype=np.uint8)
        
        def draw_undereye_poly(bottom_indices):
            pts = coords[bottom_indices]
            shifted_pts = pts.copy()
            shifted_pts[:, 1] += offset_y
            poly_pts = np.vstack([pts, shifted_pts[::-1]])
            hull = cv2.convexHull(poly_pts)
            cv2.fillConvexPoly(undereye_mask, hull, 255)
            
        draw_undereye_poly(left_eye_bottom)
        draw_undereye_poly(right_eye_bottom)
        
        blur_size = int(avg_eye_height * 1.4) | 1
        if blur_size < 3:
            blur_size = 3
        undereye_mask_feathered = cv2.GaussianBlur(undereye_mask, (blur_size, blur_size), 0) / 255.0
        
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        
        Y_float = Y.astype(np.float32) + (25.0 * strength) * undereye_mask_feathered
        Cr_float = Cr.astype(np.float32) + (5.0 * strength) * undereye_mask_feathered
        Cb_float = Cb.astype(np.float32) - (10.0 * strength) * undereye_mask_feathered
        
        Y_new = np.clip(Y_float, 0, 255).astype(np.uint8)
        Cr_new = np.clip(Cr_float, 0, 255).astype(np.uint8)
        Cb_new = np.clip(Cb_float, 0, 255).astype(np.uint8)
        
        output = cv2.cvtColor(cv2.merge([Y_new, Cr_new, Cb_new]), cv2.COLOR_YCrCb2BGR)
        
        if accumulators is not None:
            accum_color, accum_alpha = accumulators
            mask_3d = np.expand_dims(undereye_mask_feathered, axis=2)
            accum_color[:] = accum_color * (1.0 - mask_3d) + output.astype(np.float32) * mask_3d
            accum_alpha[:] = accum_alpha * (1.0 - undereye_mask_feathered) + undereye_mask_feathered
            
        return output

    def apply_eye_enhancement(self, image, coords, strength, accumulators=None):
        h, w = image.shape[:2]
        
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        left_hull = cv2.convexHull(coords[LEFT_EYE_INDICES])
        right_hull = cv2.convexHull(coords[RIGHT_EYE_INDICES])
        cv2.fillConvexPoly(eye_mask, left_hull, 255)
        cv2.fillConvexPoly(eye_mask, right_hull, 255)
        
        kernel_size = int(max(w, h) * 0.005) | 1
        if kernel_size < 3:
            kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eye_mask = cv2.dilate(eye_mask, kernel)
        
        feather_ksize = kernel_size * 2 + 1
        eye_mask_feathered = cv2.GaussianBlur(eye_mask, (feather_ksize, feather_ksize), 0) / 255.0
        eye_mask_3d = np.expand_dims(eye_mask_feathered, axis=2)
        
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        
        clip_limit = 1.5 + 2.5 * strength
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        Y_clahe = clahe.apply(Y)
        
        Y_blend = (Y.astype(np.float32) * (1.0 - eye_mask_feathered) + Y_clahe.astype(np.float32) * eye_mask_feathered).astype(np.uint8)
        contrast_enhanced = cv2.cvtColor(cv2.merge([Y_blend, Cr, Cb]), cv2.COLOR_YCrCb2BGR)
        
        blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(contrast_enhanced, 1.0 + strength * 0.8, blurred, -strength * 0.8, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        output = (contrast_enhanced * (1.0 - eye_mask_3d) + sharpened * eye_mask_3d).astype(np.uint8)
        
        if accumulators is not None:
            accum_color, accum_alpha = accumulators
            accum_color[:] = accum_color * (1.0 - eye_mask_3d) + output.astype(np.float32) * eye_mask_3d
            accum_alpha[:] = accum_alpha * (1.0 - eye_mask_feathered) + eye_mask_feathered
            
        return output

    # ==========================================
    # COLOR LOOKS AND TINTS ENGINE
    # ==========================================
    def apply_color_filter(self, image, filter_name, intensity):
        """
        Apply a color filter style (Warm Sunset, Cool Ice, Sepia, Teal & Orange, Cinematic Mono)
        and blend with the original image based on intensity (0.0 to 1.0).
        """
        if intensity <= 0.001 or filter_name == "None":
            return image
            
        filtered = image.copy()
        
        if filter_name == "Warm Sunset":
            # Warm Sunset look
            b, g, r = cv2.split(image.astype(np.float32))
            r = r * 1.14 + 8
            g = g * 1.04 + 3
            b = b * 0.88
            filtered = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
            
        elif filter_name == "Cool Ice":
            # Cool Ice look
            b, g, r = cv2.split(image.astype(np.float32))
            r = r * 0.86
            g = g * 1.02 + 4
            b = b * 1.16 + 10
            filtered = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
            
        elif filter_name == "Vintage Sepia":
            # Standard sepia matrix transformation
            sepia_matrix = np.array([
                [0.272, 0.534, 0.131],  # Blue channel coefficients
                [0.349, 0.686, 0.168],  # Green channel coefficients
                [0.393, 0.769, 0.189]   # Red channel coefficients
            ])
            filtered = cv2.transform(image, sepia_matrix)
            filtered = np.clip(filtered, 0, 255).astype(np.uint8)
            
        elif filter_name == "Teal & Orange":
            # Cinematic look: warm highlights, cool shadows
            img = image.astype(np.float32)
            b, g, r = cv2.split(img)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # Apply color adjustments based on luminance
            r_new = r + 15 * gray - 12 * (1.0 - gray)
            g_new = g + 4 * gray
            b_new = b - 15 * gray + 15 * (1.0 - gray)
            filtered = np.clip(cv2.merge([b_new, g_new, r_new]), 0, 255).astype(np.uint8)
            
        elif filter_name == "Cinematic Mono":
            # Deep monochrome with CLAHE contrast boost
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            gray_contrast = clahe.apply(gray)
            filtered = cv2.merge([gray_contrast, gray_contrast, gray_contrast])
            
        # Blend look with original image based on slider intensity
        output = cv2.addWeighted(filtered, intensity, image, 1.0 - intensity, 0)
        return output

    # ==========================================
    # FACE RESHAPING WARP ENGINE (Snapchat-style)
    # ==========================================
    def apply_reshape_warps(self, image, faces_coords, nose_reduce, cheeks_reduce, forehead_reduce, eye_enlarge=0.0, lips_plump=0.0):
        """
        Applies plastic warps on the coordinate space of the image using cv2.remap.
        Optimized by combining all deformations into a single interpolation pass.
        """
        h, w = image.shape[:2]
        
        # Initialize standard pixel grid coordinates
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Track if anything is modified to avoid remapping if not needed
        is_warped = False
        
        for coords in faces_coords:
            # Scale reference sizes based on face dimensions
            face_width = np.linalg.norm(coords[234] - coords[454])
            if face_width < 10:
                continue
                
            # 1. NOSE SIZE REDUCTION (Horizontal-dominant pinch at nose tip)
            if nose_reduce > 0.001:
                nose_tip = coords[4]
                # Nose width based on nose edges (landmarks 102 and 331)
                nose_width = np.linalg.norm(coords[102] - coords[331])
                R_nose = max(nose_width * 1.4, 15)
                
                dx = map_x - nose_tip[0]
                dy = map_y - nose_tip[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                mask = dist < R_nose
                if np.any(mask):
                    t = dist / R_nose
                    # Fetch coordinate displacement from further out to pinch center
                    # We only deform horizontally to slim the nose (avoid squishing it vertically)
                    factor = 1.0 + nose_reduce * 0.35 * (1.0 - t)**2
                    map_x[mask] = nose_tip[0] + dx[mask] * factor[mask]
                    is_warped = True
                    
            # 2. CHEEKS REDUCTION (Jawline slimming - dual pinch warp)
            if cheeks_reduce > 0.001:
                # Left jaw center (landmark 172) and Right jaw center (landmark 397)
                left_jaw = coords[172]
                right_jaw = coords[397]
                
                R_cheek = face_width * 0.35
                max_shift = face_width * 0.08 * cheeks_reduce
                
                # A. Left jaw warp (pull cheeks right towards center line)
                dx_l = map_x - left_jaw[0]
                dy_l = map_y - left_jaw[1]
                dist_l = np.sqrt(dx_l*dx_l + dy_l*dy_l)
                mask_l = dist_l < R_cheek
                if np.any(mask_l):
                    t_l = dist_l / R_cheek
                    shift_l = max_shift * (1.0 - t_l)**2
                    # Fetch from further left (subtract shift) to move image right
                    map_x[mask_l] -= shift_l[mask_l]
                    is_warped = True
                    
                # B. Right jaw warp (pull cheeks left towards center line)
                dx_r = map_x - right_jaw[0]
                dy_r = map_y - right_jaw[1]
                dist_r = np.sqrt(dx_r*dx_r + dy_r*dy_r)
                mask_r = dist_r < R_cheek
                if np.any(mask_r):
                    t_r = dist_r / R_cheek
                    shift_r = max_shift * (1.0 - t_r)**2
                    # Fetch from further right (add shift) to move image left
                    map_x[mask_r] += shift_r[mask_r]
                    is_warped = True
                    
            # 3. FOREHEAD REDUCTION (Downward vertical push of hairline)
            if forehead_reduce > 0.001:
                # Forehead top (landmark 10)
                forehead_top = coords[10]
                R_forehead = face_width * 0.45
                max_shift_y = face_width * 0.07 * forehead_reduce
                
                dx_f = map_x - forehead_top[0]
                dy_f = map_y - forehead_top[1]
                dist_f = np.sqrt(dx_f*dx_f + dy_f*dy_f)
                mask_f = dist_f < R_forehead
                if np.any(mask_f):
                    t_f = dist_f / R_forehead
                    shift_y = max_shift_y * (1.0 - t_f)**2
                    # Fetch from higher up (subtract from map_y) to shift forehead down
                    map_y[mask_f] -= shift_y[mask_f]
                    is_warped = True
                    
            # 4. EYE SIZE ENLARGEMENT (Bulge warp centered at left/right irises)
            if eye_enlarge > 0.001:
                # Left/Right irises centers (average of indices 474-477 / 469-472)
                left_eye_center = np.mean(coords[[474, 475, 476, 477]], axis=0).astype(int)
                right_eye_center = np.mean(coords[[469, 470, 471, 472]], axis=0).astype(int)
                
                eye_w = np.linalg.norm(coords[33] - coords[133])
                R_eye = max(eye_w * 1.35, 15)
                
                def bulge_eye(center):
                    nonlocal map_x, map_y, is_warped
                    dx = map_x - center[0]
                    dy = map_y - center[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    mask = dist < R_eye
                    if np.any(mask):
                        t = dist / R_eye
                        # Fetch from closer to center (multiplier < 1.0) to push outward
                        factor = 1.0 - eye_enlarge * 0.22 * (1.0 - t)**2
                        map_x[mask] = center[0] + dx[mask] * factor[mask]
                        map_y[mask] = center[1] + dy[mask] * factor[mask]
                        is_warped = True
                        
                bulge_eye(left_eye_center)
                bulge_eye(right_eye_center)
                
            # 5. LIP SIZE ENLARGEMENT (Plump lips - bulge warp centered at mouth)
            if lips_plump > 0.001:
                lips_center = np.mean(coords[LIPS_INDICES], axis=0).astype(int)
                lips_width = np.linalg.norm(coords[61] - coords[291])
                R_lips = max(lips_width * 0.75, 20)
                
                dx = map_x - lips_center[0]
                dy = map_y - lips_center[1]
                dist = np.sqrt(dx*dx + dy*dy)
                mask = dist < R_lips
                if np.any(mask):
                    t = dist / R_lips
                    factor = 1.0 - lips_plump * 0.18 * (1.0 - t)**2
                    map_x[mask] = lips_center[0] + dx[mask] * factor[mask]
                    map_y[mask] = lips_center[1] + dy[mask] * factor[mask]
                    is_warped = True
                    
        if is_warped:
            # Perform single interpolation pass to deform face
            warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return warped
            
        return image

    # ==========================================
    # MAKEUP TINTS ENGINE (Lips & Eyes Color)
    # ==========================================
    def apply_lipstick(self, image, coords, strength, color_shade, accumulators=None):
        h, w = image.shape[:2]
        
        # Lips mask
        lips_mask = np.zeros((h, w), dtype=np.uint8)
        pts = coords[LIPS_INDICES]
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(lips_mask, hull, 255)
        
        # Smooth boundaries
        lips_mask = cv2.GaussianBlur(lips_mask, (7, 7), 0) / 255.0
        lips_mask_3d = np.expand_dims(lips_mask, axis=2)
        
        color_table = {
            "Rose Red": np.array([80, 50, 220], dtype=np.float32),
            "Soft Pink": np.array([150, 100, 240], dtype=np.float32),
            "Peach Glow": np.array([100, 130, 240], dtype=np.float32),
            "Plum Berry": np.array([80, 30, 140], dtype=np.float32)
        }
        lipstick_color = color_table.get(color_shade, np.array([0, 0, 255], dtype=np.float32))
        
        # Blend lipstick color (max opacity 55% at strength 1.0)
        blend_factor = lips_mask_3d * strength * 0.55
        output = (image * (1.0 - blend_factor) + lipstick_color * blend_factor).astype(np.uint8)
        
        if accumulators is not None:
            accum_color, accum_alpha = accumulators
            blend_1d = lips_mask * strength * 0.55
            accum_color[:] = accum_color * (1.0 - blend_factor) + lipstick_color * blend_factor
            accum_alpha[:] = accum_alpha * (1.0 - blend_1d) + blend_1d
            
        return output

    def apply_eye_color(self, image, coords, strength, color_shade, accumulators=None):
        h, w = image.shape[:2]
        
        # Irises landmarks: left 474-477, right 469-472
        left_iris_pts = coords[[474, 475, 476, 477]]
        right_iris_pts = coords[[469, 470, 471, 472]]
        
        left_center = np.mean(left_iris_pts, axis=0).astype(int)
        left_r = int(np.linalg.norm(left_iris_pts[0] - left_center))
        
        right_center = np.mean(right_iris_pts, axis=0).astype(int)
        right_r = int(np.linalg.norm(right_iris_pts[0] - right_center))
        
        iris_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(iris_mask, tuple(left_center), left_r, 1.0, -1)
        cv2.circle(iris_mask, tuple(right_center), right_r, 1.0, -1)
        
        iris_mask = cv2.GaussianBlur(iris_mask, (3, 3), 0)
        iris_mask_3d = np.expand_dims(iris_mask, axis=2)
        
        color_table = {
            "Ocean Blue": np.array([220, 150, 40], dtype=np.float32),
            "Emerald Green": np.array([60, 170, 50], dtype=np.float32),
            "Honey Brown": np.array([40, 90, 160], dtype=np.float32),
            "Deep Amber": np.array([30, 130, 200], dtype=np.float32)
        }
        iris_color = color_table.get(color_shade, np.array([255, 0, 0], dtype=np.float32))
        
        # Blend contacts color (max opacity 45% at strength 1.0)
        blend_factor = iris_mask_3d * strength * 0.45
        output = (image * (1.0 - blend_factor) + iris_color * blend_factor).astype(np.uint8)
        
        if accumulators is not None:
            accum_color, accum_alpha = accumulators
            blend_1d = iris_mask * strength * 0.45
            accum_color[:] = accum_color * (1.0 - blend_factor) + iris_color * blend_factor
            accum_alpha[:] = accum_alpha * (1.0 - blend_1d) + blend_1d
            
        return output

    def generate_cube_lut(self, filter_name, intensity, filepath):
        """
        Generates a standard 33x33x33 3D LUT from a color look filter
        and writes it as a .cube file.
        """
        lut_size = 33
        lut_grid = np.zeros((1, lut_size * lut_size * lut_size, 3), dtype=np.uint8)
        
        idx = 0
        for r_idx in range(lut_size):
            r_val = int(r_idx * 255.0 / (lut_size - 1) + 0.5)
            for g_idx in range(lut_size):
                g_val = int(g_idx * 255.0 / (lut_size - 1) + 0.5)
                for b_idx in range(lut_size):
                    b_val = int(b_idx * 255.0 / (lut_size - 1) + 0.5)
                    lut_grid[0, idx] = [b_val, g_val, r_val]
                    idx += 1
                    
        # Apply the color lookup filter
        lut_processed = self.apply_color_filter(lut_grid, filter_name, intensity)
        
        # Write .cube format
        with open(filepath, "w") as f:
            f.write("# Created by DaVinci Resolve Beauty Companion\n")
            f.write(f"LUT_3D_SIZE {lut_size}\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            
            for idx in range(lut_size * lut_size * lut_size):
                b, g, r = lut_processed[0, idx]
                r_f = r / 255.0
                g_f = g / 255.0
                b_f = b / 255.0
                f.write(f"{r_f:.6f} {g_f:.6f} {b_f:.6f}\n")
