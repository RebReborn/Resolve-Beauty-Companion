# Resolve Beauty Companion

A high-performance, real-time cross-platform desktop application designed as a companion utility to DaVinci Resolve. It offers Snapchat-style face smoothing and beauty filters to enhance video clips. You can import video clips, adjust filters with visual feedback, and export the processed clips using industry-standard codecs (ProRes, DNxHD, H.264) for seamless re-import back into DaVinci Resolve.

---

## Features

1. **Draggable Before/After Split Viewer**:
   - Compare the original and filtered frames in real-time.
   - Drag the split line slider left and right to inspect details.
   - Switch between Split Screen, After (Filtered only), and Before (Original only) comparison modes.

2. **Precision Face Mesh & Skin Masking**:
   - Uses modern **MediaPipe Face Landmarker** to track 478 points (including eyes and cheeks).
   - Generates a dynamic skin-tone mask in the YCrCb color space.
   - Excludes eyes, eyebrows, lips, and nostrils to avoid unwanted blurring.
   - Feathers the mask edges softly to ensure seamless blending.

3. **High-Quality Beauty Filters (Adjustable Sliders)**:
   - **Skin Smoothing (0-100%)**: Edge-preserving bilateral filter applied only to skin regions, blended with the original texture to preserve detail.
   - **Skin Brightening (0-100%)**: Non-destructive luminance boost in YCrCb space.
   - **Blush / Warmth (0-100%)**: Subtle peach/pink tint applied to cheek landmarks (117 & 346) with feathered circular overlays.
   - **Under-eye Lighten (0-100%)**: Targeted brightening of dark circles below the eyes, neutralizing bluish tones.
   - **Eye Enhancement (0-100%)**: Boosts iris details, whites, and eyelash lines using local contrast CLAHE and unsharp masking.

4. **Timeline Scrubbing & Playback**:
   - Play/Pause video or scrub the timeline using the progress slider.
   - Source FPS and resolution details displayed natively.

5. **Presets and Codecs**:
   - Save/Load custom presets as JSON.
   - Built-in presets: *Natural Glow*, *Hollywood Smooth*, *High Glamour*, *Subtle Polish*.
   - Export profiles: ProRes (Standard/HQ), DNxHD, H.264 (MP4/MOV) with automatic fallback to standard H.264 if professional codecs fail to initialize.

---

## Installation & Setup

1. **Verify Python Installation**:
   Make sure you have Python 3.10+ installed.

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux (Terminal)**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**:
   ```bash
   python main.py
   ```

---

## Usage Guide

1. **Load Source Media**:
   - Click **Select Video File** to choose an exported clip from DaVinci Resolve.
   - Alternatively, click **Start Webcam Preview** to test filters live on your webcam.

2. **Adjust Filters**:
   - Modify the sliders to tune details.
   - Double-click any slider to reset it to its default value ($0\%$).
   - Choose a predefined starting point from the **Filter Presets** dropdown.

3. **Compare Quality**:
   - Drag the split line on the screen.
   - Toggle comparison modes with the top header buttons.

4. **Export**:
   - Choose your output profile from **Output Codec Profile**.
   - Click **Export Video** and select the destination.
   - The progress bar shows completion status, speed, and ETA.
   - Drag the exported clip directly back into DaVinci Resolve.

---

## Keyboard Shortcuts

- `Space`: Play/Pause video preview or freeze webcam.
- `Enter`: Open export video dialog.
- `Ctrl + Q`: Quit the application.
