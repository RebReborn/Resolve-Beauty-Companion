from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QComboBox, QFileDialog, QLabel, 
                             QProgressBar, QGroupBox, QMessageBox, QSlider, QScrollArea,
                             QCheckBox)
from PyQt6.QtCore import Qt, QTimer, QSize, QPointF
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QImage, QPainter, QPixmap, QColor, QPen, QBrush, QLinearGradient, QPolygonF
import os
import cv2
import numpy as np

from ui.widgets import PrecisionSlider, BeforeAfterViewer
from core.filters import BeautyFilterEngine
from core.processor import VideoProcessorThread, WebcamThread, VideoReader, CODEC_MAP
from utils.presets import DEFAULT_PRESETS, save_preset_to_file, load_preset_from_file
from core.batch_manager import BatchQueueProcessor
from ui.batch_dialog import BatchManagerDialog

class MainWindow(QMainWindow):
    def __init__(self, clip_path=None):
        super().__init__()
        self.setWindowTitle("Resolve Beauty Companion")
        self.setMinimumSize(QSize(1200, 800))
        
        # Core engines
        self.filter_engine = BeautyFilterEngine()
        self.video_reader = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._on_play_step)
        
        # Active workers
        self.webcam_thread = None
        self.export_thread = None
        
        # State
        self.current_frame_idx = 0
        self.is_playing = False
        self.active_frame = None  # Original BGR numpy array
        
        # Batch queue system
        self.batch_processor = BatchQueueProcessor()
        self.batch_dialog = BatchManagerDialog(self.batch_processor, self, self)
        
        self.init_ui()
        self.apply_theme()
        self.load_default_presets()
        self.setup_shortcuts()
        self.create_app_icon()
        self.setup_menu_bar()
        
        # Auto-load clip if provided
        if clip_path:
            QTimer.singleShot(100, lambda: self.load_video_from_path(clip_path))
        
    def init_ui(self):
        # Main central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # ==========================================
        # LEFT PANEL (Control Sidebar in Scroll Area)
        # ==========================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(340)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #121212;
            }
            QScrollBar:vertical {
                border: none;
                background: #1a1a1a;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3d3d3d;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #ff9f1c;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        sidebar = QWidget()
        sidebar.setStyleSheet("background-color: #121212;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(12)
        
        # --- Group 1: Presets ---
        preset_group = QGroupBox("Filter Presets")
        preset_layout = QVBoxLayout(preset_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px;
                font-weight: 500;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1a;
                color: #e0e0e0;
                selection-background-color: #ff9f1c;
                selection-color: #121212;
            }
        """)
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(self.preset_combo)
        
        preset_btn_layout = QHBoxLayout()
        self.save_preset_btn = QPushButton("Save Preset")
        self.save_preset_btn.clicked.connect(self._on_save_preset)
        self.load_preset_btn = QPushButton("Load Preset File")
        self.load_preset_btn.clicked.connect(self._on_load_preset)
        
        preset_btn_layout.addWidget(self.save_preset_btn)
        preset_btn_layout.addWidget(self.load_preset_btn)
        preset_layout.addLayout(preset_btn_layout)
        sidebar_layout.addWidget(preset_group)
        
        # --- Group 1.5: Cinematic Looks ---
        looks_group = QGroupBox("Cinematic Looks")
        looks_layout = QVBoxLayout(looks_group)
        
        self.look_combo = QComboBox()
        self.look_combo.addItems(["None", "Warm Sunset", "Cool Ice", "Vintage Sepia", "Teal & Orange", "Cinematic Mono"])
        self.look_combo.setStyleSheet(self.preset_combo.styleSheet())
        self.look_combo.currentTextChanged.connect(self._on_slider_changed)
        
        self.slider_look_intensity = PrecisionSlider("Look Intensity", default_val=100)
        self.slider_look_intensity.valueChanged.connect(self._on_slider_changed)
        
        looks_layout.addWidget(self.look_combo)
        looks_layout.addWidget(self.slider_look_intensity)
        sidebar_layout.addWidget(looks_group)
        
        # --- Group 1.6: Virtual Cosmetics Suite ---
        makeup_group = QGroupBox("Virtual Cosmetics Suite")
        makeup_layout = QVBoxLayout(makeup_group)
        makeup_layout.setSpacing(6)
        
        lipstick_label = QLabel("Lipstick Shade:")
        lipstick_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        makeup_layout.addWidget(lipstick_label)
        
        self.lipstick_combo = QComboBox()
        self.lipstick_combo.addItems(["None", "Rose Red", "Soft Pink", "Peach Glow", "Plum Berry"])
        self.lipstick_combo.setStyleSheet(self.preset_combo.styleSheet())
        self.lipstick_combo.currentTextChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.lipstick_combo)
        
        self.slider_lipstick_strength = PrecisionSlider("Lipstick Strength")
        self.slider_lipstick_strength.valueChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.slider_lipstick_strength)

        self.slider_lip_gloss_strength = PrecisionSlider("Lip Gloss (Specular)")
        self.slider_lip_gloss_strength.valueChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.slider_lip_gloss_strength)
        
        contacts_label = QLabel("Eye Color Shade:")
        contacts_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        makeup_layout.addWidget(contacts_label)
        
        self.contacts_combo = QComboBox()
        self.contacts_combo.addItems(["Natural", "Ocean Blue", "Emerald Green", "Honey Brown", "Deep Amber"])
        self.contacts_combo.setStyleSheet(self.preset_combo.styleSheet())
        self.contacts_combo.currentTextChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.contacts_combo)
        
        self.slider_contacts_strength = PrecisionSlider("Eye Color Strength")
        self.slider_contacts_strength.valueChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.slider_contacts_strength)

        self.slider_eyeliner_strength = PrecisionSlider("Eyeliner & Mascara")
        self.slider_eyeliner_strength.valueChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.slider_eyeliner_strength)

        eyeshadow_label = QLabel("Eyeshadow Shade:")
        eyeshadow_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        makeup_layout.addWidget(eyeshadow_label)

        self.eyeshadow_combo = QComboBox()
        self.eyeshadow_combo.addItems(["None", "Royal Purple", "Rose Gold", "Sunset Bronze", "Ocean Blue"])
        self.eyeshadow_combo.setStyleSheet(self.preset_combo.styleSheet())
        self.eyeshadow_combo.currentTextChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.eyeshadow_combo)

        self.slider_eyeshadow_strength = PrecisionSlider("Eyeshadow Strength")
        self.slider_eyeshadow_strength.valueChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.slider_eyeshadow_strength)

        self.slider_facial_highlighter_strength = PrecisionSlider("Facial Highlighter")
        self.slider_facial_highlighter_strength.valueChanged.connect(self._on_slider_changed)
        makeup_layout.addWidget(self.slider_facial_highlighter_strength)
        
        sidebar_layout.addWidget(makeup_group)
        
        # --- Group 2: Filter Sliders ---
        sliders_group = QGroupBox("Beauty Adjustment Controls")
        sliders_layout = QVBoxLayout(sliders_group)
        sliders_layout.setSpacing(10)
        
        # Sliders declarations
        self.slider_smoothing = PrecisionSlider("Skin Smoothing")
        self.slider_texture_recovery = PrecisionSlider("Skin Texture Recovery", default_val=30)
        self.slider_brightening = PrecisionSlider("Skin Brightening")
        self.slider_blush = PrecisionSlider("Blush / Warmth")
        self.slider_eye = PrecisionSlider("Eye Enhancement")
        self.slider_undereye = PrecisionSlider("Under-eye Lighten")
        
        # Add to layout
        sliders_layout.addWidget(self.slider_smoothing)
        sliders_layout.addWidget(self.slider_texture_recovery)
        sliders_layout.addWidget(self.slider_brightening)
        sliders_layout.addWidget(self.slider_blush)
        sliders_layout.addWidget(self.slider_undereye)
        sliders_layout.addWidget(self.slider_eye)
        
        # Connect change signals to update preview
        self.slider_smoothing.valueChanged.connect(self._on_slider_changed)
        self.slider_texture_recovery.valueChanged.connect(self._on_slider_changed)
        self.slider_brightening.valueChanged.connect(self._on_slider_changed)
        self.slider_blush.valueChanged.connect(self._on_slider_changed)
        self.slider_eye.valueChanged.connect(self._on_slider_changed)
        self.slider_undereye.valueChanged.connect(self._on_slider_changed)
        
        sidebar_layout.addWidget(sliders_group)
        
        # --- Group 2.2: Body & Neck Retouching ---
        body_group = QGroupBox("Body & Neck Skin Retouching")
        body_layout = QVBoxLayout(body_group)
        body_layout.setSpacing(10)
        
        self.body_retouching_checkbox = QCheckBox("Enable Body & Neck Retouching")
        self.body_retouching_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e0e0e0;
                font-size: 11px;
                margin-top: 5px;
                margin-bottom: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
            }
            QCheckBox::indicator:hover {
                border-color: #ff9f1c;
            }
        """)
        self.body_retouching_checkbox.stateChanged.connect(self._on_slider_changed)
        body_layout.addWidget(self.body_retouching_checkbox)
        
        self.slider_body_sensitivity = PrecisionSlider("Body Skin Sensitivity", default_val=50)
        self.slider_body_sensitivity.valueChanged.connect(self._on_slider_changed)
        body_layout.addWidget(self.slider_body_sensitivity)
        
        sidebar_layout.addWidget(body_group)
        
        # --- Group 2.5: Face Reshaping Controls ---
        reshaping_group = QGroupBox("Face Reshaping (Snapchat-style)")
        reshaping_layout = QVBoxLayout(reshaping_group)
        reshaping_layout.setSpacing(10)
        
        self.slider_nose = PrecisionSlider("Nose Size Reduce")
        self.slider_cheeks = PrecisionSlider("Cheek Slimming")
        self.slider_forehead = PrecisionSlider("Forehead Reduce")
        self.slider_eye_size = PrecisionSlider("Eye Size (Enlarge)")
        self.slider_lip_size = PrecisionSlider("Lip Size (Plump)")
        
        self.slider_nose.valueChanged.connect(self._on_slider_changed)
        self.slider_cheeks.valueChanged.connect(self._on_slider_changed)
        self.slider_forehead.valueChanged.connect(self._on_slider_changed)
        self.slider_eye_size.valueChanged.connect(self._on_slider_changed)
        self.slider_lip_size.valueChanged.connect(self._on_slider_changed)
        
        reshaping_layout.addWidget(self.slider_nose)
        reshaping_layout.addWidget(self.slider_cheeks)
        reshaping_layout.addWidget(self.slider_forehead)
        reshaping_layout.addWidget(self.slider_eye_size)
        reshaping_layout.addWidget(self.slider_lip_size)
        sidebar_layout.addWidget(reshaping_group)
        
        # --- Group 3: Media Operations ---
        media_group = QGroupBox("Source Selection")
        media_layout = QVBoxLayout(media_group)
        
        self.open_video_btn = QPushButton("Select Video File")
        self.open_video_btn.setIconSize(QSize(16, 16))
        self.open_video_btn.clicked.connect(self._on_open_video)
        
        self.sync_resolve_btn = QPushButton("Sync Clip from Resolve")
        self.sync_resolve_btn.clicked.connect(self._on_sync_resolve)
        
        self.webcam_btn = QPushButton("Start Webcam Preview")
        self.webcam_btn.clicked.connect(self._on_toggle_webcam)
        
        media_layout.addWidget(self.open_video_btn)
        media_layout.addWidget(self.sync_resolve_btn)
        media_layout.addWidget(self.webcam_btn)
        sidebar_layout.addWidget(media_group)
        
        # --- Group 4: Export Panel ---
        export_group = QGroupBox("Resolve Compatibility Export")
        export_layout = QVBoxLayout(export_group)
        
        codec_label = QLabel("Output Codec Profile:")
        codec_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        export_layout.addWidget(codec_label)
        
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(list(CODEC_MAP.keys()))
        self.codec_combo.setStyleSheet(self.preset_combo.styleSheet())
        self.codec_combo.currentTextChanged.connect(self._on_codec_changed)
        export_layout.addWidget(self.codec_combo)
        
        self.export_alpha_checkbox = QCheckBox("Export Alpha Overlay Only")
        self.export_alpha_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e0e0e0;
                font-size: 11px;
                margin-top: 5px;
                margin-bottom: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
            }
            QCheckBox::indicator:hover {
                border-color: #ff9f1c;
            }
        """)
        self.export_alpha_checkbox.stateChanged.connect(self._on_export_alpha_toggled)
        export_layout.addWidget(self.export_alpha_checkbox)
        
        self.gpu_checkbox = QCheckBox("Enable GPU Acceleration (DirectML ONNX)")
        self.gpu_checkbox.setStyleSheet(self.export_alpha_checkbox.styleSheet())
        self.gpu_checkbox.stateChanged.connect(self._on_slider_changed)
        export_layout.addWidget(self.gpu_checkbox)
        
        self.export_btn = QPushButton("Export Video")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9f1c; /* Resolve Amber */
                color: #121212;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #ffa834;
            }
            QPushButton:pressed {
                background-color: #e68e14;
            }
            QPushButton:disabled {
                background-color: #2b2b2b;
                color: #5d5d5d;
            }
        """)
        self.export_btn.clicked.connect(self._on_export_video)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        batch_btn_layout = QHBoxLayout()
        self.add_to_batch_btn = QPushButton("Add to Batch Queue")
        self.add_to_batch_btn.setEnabled(False)
        self.add_to_batch_btn.clicked.connect(self._on_add_to_batch_queue)
        
        self.show_queue_btn = QPushButton("Show Queue Manager")
        self.show_queue_btn.clicked.connect(self._on_show_batch_dialog)
        
        batch_btn_layout.addWidget(self.add_to_batch_btn)
        batch_btn_layout.addWidget(self.show_queue_btn)
        export_layout.addLayout(batch_btn_layout)
        
        sidebar_layout.addWidget(export_group)
        sidebar_layout.addStretch()
        
        scroll_area.setWidget(sidebar)
        main_layout.addWidget(scroll_area)
        
        # ==========================================
        # RIGHT PANEL (Viewer & Timeline Transport)
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)
        
        # --- Viewer Comparison Modes Header ---
        compare_header = QHBoxLayout()
        compare_header.setSpacing(10)
        
        mode_label = QLabel("Comparison Mode:")
        mode_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        compare_header.addWidget(mode_label)
        
        self.btn_mode_split = QPushButton("Split Screen")
        self.btn_mode_split.setCheckable(True)
        self.btn_mode_split.setChecked(True)
        self.btn_mode_split.clicked.connect(lambda: self._set_compare_mode("split"))
        
        self.btn_mode_filtered = QPushButton("After (Filtered)")
        self.btn_mode_filtered.setCheckable(True)
        self.btn_mode_filtered.clicked.connect(lambda: self._set_compare_mode("processed"))
        
        self.btn_mode_original = QPushButton("Before (Original)")
        self.btn_mode_original.setCheckable(True)
        self.btn_mode_original.clicked.connect(lambda: self._set_compare_mode("original"))
        
        compare_header.addWidget(self.btn_mode_split)
        compare_header.addWidget(self.btn_mode_filtered)
        compare_header.addWidget(self.btn_mode_original)
        compare_header.addStretch()
        
        right_layout.addLayout(compare_header)
        
        # --- Viewer ---
        self.viewer = BeforeAfterViewer()
        right_layout.addWidget(self.viewer, stretch=1)
        
        # --- Transport Bar ---
        transport_widget = QWidget()
        transport_widget.setStyleSheet("background-color: #1e1e1e; border-radius: 6px;")
        transport_layout = QVBoxLayout(transport_widget)
        transport_layout.setContentsMargins(10, 10, 10, 10)
        
        scrub_layout = QHBoxLayout()
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setStyleSheet("color: #a0a0a0; font-family: monospace; font-size: 11px;")
        
        self.scrub_bar = QSlider(Qt.Orientation.Horizontal)
        self.scrub_bar.setRange(0, 100)
        self.scrub_bar.setValue(0)
        self.scrub_bar.setEnabled(False)
        self.scrub_bar.sliderMoved.connect(self._on_scrub_moved)
        self.scrub_bar.sliderPressed.connect(self._on_scrub_pressed)
        
        scrub_layout.addWidget(self.scrub_bar)
        scrub_layout.addWidget(self.time_label)
        transport_layout.addLayout(scrub_layout)
        
        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(80)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._on_toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        self.fps_label = QLabel("Source FPS: -")
        self.fps_label.setStyleSheet("color: #a0a0a0; font-size: 12px;")
        controls_layout.addStretch()
        controls_layout.addWidget(self.fps_label)
        controls_layout.addStretch()
        
        transport_layout.addLayout(controls_layout)
        right_layout.addWidget(transport_widget)
        
        # --- Export Progress ---
        self.progress_panel = QWidget()
        self.progress_panel.setVisible(False)
        progress_layout = QHBoxLayout(self.progress_panel)
        progress_layout.setContentsMargins(5, 5, 5, 5)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                text-align: center;
                color: #fff;
                font-weight: bold;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #ff9f1c;
                border-radius: 3px;
            }
        """)
        self.eta_label = QLabel("ETA: Calculating...")
        self.eta_label.setStyleSheet("color: #ff9f1c; font-weight: 500;")
        
        progress_layout.addWidget(self.progress_bar, stretch=1)
        progress_layout.addWidget(self.eta_label)
        right_layout.addWidget(self.progress_panel)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        # StatusBar
        self.statusBar().showMessage("Ready")
        
    def apply_theme(self):
        # Complete charcoal dark style matching Resolve color palettes
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QGroupBox {
                border: 1px solid #2d2d2d;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #ff9f1c;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
            }
            QPushButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #383838;
                border-color: #ff9f1c;
            }
            QPushButton:pressed {
                background-color: #1f1f1f;
            }
            QPushButton:checked {
                background-color: #ff9f1c;
                color: #121212;
                border-color: #ff9f1c;
                font-weight: bold;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QStatusBar {
                background-color: #1a1a1a;
                color: #8c8c8c;
                font-size: 11px;
                border-top: 1px solid #2a2a2a;
            }
        """)
        
    def setup_shortcuts(self):
        # Space -> toggle preview play/pause
        self.action_play = QAction(self)
        self.action_play.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.action_play.triggered.connect(self._on_toggle_play)
        self.addAction(self.action_play)
        
        # Enter -> Trigger process export
        self.action_process = QAction(self)
        self.action_process.setShortcut(QKeySequence(Qt.Key.Key_Return))
        self.action_process.triggered.connect(self._on_export_video)
        self.addAction(self.action_process)
        
        # Ctrl+Q -> Quit app
        self.action_quit = QAction(self)
        self.action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        self.action_quit.triggered.connect(self.close)
        self.addAction(self.action_quit)
        
    def load_default_presets(self):
        for name in DEFAULT_PRESETS.keys():
            self.preset_combo.addItem(name)
        # Select first preset "Natural Glow"
        self.preset_combo.setCurrentText("Natural Glow")
        self._on_preset_selected("Natural Glow")
        
    def get_slider_params(self):
        return {
            "skin_smoothing": self.slider_smoothing.value(),
            "skin_texture_recovery": self.slider_texture_recovery.value(),
            "blush_warmth": self.slider_blush.value(),
            "skin_brightening": self.slider_brightening.value(),
            "eye_enhancement": self.slider_eye.value(),
            "undereye_lighten": self.slider_undereye.value(),
            "nose_reduce": self.slider_nose.value(),
            "cheeks_reduce": self.slider_cheeks.value(),
            "forehead_reduce": self.slider_forehead.value(),
            "eye_enlarge": self.slider_eye_size.value(),
            "lips_plump": self.slider_lip_size.value(),
            "lipstick_shade": self.lipstick_combo.currentText(),
            "lipstick_strength": self.slider_lipstick_strength.value(),
            "eye_color_shade": self.contacts_combo.currentText(),
            "eye_color_strength": self.slider_contacts_strength.value(),
            "color_look": self.look_combo.currentText(),
            "look_intensity": self.slider_look_intensity.value(),
            "eyeliner_strength": self.slider_eyeliner_strength.value(),
            "eyeshadow_shade": self.eyeshadow_combo.currentText(),
            "eyeshadow_strength": self.slider_eyeshadow_strength.value(),
            "lip_gloss_strength": self.slider_lip_gloss_strength.value(),
            "facial_highlighter_strength": self.slider_facial_highlighter_strength.value(),
            "gpu_acceleration": self.gpu_checkbox.isChecked(),
            "enable_body_retouching": self.body_retouching_checkbox.isChecked(),
            "body_sensitivity": 0.5 + 2.0 * (self.slider_body_sensitivity.value() / 100.0)
        }
        
    def update_sliders_ui(self, params):
        # Block signals temporarily to prevent loop updates
        self.slider_smoothing.blockSignals(True)
        self.slider_texture_recovery.blockSignals(True)
        self.slider_brightening.blockSignals(True)
        self.slider_blush.blockSignals(True)
        self.slider_eye.blockSignals(True)
        self.slider_undereye.blockSignals(True)
        self.slider_nose.blockSignals(True)
        self.slider_cheeks.blockSignals(True)
        self.slider_forehead.blockSignals(True)
        self.slider_eye_size.blockSignals(True)
        self.slider_lip_size.blockSignals(True)
        self.lipstick_combo.blockSignals(True)
        self.slider_lipstick_strength.blockSignals(True)
        self.contacts_combo.blockSignals(True)
        self.slider_contacts_strength.blockSignals(True)
        self.look_combo.blockSignals(True)
        self.slider_look_intensity.blockSignals(True)
        self.slider_eyeliner_strength.blockSignals(True)
        self.eyeshadow_combo.blockSignals(True)
        self.slider_eyeshadow_strength.blockSignals(True)
        self.slider_lip_gloss_strength.blockSignals(True)
        self.slider_facial_highlighter_strength.blockSignals(True)
        self.gpu_checkbox.blockSignals(True)
        self.body_retouching_checkbox.blockSignals(True)
        self.slider_body_sensitivity.blockSignals(True)
        
        self.slider_smoothing.setValue(params.get("skin_smoothing", 0.0))
        self.slider_texture_recovery.setValue(params.get("skin_texture_recovery", 0.0))
        self.slider_brightening.setValue(params.get("skin_brightening", 0.0))
        self.slider_blush.setValue(params.get("blush_warmth", 0.0))
        self.slider_eye.setValue(params.get("eye_enhancement", 0.0))
        self.slider_undereye.setValue(params.get("undereye_lighten", 0.0))
        
        self.slider_nose.setValue(params.get("nose_reduce", 0.0))
        self.slider_cheeks.setValue(params.get("cheeks_reduce", 0.0))
        self.slider_forehead.setValue(params.get("forehead_reduce", 0.0))
        self.slider_eye_size.setValue(params.get("eye_enlarge", 0.0))
        self.slider_lip_size.setValue(params.get("lips_plump", 0.0))
        self.lipstick_combo.setCurrentText(params.get("lipstick_shade", "None"))
        self.slider_lipstick_strength.setValue(params.get("lipstick_strength", 0.0))
        self.contacts_combo.setCurrentText(params.get("eye_color_shade", "Natural"))
        self.slider_contacts_strength.setValue(params.get("eye_color_strength", 0.0))
        self.look_combo.setCurrentText(params.get("color_look", "None"))
        self.slider_look_intensity.setValue(params.get("look_intensity", 1.0))
        self.slider_eyeliner_strength.setValue(params.get("eyeliner_strength", 0.0))
        self.eyeshadow_combo.setCurrentText(params.get("eyeshadow_shade", "None"))
        self.slider_eyeshadow_strength.setValue(params.get("eyeshadow_strength", 0.0))
        self.slider_lip_gloss_strength.setValue(params.get("lip_gloss_strength", 0.0))
        self.slider_facial_highlighter_strength.setValue(params.get("facial_highlighter_strength", 0.0))
        self.gpu_checkbox.setChecked(params.get("gpu_acceleration", False))
        self.body_retouching_checkbox.setChecked(params.get("enable_body_retouching", False))
        # sensitivity default is 1.5, which maps to 50 on the slider
        sens_val = int((params.get("body_sensitivity", 1.5) - 0.5) / 2.0 * 100.0)
        self.slider_body_sensitivity.setValue(sens_val)
        
        self.slider_smoothing.blockSignals(False)
        self.slider_texture_recovery.blockSignals(False)
        self.slider_brightening.blockSignals(False)
        self.slider_blush.blockSignals(False)
        self.slider_eye.blockSignals(False)
        self.slider_undereye.blockSignals(False)
        self.slider_nose.blockSignals(False)
        self.slider_cheeks.blockSignals(False)
        self.slider_forehead.blockSignals(False)
        self.slider_eye_size.blockSignals(False)
        self.slider_lip_size.blockSignals(False)
        self.lipstick_combo.blockSignals(False)
        self.slider_lipstick_strength.blockSignals(False)
        self.contacts_combo.blockSignals(False)
        self.slider_contacts_strength.blockSignals(False)
        self.look_combo.blockSignals(False)
        self.slider_look_intensity.blockSignals(False)
        self.slider_eyeliner_strength.blockSignals(False)
        self.eyeshadow_combo.blockSignals(False)
        self.slider_eyeshadow_strength.blockSignals(False)
        self.slider_lip_gloss_strength.blockSignals(False)
        self.slider_facial_highlighter_strength.blockSignals(False)
        self.gpu_checkbox.blockSignals(False)
        self.body_retouching_checkbox.blockSignals(False)
        self.slider_body_sensitivity.blockSignals(False)
        
        # Redraw screen
        self._update_preview()

    # ==========================================
    # SLIDER AND PRESET SLOTS
    # ==========================================
    def _on_slider_changed(self):
        # If webcam is running, sliders are read live by the webcam loop.
        # Otherwise update static image preview
        if self.webcam_thread is None:
            self._update_preview()
            
    def _on_preset_selected(self, preset_name):
        if preset_name in DEFAULT_PRESETS:
            self.update_sliders_ui(DEFAULT_PRESETS[preset_name])
            self.statusBar().showMessage(f"Preset loaded: {preset_name}")
            
    def _on_save_preset(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Preset File", "", "JSON Files (*.json)"
        )
        if filepath:
            params = self.get_slider_params()
            if save_preset_to_file(filepath, params):
                self.statusBar().showMessage(f"Saved custom preset: {os.path.basename(filepath)}")
                
    def _on_load_preset(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Preset File", "", "JSON Files (*.json)"
        )
        if filepath:
            params = load_preset_from_file(filepath)
            if params:
                self.update_sliders_ui(params)
                self.preset_combo.setCurrentIndex(-1)  # Clear selection
                self.statusBar().showMessage(f"Loaded preset file: {os.path.basename(filepath)}")
            else:
                QMessageBox.warning(self, "Preset Load Error", "Failed to parse the preset file.")

    # ==========================================
    # SOURCE SELECTOR SLOTS
    # ==========================================
    def _on_open_video(self):
        # Stop webcam if running
        if self.webcam_thread is not None:
            self._on_toggle_webcam()
            
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Source Video", "", 
            "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf)"
        )
        if filepath:
            self.load_video_from_path(filepath)

    def load_video_from_path(self, filepath):
        self.statusBar().showMessage(f"Loading video: {os.path.basename(filepath)}...")
        self.open_video_btn.setEnabled(False)
        
        try:
            # Release existing
            if self.video_reader:
                self.video_reader.release()
                
            self.video_reader = VideoReader(filepath)
            self.current_frame_idx = 0
            
            # Setup seekbar slider
            self.scrub_bar.setEnabled(True)
            self.scrub_bar.setRange(0, self.video_reader.total_frames - 1)
            self.scrub_bar.setValue(0)
            
            self.play_btn.setEnabled(True)
            self.play_btn.setText("Play")
            self.is_playing = False
            
            self.export_btn.setEnabled(True)
            self.add_to_batch_btn.setEnabled(True)
            self.add_batch_action.setEnabled(True)
            
            # Update labels
            self.fps_label.setText(f"Source FPS: {self.video_reader.fps:.2f} | {self.video_reader.width}x{self.video_reader.height}")
            self._update_time_label()
            
            # Read first frame
            self.active_frame = self.video_reader.read_frame(0)
            self._update_preview()
            
            self.statusBar().showMessage(f"Loaded {os.path.basename(filepath)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Video Load Error", f"Could not load video: {str(e)}")
            self.statusBar().showMessage("Video loading failed")
            
        self.open_video_btn.setEnabled(True)

    def _on_sync_resolve(self):
        # Stop webcam if running
        if self.webcam_thread is not None:
            self._on_toggle_webcam()
            
        self.statusBar().showMessage("Connecting to DaVinci Resolve...")
        try:
            import sys
            import os
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
                
            from ResolveBeautyBridge import get_resolve_clip_path
            clip_path = get_resolve_clip_path()
            
            if clip_path:
                self.load_video_from_path(clip_path)
            else:
                QMessageBox.warning(
                    self, "Resolve Sync Error",
                    "Could not sync clip from Resolve.\n\n"
                    "Please make sure DaVinci Resolve is running, a project is open, "
                    "a timeline is active, and external scripting is enabled in "
                    "Preferences > System > General > External scripting using: Local."
                )
        except Exception as e:
            QMessageBox.critical(self, "Sync Error", f"Failed to connect to Resolve: {str(e)}")

    def _on_export_alpha_toggled(self, state):
        if state == 2:  # Checked
            idx = self.codec_combo.findText("ProRes (4444 Alpha)")
            if idx >= 0:
                self.codec_combo.setCurrentIndex(idx)

    def _on_codec_changed(self, text):
        if text != "ProRes (4444 Alpha)":
            self.export_alpha_checkbox.blockSignals(True)
            self.export_alpha_checkbox.setChecked(False)
            self.export_alpha_checkbox.blockSignals(False)
        else:
            self.export_alpha_checkbox.blockSignals(True)
            self.export_alpha_checkbox.setChecked(True)
            self.export_alpha_checkbox.blockSignals(False)

    def _on_toggle_webcam(self):
        # If webcam is running, turn it off
        if self.webcam_thread is not None:
            self.statusBar().showMessage("Stopping Webcam Preview...")
            self.webcam_thread.stop()
            self.webcam_thread.wait()
            self.webcam_thread = None
            
            self.webcam_btn.setText("Start Webcam Preview")
            self.webcam_btn.setChecked(False)
            
            self.active_frame = None
            self.viewer.set_frames(None, None)
            
            # Re-enable controls if a video was previously loaded
            if self.video_reader:
                self.scrub_bar.setEnabled(True)
                self.play_btn.setEnabled(True)
                self.export_btn.setEnabled(True)
                self.add_to_batch_btn.setEnabled(True)
                self.add_batch_action.setEnabled(True)
                self.active_frame = self.video_reader.read_frame(self.current_frame_idx)
                self._update_preview()
            self.statusBar().showMessage("Webcam preview stopped")
            
        else:
            # If playing video, pause it
            if self.is_playing:
                self._on_toggle_play()
                
            self.statusBar().showMessage("Starting Webcam Preview (Mirror Mode)...")
            self.webcam_btn.setText("Stop Webcam")
            self.webcam_btn.setChecked(True)
            
            # Disable video playback controls
            self.scrub_bar.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.add_to_batch_btn.setEnabled(False)
            self.add_batch_action.setEnabled(False)
            
            # Start Webcam worker thread
            self.webcam_thread = WebcamThread(self.filter_engine, self.get_slider_params)
            self.webcam_thread.frame_ready.connect(self._on_webcam_frame)
            self.webcam_thread.error.connect(self._on_webcam_error)
            self.webcam_thread.start()

    def _on_webcam_frame(self, processed_frame, original_frame):
        self.viewer.set_frames(original_frame, processed_frame)
        
    def _on_webcam_error(self, err_msg):
        QMessageBox.critical(self, "Webcam Error", err_msg)
        self._on_toggle_webcam()

    # ==========================================
    # PLAYBACK TIMELINE CONTROL SLOTS
    # ==========================================
    def _on_toggle_play(self):
        if not self.video_reader:
            return
            
        if self.is_playing:
            self.video_timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
            self.statusBar().showMessage("Paused")
        else:
            # Calculate interval in ms
            interval = int(1000.0 / self.video_reader.fps)
            self.video_timer.start(interval)
            self.play_btn.setText("Pause")
            self.is_playing = True
            self.statusBar().showMessage("Playing...")

    def _on_play_step(self):
        if not self.video_reader:
            return
            
        next_idx = self.current_frame_idx + 1
        if next_idx >= self.video_reader.total_frames:
            # Loop around
            next_idx = 0
            
        self.current_frame_idx = next_idx
        
        # Block signals on scrub bar so the user scrub position moves but doesn't trigger seek callbacks
        self.scrub_bar.blockSignals(True)
        self.scrub_bar.setValue(self.current_frame_idx)
        self.scrub_bar.blockSignals(False)
        
        self._update_time_label()
        
        # Read frame
        frame = self.video_reader.read_frame(self.current_frame_idx)
        if frame is not None:
            self.active_frame = frame
            self._update_preview()

    def _on_scrub_pressed(self):
        # Pause playback during scrubbing
        if self.is_playing:
            self._on_toggle_play()

    def _on_scrub_moved(self, frame_idx):
        if not self.video_reader:
            return
            
        self.current_frame_idx = frame_idx
        self._update_time_label()
        
        frame = self.video_reader.read_frame(self.current_frame_idx)
        if frame is not None:
            self.active_frame = frame
            self._update_preview()

    def _update_time_label(self):
        if not self.video_reader:
            return
            
        curr_seconds = self.current_frame_idx / self.video_reader.fps
        total_seconds = self.video_reader.total_frames / self.video_reader.fps
        
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds % 1) * 100)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:02d}"
            
        self.time_label.setText(f"{format_time(curr_seconds)} / {format_time(total_seconds)}")

    def _set_compare_mode(self, mode):
        # Sync widget check states
        self.btn_mode_split.setChecked(mode == "split")
        self.btn_mode_filtered.setChecked(mode == "processed")
        self.btn_mode_original.setChecked(mode == "original")
        
        self.viewer.set_mode(mode)
        
    def _update_preview(self):
        if self.active_frame is None:
            return
            
        # For high-speed GUI responsiveness, downscale preview frame to max width 640px
        params = self.get_slider_params()
        processed = self.filter_engine.process_frame(self.active_frame, params, preview_width=640)
        
        # Match dimensions of original and processed frames for side-by-side comparison in viewer
        original_preview = self.active_frame
        h_p, w_p = processed.shape[:2]
        h_o, w_o = original_preview.shape[:2]
        if w_o != w_p or h_o != h_p:
            original_preview = cv2.resize(original_preview, (w_p, h_p), interpolation=cv2.INTER_AREA)
            
        self.viewer.set_frames(original_preview, processed)

    # ==========================================
    # VIDEO EXPORT SLOTS
    # ==========================================
    def _on_export_video(self):
        if not self.video_reader:
            return
            
        # Pause playback if running
        if self.is_playing:
            self._on_toggle_play()
            
        # Open save file dialog
        default_name = "beauty_" + os.path.basename(self.video_reader.filepath)
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Export Video As", default_name,
            "Video Files (*.mov *.mp4)"
        )
        
        if filepath:
            self.statusBar().showMessage("Initializing export thread...")
            self._set_export_ui_active(True)
            
            # Setup thread
            codec = self.codec_combo.currentText()
            params = self.get_slider_params()
            
            self.export_thread = VideoProcessorThread(
                self.video_reader.filepath, filepath, codec, self.filter_engine, params
            )
            self.export_thread.progress.connect(self._on_export_progress)
            self.export_thread.completed.connect(self._on_export_completed)
            self.export_thread.error.connect(self._on_export_error)
            self.export_thread.start()
            
    def _on_export_progress(self, frame_idx, percentage, eta_str):
        self.progress_bar.setValue(int(percentage))
        self.eta_label.setText(f"ETA: {eta_str}")
        self.statusBar().showMessage(f"Processing frame {frame_idx} ({int(percentage)}%)")
        
    def _on_export_completed(self, output_path, use_fallback):
        self._set_export_ui_active(False)
        self.statusBar().showMessage(f"Export complete: {os.path.basename(output_path)}")
        
        msg = f"Video successfully exported to:\n{output_path}"
        if use_fallback:
            msg += "\n\nNote: The requested codec failed to initialize, so H.264 MP4 format was used as fallback."
            
        QMessageBox.information(self, "Export Successful", msg)
        
    def _on_export_error(self, err_msg):
        self._set_export_ui_active(False)
        self.statusBar().showMessage("Export failed")
        QMessageBox.critical(self, "Export Error", f"An error occurred during video processing:\n{err_msg}")
        
    def _set_export_ui_active(self, active):
        # Toggle buttons disable state to prevent concurrent modifications
        self.open_video_btn.setEnabled(not active)
        self.sync_resolve_btn.setEnabled(not active)
        self.webcam_btn.setEnabled(not active)
        self.export_btn.setEnabled(not active)
        self.add_to_batch_btn.setEnabled(not active)
        self.add_batch_action.setEnabled(not active)
        self.show_queue_btn.setEnabled(not active)
        self.export_alpha_checkbox.setEnabled(not active)
        self.save_preset_btn.setEnabled(not active)
        self.load_preset_btn.setEnabled(not active)
        self.preset_combo.setEnabled(not active)
        self.codec_combo.setEnabled(not active)
        self.play_btn.setEnabled(not active)
        
        # Display progress panel
        self.progress_panel.setVisible(active)
        if active:
            self.progress_bar.setValue(0)
            self.eta_label.setText("ETA: Calculating...")
            
    def closeEvent(self, event):
        # Shut down threads before quitting
        if self.webcam_thread:
            self.webcam_thread.stop()
            self.webcam_thread.wait()
        if self.export_thread:
            self.export_thread.stop()
            self.export_thread.wait()
        if self.video_reader:
            self.video_reader.release()
        if self.batch_processor:
            self.batch_processor.pause_queue()
        event.accept()

    # ==========================================
    # APP ICON AND MENUBAR INITIALIZATION
    # ==========================================
    def create_app_icon(self):
        # Path to project root icon.png
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icon.png")
        if not os.path.exists(icon_path):
            pixmap = QPixmap(256, 256)
            pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw gradient background circle
            grad = QLinearGradient(0, 0, 256, 256)
            grad.setColorAt(0.0, QColor("#ff9f1c"))  # Resolve Amber
            grad.setColorAt(1.0, QColor("#e63946"))  # Coral
            
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(12, 12, 232, 232)
            
            # Draw camera lens rings outline in white
            painter.setPen(QPen(QColor(255, 255, 255), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(68, 68, 120, 120)
            painter.drawEllipse(98, 98, 60, 60)
            
            # Draw beautiful sparkles at top-right
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255))
            
            # Stylized diamond star polygon
            sparkle = QPolygonF([
                QPointF(200, 50),
                QPointF(206, 60),
                QPointF(216, 66),
                QPointF(206, 72),
                QPointF(200, 82),
                QPointF(194, 72),
                QPointF(184, 66),
                QPointF(194, 60)
            ])
            painter.drawPolygon(sparkle)
            
            painter.end()
            
            try:
                pixmap.save(icon_path, "PNG")
            except Exception as e:
                print(f"Failed to save icon.png: {e}")
                
        self.setWindowIcon(QIcon(icon_path))

    def setup_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border-bottom: 1px solid #2a2a2a;
                font-weight: 500;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #ff9f1c;
                color: #121212;
                font-weight: bold;
            }
            QMenu {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #2d2d2d;
            }
            QMenu::item {
                padding: 6px 20px 6px 25px;
            }
            QMenu::item:selected {
                background-color: #ff9f1c;
                color: #121212;
            }
            QMenu::separator {
                height: 1px;
                background-color: #2d2d2d;
                margin: 4px 0px;
            }
        """)
        
        # 1. File Menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Video File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_video)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_lut_action = QAction("Export Look as 3D LUT (.cube)...", self)
        export_lut_action.triggered.connect(self._on_export_lut)
        file_menu.addAction(export_lut_action)
        
        file_menu.addSeparator()
        
        save_preset_action = QAction("Save Preset As...", self)
        save_preset_action.setShortcut("Ctrl+S")
        save_preset_action.triggered.connect(self._on_save_preset)
        file_menu.addAction(save_preset_action)
        
        load_preset_action = QAction("Load Preset File...", self)
        load_preset_action.setShortcut("Ctrl+L")
        load_preset_action.triggered.connect(self._on_load_preset)
        file_menu.addAction(load_preset_action)
        
        file_menu.addSeparator()
        
        add_batch_action = QAction("Add Current Clip to Queue...", self)
        add_batch_action.setShortcut("Ctrl+B")
        add_batch_action.triggered.connect(self._on_add_to_batch_queue)
        file_menu.addAction(add_batch_action)
        self.add_batch_action = add_batch_action
        self.add_batch_action.setEnabled(False)
        
        show_batch_action = QAction("Open Batch Manager Dialog...", self)
        show_batch_action.triggered.connect(self._on_show_batch_dialog)
        file_menu.addAction(show_batch_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 2. View Menu
        view_menu = menubar.addMenu("View")
        
        split_action = QAction("Split Screen Mode", self)
        split_action.triggered.connect(lambda: self._set_compare_mode("split"))
        view_menu.addAction(split_action)
        
        filtered_action = QAction("Filtered (After) Mode", self)
        filtered_action.triggered.connect(lambda: self._set_compare_mode("processed"))
        view_menu.addAction(filtered_action)
        
        original_action = QAction("Original (Before) Mode", self)
        original_action.triggered.connect(lambda: self._set_compare_mode("original"))
        view_menu.addAction(original_action)
        
        # 3. Help Menu
        help_menu = menubar.addMenu("Help")
        
        manual_action = QAction("Instruction Manual...", self)
        manual_action.setShortcut("F1")
        manual_action.triggered.connect(self._on_manual_triggered)
        help_menu.addAction(manual_action)
        
        shortcuts_action = QAction("Shortcuts Reference...", self)
        shortcuts_action.triggered.connect(self._on_shortcuts_triggered)
        help_menu.addAction(shortcuts_action)
        
        about_action = QAction("About Resolve Beauty Companion...", self)
        about_action.triggered.connect(self._on_about_triggered)
        help_menu.addAction(about_action)

    def _on_export_lut(self):
        params = self.get_slider_params()
        filter_name = params.get('color_look', 'None')
        intensity = params.get('look_intensity', 1.0)
        
        if filter_name == "None":
            QMessageBox.warning(
                self, "Export 3D LUT",
                "No Cinematic Look filter is currently selected.\n"
                "Please select a look from the 'Cinematic Looks' panel before exporting."
            )
            return
            
        default_name = filter_name.lower().replace(" ", "_") + ".cube"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Look as 3D LUT", default_name,
            "LUT Files (*.cube)"
        )
        
        if filepath:
            self.statusBar().showMessage("Generating 3D LUT...")
            try:
                self.filter_engine.generate_cube_lut(filter_name, intensity, filepath)
                self.statusBar().showMessage(f"LUT exported: {os.path.basename(filepath)}")
                QMessageBox.information(
                    self, "LUT Export Successful",
                    f"Successfully generated 3D LUT for look '{filter_name}' at:\n{filepath}"
                )
            except Exception as e:
                QMessageBox.critical(self, "LUT Export Error", f"Failed to export 3D LUT:\n{str(e)}")

    def _on_about_triggered(self):
        QMessageBox.about(
            self, "About Resolve Beauty Companion",
            "<h3>Resolve Beauty Companion v1.0.0</h3>"
            "<p>A professional high-performance companion app to DaVinci Resolve for real-time face smoothing and beauty filtering.</p>"
            "<p>Features Snapchat-style face reshaping, bilateral skin smoothing, eye and cheek enhancements, and custom JSON preset management.</p>"
            "<p>Built using PyQt6, OpenCV, MediaPipe Tasks, NumPy, and SciPy.</p>"
            "<p><i>Copyright &copy; 2026. All rights reserved.</i></p>"
        )
        
    def _on_shortcuts_triggered(self):
        QMessageBox.information(
            self, "Keyboard Shortcuts Reference",
            "<b>F1</b>: Open the User Instruction Manual<br>"
            "<b>Spacebar</b>: Toggle play/pause video preview (or freeze/resume webcam)<br>"
            "<b>Enter / Return</b>: Trigger video export processing<br>"
            "<b>Ctrl + O</b>: Open video file<br>"
            "<b>Ctrl + S</b>: Save preset file<br>"
            "<b>Ctrl + L</b>: Load preset file<br>"
            "<b>Ctrl + Q</b>: Quit the application"
        )

    def _on_manual_triggered(self):
        dialog = InstructionManualDialog(self)
        dialog.exec()

    def _on_add_to_batch_queue(self):
        if not self.video_reader:
            return
            
        if self.is_playing:
            self._on_toggle_play()
            
        default_name = "beauty_" + os.path.basename(self.video_reader.filepath)
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Export Queue Video As", default_name,
            "Video Files (*.mov *.mp4)"
        )
        
        if filepath:
            codec = self.codec_combo.currentText()
            params = self.get_slider_params()
            preset_name = self.preset_combo.currentText() or "Custom"
            
            self.batch_processor.add_item(
                input_path=self.video_reader.filepath,
                output_path=filepath,
                codec_name=codec,
                params=params,
                preset_name=preset_name
            )
            
            self.statusBar().showMessage(f"Added clip to Batch Queue: {os.path.basename(filepath)}")
            self._on_show_batch_dialog()

    def _on_show_batch_dialog(self):
        self.batch_dialog.show()
        self.batch_dialog.raise_()
        self.batch_dialog.activateWindow()

from PyQt6.QtWidgets import QDialog, QTextBrowser

class InstructionManualDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resolve Beauty Companion - Instruction Manual")
        self.setMinimumSize(800, 650)
        self.setStyleSheet("""
            QDialog {
                background-color: #121212;
            }
            QPushButton {
                background-color: #ff9f1c;
                color: #121212;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #ffa834;
            }
            QPushButton:pressed {
                background-color: #e68e14;
            }
            QTextBrowser {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #2d2d2d;
                border-radius: 6px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # HTML instruction browser
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(self.get_manual_html())
        layout.addWidget(self.browser)
        
        # Bottom controls
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close Manual")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
    def get_manual_html(self):
        return """
        <html>
        <head>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #e0e0e0;
                background-color: #1a1a1a;
                margin: 0;
                padding: 0;
            }
            h1 {
                color: #ff9f1c;
                border-bottom: 2px solid #ff9f1c;
                padding-bottom: 10px;
                font-size: 22px;
                margin-top: 0;
            }
            h2 {
                color: #ff9f1c;
                border-bottom: 1px solid #2d2d2d;
                padding-bottom: 5px;
                font-size: 16px;
                margin-top: 20px;
            }
            h3 {
                color: #ffa834;
                font-size: 13px;
                margin-top: 15px;
                margin-bottom: 5px;
            }
            p, li {
                font-size: 12px;
                line-height: 1.6;
            }
            code {
                font-family: Consolas, monospace;
                background-color: #2b2b2b;
                color: #ff9f1c;
                padding: 2px 5px;
                border-radius: 3px;
                font-size: 11px;
            }
            ul, ol {
                margin-top: 5px;
                margin-bottom: 15px;
                padding-left: 20px;
            }
            .accent {
                color: #ff9f1c;
                font-weight: bold;
            }
            .tip-box {
                background-color: #24211a;
                border-left: 4px solid #ff9f1c;
                padding: 10px 15px;
                margin: 15px 0;
                border-radius: 0 4px 4px 0;
                font-size: 12px;
            }
        </style>
        </head>
        <body>
        <h1>Resolve Beauty Companion - User Manual</h1>

        <p>Welcome to the <b>Resolve Beauty Companion</b> user manual. This application acts as a real-time facial retouching, beauty filtering, and face reshaping companion to <b>DaVinci Resolve</b>.</p>

        <div class="tip-box">
            <b>Quick Start:</b> If DaVinci Resolve is active, press the <b>Sync Clip from Resolve</b> button under <i>Source Selection</i> in the sidebar to auto-load the current timeline clip.
        </div>

        <h2>1. Adjustment Sliders Guide</h2>
        <p>These features apply skin-smoothing and lighting enhancements to the face:</p>
        <ul>
            <li><b>Skin Smoothing:</b> Utilizes bilateral filtering to smooth skin blemishes, pores, and tone, while ignoring contours like eyes, nostrils, eyebrows, and lips.</li>
            <li><b>Skin Texture Recovery:</b> Extracts and blends original skin pore details back onto smoothed regions to maintain organic realism and prevent artificial blur.</li>
            <li><b>Skin Brightening:</b> Increases lighting luminance on skin zones for a radiant glow.</li>
            <li><b>Blush / Warmth:</b> Automatically places warmth and red tones on the cheeks.</li>
            <li><b>Under-eye Lighten:</b> Brightens shadow details directly under the eyes, minimizing dark circles.</li>
            <li><b>Eye Enhancement:</b> Sharpens details and adjusts local contrast inside the eye shape for brighter, more vivid eyes.</li>
        </ul>

        <h2>2. Face Reshaping & Cosmetic Makeup</h2>
        <p>These settings warp facial dimensions and apply overlays:</p>
        <ul>
            <li><b>Nose Size Reduce:</b> Horizontal pinch warp that slims the tip and width of the nose.</li>
            <li><b>Cheek Slimming:</b> Dual jawline pinch warps that slim the lower cheek contours.</li>
            <li><b>Forehead Reduce:</b> Shrugs the hairline vertically downwards.</li>
            <li><b>Eye Size (Enlarge):</b> Bulge warps centered on left/right irises to enlarge the eyes.</li>
            <li><b>Lip Size (Plump):</b> Bulge warp centered on mouth coordinates to plump the lips.</li>
            <li><b>Lipstick Shade & Strength:</b> Color makeup overlay on the lip contour (Rose Red, Soft Pink, Peach Glow, Plum Berry).</li>
            <li><b>Eye Color Shade & Strength:</b> Colored contact lens overlay on the iris boundaries (Ocean Blue, Emerald Green, Honey Brown, Deep Amber).</li>
            <li><b>Eyeliner & Mascara:</b> Draws a thick, dark, winged line along the upper eyelids with soft blurring to simulate mascara volume.</li>
            <li><b>Eyeshadow Shade & Strength:</b> Applies a feathery, multi-colored gradient on the upper eyelids and socket regions.</li>
            <li><b>Lip Gloss (Specular):</b> Overlays shimmering, screen-blended specular highlights on the upper and lower lips.</li>
            <li><b>Facial Highlighter:</b> Adds subtle, screen-blended champagne gold highlights on the cheekbones and nose bridge for a three-dimensional contouring effect.</li>
        </ul>

        <h2>3. Cinematic Looks & 3D LUT Export</h2>
        <p>Export look presets directly into Resolve color nodes:</p>
        <ul>
            <li><b>Save Look as LUT:</b> Select <span class="accent">File > Export Look as 3D LUT (.cube)...</span> to output a standard 33x33x33 3D LUT file.</li>
            <li><b>Load into DaVinci Resolve:</b>
                <ol>
                    <li>In DaVinci Resolve, go to <b>Project Settings > Color Management</b>.</li>
                    <li>Click <b>Open LUT Folder</b> under Lookup Tables.</li>
                    <li>Paste your exported <code>.cube</code> file inside.</li>
                    <li>Click <b>Update Lists</b>. The LUT can now be applied to any grading node.</li>
                </ol>
            </li>
        </ul>

        <h2>4. ProRes 4444 Alpha Overlay Export</h2>
        <p>Export transparent overlays to layer beauty effects on top of Resolve clips:</p>
        <ol>
            <li>Configure skin smoothing and makeup adjustments.</li>
            <li>Check <b>Export Alpha Overlay Only</b> (locks output codec to <b>ProRes (4444 Alpha)</b>).</li>
            <li>Export the video. Drag the transparent <code>.mov</code> clip into DaVinci Resolve directly onto a track **above** your source graded clip.</li>
        </ol>

        <h2>5. Resolve Timeline Scripting Bridge</h2>
        <p>Launch the companion app directly from Resolve's workspace:</p>
        <ol>
            <li>Copy the bridge script <code>ResolveBeautyBridge.py</code> from the app directory.</li>
            <li>Paste it inside Resolve's scripting folder:
                <br><code>C:\\Users\\REBORN PIX3LS\\AppData\\Roaming\\Blackmagic Design\\DaVinci Resolve\\Support\\Developer\\Scripting\\Utility\\</code>
            </li>
            <li>Restart Resolve and select: <b>Workspace > Scripts > ResolveBeautyBridge</b>.</li>
        </ol>

        <h2>6. ONNX DirectML GPU Acceleration</h2>
        <p>For high-resolution 4K timelines or heavy presets, enable GPU acceleration:</p>
        <ul>
            <li>Check <b>Enable GPU Acceleration (DirectML ONNX)</b> in the export panel.</li>
            <li>This offloads face mesh landmark estimation from the CPU to the GPU using Microsoft DirectML, compatible with AMD, Intel, and NVIDIA graphics processors.</li>
            <li>A seamless automatic fallback to standard MediaPipe CPU is built-in if no compatible DirectX 12 graphics processor is detected or if initialization fails.</li>
        </ul>

        <h2>7. Body & Neck Skin Retouching</h2>
        <p>Apply skin smoothing and skin brightening to the neck and upper body chest regions:</p>
        <ul>
            <li>Check <b>Enable Body & Neck Retouching</b> in the sidebar.</li>
            <li>This dynamically samples skin color tones from the tracked face oval and runs an adaptive YCrCb color segmentation on a cropped region of interest (ROI) extending below the chin.</li>
            <li>Adjust <b>Body Skin Sensitivity</b> to adapt the skin-color matching range. Higher values expand the matching threshold (useful for varied lighting), while lower values prevent color bleed onto background elements.</li>
        </ul>
        </body>
        </html>
        """
