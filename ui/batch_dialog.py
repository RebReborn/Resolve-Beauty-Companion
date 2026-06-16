from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QProgressBar, QComboBox, QFileDialog, QMessageBox, QLabel)
from PyQt6.QtCore import Qt, QSize
import os

from core.batch_manager import BatchQueueItem

class BatchManagerDialog(QDialog):
    """
    Non-modal Dialog representing the Batch queue management panel.
    """
    def __init__(self, processor, main_window, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.main_window = main_window
        self.setWindowTitle("Batch Render Queue Manager")
        self.setMinimumSize(QSize(900, 500))
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
        
        self.init_ui()
        self.apply_theme()
        
        # Load initial queue
        self.refresh_table()
        
        # Connect processor signals
        self.processor.item_added.connect(self._on_item_added)
        self.processor.item_removed.connect(self._on_item_removed)
        self.processor.item_updated.connect(self._on_item_updated)
        self.processor.queue_status_changed.connect(self._on_queue_status_changed)
        self.processor.queue_finished.connect(self._on_queue_finished)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Header title
        title_label = QLabel("Batch Export Timeline Queue")
        title_label.setStyleSheet("color: #ff9f1c; font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)

        # Queue Table
        self.table = QTableWidget()
        self.headers = ["Input Clip", "Preset/Profile", "Codec", "Output Path", "Status", "Progress", "ETA"]
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Align columns sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 130)
        self.table.setColumnWidth(5, 140)
        
        layout.addWidget(self.table)
        
        # Button controls
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.add_btn = QPushButton("Add Clips...")
        self.add_btn.clicked.connect(self._on_add_clips)
        
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._on_remove_selected)
        
        self.clear_btn = QPushButton("Clear Finished")
        self.clear_btn.clicked.connect(self._on_clear_finished)
        
        self.start_btn = QPushButton("Start Queue")
        self.start_btn.clicked.connect(self._on_start_queue)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9f1c;
                color: #121212;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffa834;
            }
        """)
        
        self.pause_btn = QPushButton("Pause Queue")
        self.pause_btn.clicked.connect(self._on_pause_queue)
        self.pause_btn.setEnabled(False)
        
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.start_btn)
        
        layout.addLayout(btn_layout)

    def apply_theme(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #121212;
            }
            QTableWidget {
                background-color: #1a1a1a;
                color: #e0e0e0;
                gridline-color: #2d2d2d;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #ff9f1c;
                padding: 6px;
                border: 1px solid #2d2d2d;
                font-weight: bold;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 6px;
                font-size: 11px;
            }
            QTableWidget::item:selected {
                background-color: #ff9f1c;
                color: #121212;
            }
            QPushButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px 15px;
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
            QPushButton:disabled {
                background-color: #1a1a1a;
                border-color: #2b2b2b;
                color: #5d5d5d;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)

    def refresh_table(self):
        self.table.setRowCount(0)
        for item in self.processor.items:
            self._insert_row_for_item(item)

    def _insert_row_for_item(self, item):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Column 0: Input Clip
        name_item = QTableWidgetItem(item.get_display_name())
        name_item.setToolTip(item.input_path)
        name_item.setData(Qt.ItemDataRole.UserRole, item.id)
        self.table.setItem(row, 0, name_item)
        
        # Column 1: Preset Dropdown
        preset_combo = self._create_preset_combobox(item)
        self.table.setCellWidget(row, 1, preset_combo)
        
        # Column 2: Codec Dropdown
        codec_combo = self._create_codec_combobox(item)
        self.table.setCellWidget(row, 2, codec_combo)
        
        # Column 3: Output Path
        out_item = QTableWidgetItem(os.path.basename(item.output_path))
        out_item.setToolTip(item.output_path)
        self.table.setItem(row, 3, out_item)
        
        # Column 4: Status
        status_item = QTableWidgetItem(item.status)
        if item.status == "Completed":
            status_item.setForeground(Qt.GlobalColor.green)
        elif item.status == "Failed":
            status_item.setForeground(Qt.GlobalColor.red)
        elif item.status == "Processing":
            status_item.setForeground(QColor("#ff9f1c"))
        self.table.setItem(row, 4, status_item)
        
        # Column 5: Progress Bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(int(item.progress))
        progress_bar.setTextVisible(True)
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                text-align: center;
                color: #fff;
                background-color: #121212;
                font-size: 10px;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #ff9f1c;
                border-radius: 2px;
            }
        """)
        self.table.setCellWidget(row, 5, progress_bar)
        
        # Column 6: ETA
        eta_item = QTableWidgetItem(item.eta)
        self.table.setItem(row, 6, eta_item)

        # Update enable status
        is_queued = (item.status == "Queued")
        preset_combo.setEnabled(is_queued)
        codec_combo.setEnabled(is_queued)

    def _find_row_by_id(self, item_id):
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == item_id:
                return r
        return -1

    def _create_preset_combobox(self, item):
        from utils.presets import DEFAULT_PRESETS
        combo = QComboBox()
        combo.addItems(list(DEFAULT_PRESETS.keys()) + ["Current GUI Settings"])
        combo.setCurrentText(item.preset_name)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: none;
                padding: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1a;
                color: #e0e0e0;
                selection-background-color: #ff9f1c;
                selection-color: #121212;
            }
        """)
        
        def on_preset_changed(text):
            if text in DEFAULT_PRESETS:
                item.params = DEFAULT_PRESETS[text].copy()
                item.preset_name = text
            elif text == "Current GUI Settings" and self.main_window:
                item.params = self.main_window.get_slider_params()
                item.preset_name = "Current GUI Settings"
                
        combo.currentTextChanged.connect(on_preset_changed)
        return combo

    def _create_codec_combobox(self, item):
        from core.processor import CODEC_MAP
        combo = QComboBox()
        combo.addItems(list(CODEC_MAP.keys()))
        combo.setCurrentText(item.codec_name)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: none;
                padding: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1a;
                color: #e0e0e0;
                selection-background-color: #ff9f1c;
                selection-color: #121212;
            }
        """)
        
        def on_codec_changed(text):
            item.codec_name = text
            if text == "ProRes (4444 Alpha)":
                item.params['export_alpha'] = True
            else:
                item.params['export_alpha'] = False
                
        combo.currentTextChanged.connect(on_codec_changed)
        return combo

    # ==========================================
    # PROCESSOR SIGNAL SLOTS
    # ==========================================
    def _on_item_added(self, item):
        self._insert_row_for_item(item)

    def _on_item_removed(self, item_id):
        row = self._find_row_by_id(item_id)
        if row != -1:
            self.table.removeRow(row)

    def _on_item_updated(self, item):
        row = self._find_row_by_id(item.id)
        if row == -1:
            return
            
        # Update Status text
        status_item = self.table.item(row, 4)
        if status_item:
            status_item.setText(item.status)
            if item.status == "Completed":
                status_item.setForeground(Qt.GlobalColor.green)
            elif item.status == "Failed":
                status_item.setForeground(Qt.GlobalColor.red)
            elif item.status == "Processing":
                status_item.setForeground(Qt.GlobalColor.yellow)
            else:
                status_item.setForeground(QTableWidgetItem().foreground())
                
        # Update Progress Bar value
        p_bar = self.table.cellWidget(row, 5)
        if p_bar:
            p_bar.setValue(int(item.progress))
            
        # Update ETA text
        eta_item = self.table.item(row, 6)
        if eta_item:
            eta_item.setText(item.eta)
            
        # Update tooltips or details if item failed
        if item.status == "Failed" and item.error_message:
            self.table.item(row, 0).setToolTip(f"Error: {item.error_message}\nPath: {item.input_path}")
            
        # Lock comboboxes if not Queued
        preset_combo = self.table.cellWidget(row, 1)
        codec_combo = self.table.cellWidget(row, 2)
        is_queued = (item.status == "Queued")
        if preset_combo:
            preset_combo.setEnabled(is_queued)
        if codec_combo:
            codec_combo.setEnabled(is_queued)

    def _on_queue_status_changed(self, is_processing):
        self.start_btn.setEnabled(not is_processing)
        self.pause_btn.setEnabled(is_processing)
        self.add_btn.setEnabled(not is_processing)
        self.remove_btn.setEnabled(not is_processing)
        self.clear_btn.setEnabled(not is_processing)

    def _on_queue_finished(self):
        QMessageBox.information(self, "Batch Processing Finished", "All queued video export jobs have finished rendering.")

    # ==========================================
    # USER ACTION SLOTS
    # ==========================================
    def _on_add_clips(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Select Clips to Queue", "", 
            "Video Files (*.mp4 *.mov *.avi *.mkv *.mxf)"
        )
        if not filepaths:
            return
            
        from utils.presets import DEFAULT_PRESETS
        
        # Grab main window configuration for codec/params
        active_codec = "ProRes (Standard)"
        active_params = {}
        if self.main_window:
            active_codec = self.main_window.codec_combo.currentText()
            active_params = self.main_window.get_slider_params()
        else:
            active_params = DEFAULT_PRESETS["Natural Glow"].copy()
            
        for filepath in filepaths:
            # Generate default output name
            base, ext = os.path.splitext(filepath)
            out_path = f"{base}_beauty{ext}"
            
            # Add to processor
            self.processor.add_item(
                input_path=filepath,
                output_path=out_path,
                codec_name=active_codec,
                params=active_params,
                preset_name="Current GUI Settings" if self.main_window else "Natural Glow"
            )

    def _on_remove_selected(self):
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            return
            
        row = selected_ranges[0].topRow()
        item = self.table.item(row, 0)
        if not item:
            return
            
        item_id = item.data(Qt.ItemDataRole.UserRole)
        self.processor.remove_item(item_id)

    def _on_clear_finished(self):
        self.processor.clear_completed()

    def _on_start_queue(self):
        self.processor.start_queue()

    def _on_pause_queue(self):
        self.processor.pause_queue()
        
    def closeEvent(self, event):
        # We just hide the window, don't destroy it so it runs in background
        event.accept()
