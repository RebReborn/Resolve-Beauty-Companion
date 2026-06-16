from PyQt6.QtWidgets import QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout
from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QColor, QPen
import cv2
import numpy as np

class ResetClickSlider(QSlider):
    """
    Subclass of QSlider that resets to a default value when double-clicked.
    """
    def __init__(self, default_val=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_val = default_val
        
    def mouseDoubleClickEvent(self, event):
        self.setValue(self.default_val)
        super().mouseDoubleClickEvent(event)


class PrecisionSlider(QWidget):
    """
    A widget grouping a label, a double-click resettable slider,
    and a value readout percentage label.
    """
    valueChanged = pyqtSignal(float)  # Emits value normalized to 0.0 - 1.0
    
    def __init__(self, label_text, default_val=0, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.default_val = default_val
        
        # Horizontal Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        
        # Name Label
        self.name_label = QLabel(label_text)
        self.name_label.setStyleSheet("color: #e0e0e0; font-weight: 500; font-size: 12px;")
        self.name_label.setFixedWidth(120)
        layout.addWidget(self.name_label)
        
        # Slider
        self.slider = ResetClickSlider(default_val, Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default_val)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3d3d3d;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #ff9f1c; /* Resolve Amber */
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #e0e0e0;
                border: 1px solid #5d5d5d;
                width: 14px;
                height: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #ff9f1c;
                border-color: #ffb74d;
            }
        """)
        layout.addWidget(self.slider)
        
        # Value Label
        self.value_label = QLabel(f"{default_val}%")
        self.value_label.setStyleSheet("color: #ff9f1c; font-weight: bold; font-size: 12px;")
        self.value_label.setFixedWidth(45)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.value_label)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_value_changed)
        
    def _on_value_changed(self, val):
        self.value_label.setText(f"{val}%")
        self.valueChanged.emit(val / 100.0)
        
    def value(self):
        return self.slider.value() / 100.0
        
    def setValue(self, val):
        # Expects 0.0 - 1.0
        slider_val = int(val * 100.0)
        self.slider.setValue(slider_val)


class BeforeAfterViewer(QWidget):
    """
    A custom video widget that displays the original and filtered frames
    with support for "Original only", "Filtered only", and a draggable "Split Screen".
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_frame = None   # NumPy BGR array
        self.processed_frame = None  # NumPy BGR array
        
        self.split_ratio = 0.5
        self.is_dragging_split = False
        self.show_split = True
        self.mode = "split"  # "original", "processed", "split"
        
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #121212;")
        
    def set_frames(self, original, processed):
        self.original_frame = original
        self.processed_frame = processed
        self.update()  # Trigger paintEvent
        
    def set_mode(self, mode):
        # mode is "original", "processed", or "split"
        self.mode = mode
        self.update()
        
    def get_display_rect(self, frame_w, frame_h):
        widget_w = self.width()
        widget_h = self.height()
        if widget_w <= 0 or widget_h <= 0 or frame_w <= 0 or frame_h <= 0:
            return QRect(0, 0, widget_w, widget_h)
            
        aspect_ratio = frame_w / frame_h
        widget_ratio = widget_w / widget_h
        
        if widget_ratio > aspect_ratio:
            # height limits scaling
            h = widget_h
            w = int(h * aspect_ratio)
            x = (widget_w - w) // 2
            y = 0
        else:
            # width limits scaling
            w = widget_w
            h = int(w / aspect_ratio)
            x = 0
            y = (widget_h - h) // 2
            
        return QRect(x, y, w, h)
        
    def numpy_to_qimage(self, bgr_img):
        if bgr_img is None:
            return QImage()
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        # QImage constructor takes a pointer to data. Use copy() to avoid segfaults
        return QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw dark canvas background
        painter.fillRect(self.rect(), QColor(15, 15, 15))
        
        if self.original_frame is None:
            # Draw placeholder text when no video is loaded
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(self.font())
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a Video or start Webcam Preview to begin")
            return
            
        frame_h, frame_w = self.original_frame.shape[:2]
        rect = self.get_display_rect(frame_w, frame_h)
        
        # Convert frames to QImage
        q_original = self.numpy_to_qimage(self.original_frame)
        q_processed = self.numpy_to_qimage(self.processed_frame)
        
        if self.mode == "original":
            painter.drawImage(rect, q_original)
        elif self.mode == "processed":
            painter.drawImage(rect, q_processed)
        else:  # "split" mode
            # Calculate split coordinate in widget pixels
            split_x = rect.x() + int(rect.width() * self.split_ratio)
            
            # 1. Draw Original on Left Half
            painter.save()
            painter.setClipRect(0, 0, split_x, self.height())
            painter.drawImage(rect, q_original)
            painter.restore()
            
            # 2. Draw Processed on Right Half
            painter.save()
            painter.setClipRect(split_x, 0, self.width(), self.height())
            painter.drawImage(rect, q_processed)
            painter.restore()
            
            # 3. Draw Split Separator Line and Handle
            if self.show_split:
                # Vertical line
                pen = QPen(QColor(255, 159, 28), 2)  # Amber
                painter.setPen(pen)
                painter.drawLine(split_x, rect.y(), split_x, rect.y() + rect.height())
                
                # Circular handle in center
                handle_y = rect.y() + rect.height() // 2
                painter.setBrush(QColor(255, 159, 28))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(split_x - 7, handle_y - 15, 14, 30)
                
                # Tiny white grip lines inside handle
                painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
                painter.drawLine(split_x - 3, handle_y - 8, split_x - 3, handle_y + 8)
                painter.drawLine(split_x + 3, handle_y - 8, split_x + 3, handle_y + 8)
                
    def mousePressEvent(self, event):
        if self.mode != "split" or self.original_frame is None:
            return
            
        frame_h, frame_w = self.original_frame.shape[:2]
        rect = self.get_display_rect(frame_w, frame_h)
        split_x = rect.x() + int(rect.width() * self.split_ratio)
        
        # Clicked near split line? Let's check (within 15px radius)
        if abs(event.position().x() - split_x) < 15:
            self.is_dragging_split = True
            
    def mouseMoveEvent(self, event):
        if self.mode != "split" or self.original_frame is None:
            return
            
        frame_h, frame_w = self.original_frame.shape[:2]
        rect = self.get_display_rect(frame_w, frame_h)
        
        # Show pointer cursor when hovering over split line
        split_x = rect.x() + int(rect.width() * self.split_ratio)
        if abs(event.position().x() - split_x) < 15:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
        if self.is_dragging_split:
            mouse_x = event.position().x()
            # Constrain to video boundary
            relative_x = mouse_x - rect.x()
            ratio = relative_x / rect.width()
            self.split_ratio = np.clip(ratio, 0.02, 0.98)
            self.update()
            
    def mouseReleaseEvent(self, event):
        self.is_dragging_split = False
