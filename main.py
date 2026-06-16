import sys
import os
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    # Force high-DPI scaling support for high-res monitors
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    
    # Fusion style is standard, clean, and easily stylable via stylesheet
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
