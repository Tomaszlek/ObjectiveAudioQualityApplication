# main.py
import sys
from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet
from Controllers.app_controller import AppController

def main():
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    controller = AppController(app)
    controller.run()

if __name__ == "__main__":
    main()