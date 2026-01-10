import sys
from PyQt6 import QtWidgets, uic
from pathlib import Path


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        #wczytuje interfejs użytkownika z pliku .ui
        ui_path = Path(__file__).parent / "main_window.ui"
        uic.loadUi(ui_path, self)
