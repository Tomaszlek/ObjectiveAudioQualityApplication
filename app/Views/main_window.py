import sys
from PyQt6 import QtWidgets, uic
from pathlib import Path


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Wczytujemy interfejs użytkownika z pliku .ui
        # Ścieżka jest budowana względem lokalizacji tego pliku
        ui_path = Path(__file__).parent / "main_window.ui"
        uic.loadUi(ui_path, self)
