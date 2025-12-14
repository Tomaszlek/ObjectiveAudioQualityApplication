from PyQt6.QtCore import QObject, pyqtSignal
from app.Models import audio_tools

class FragmentFinderWorker(QObject):
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)

    def __init__(self, y, sr, duration):
        super().__init__()
        self.y = y
        self.sr = sr
        self.dur = duration

    def run(self):
        try:
            # Algorytm jest teraz w audio_tools, worker tylko go odpala w tle
            res = audio_tools.find_best_fragment(self.y, self.sr, self.dur)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))