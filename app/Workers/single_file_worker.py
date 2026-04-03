# single_file_worker.py
from PyQt6.QtCore import QObject, pyqtSignal
from pathlib import Path
import librosa
import soundfile as sf
import hashlib
from app.Models import audio_tools


class SingleFileProcessorWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_path, output_dir):
        super().__init__()
        self.path = file_path
        self.out_dir = output_dir

    def run(self):
        try:
            print(f"Przetwarzanie: {Path(self.path).name}")
            y, sr = librosa.load(self.path, sr=48000, mono=True)

            if len(y) == 0: # weryfikacja dlugosci pliku, jak jest 0, to nie puszczamy dalej
                raise ValueError("Plik audio jest pusty (0 próbek).")

            start, end = audio_tools.find_best_fragment(y, sr, 7.0)

            # wycięcie i normalizacja
            frag = y[start:end]
            norm_frag = audio_tools.normalize_loudness(frag, sr)

            orig_name = Path(self.path).stem
            # hash żeby uniknąć kolizji nazw
            h = hashlib.md5(f"{orig_name}{start}".encode()).hexdigest()[:6]

            ref_path = self.out_dir / f"{orig_name}_{h}_ref.wav"
            deg_path = self.out_dir / f"{orig_name}_{h}_deg.wav"

            # zapis (kopia ref -> deg)
            sf.write(str(ref_path), norm_frag, sr)
            sf.write(str(deg_path), norm_frag, sr)

            self.finished.emit({
                'ref_path': str(ref_path),
                'deg_path': str(deg_path),
                'bitrate': 'WAV',
                'noise_level': 0, 'filter_cutoff': 0,
                'status': 'Gotowy'
            })

        except Exception as e:
            print(f"Błąd: {e}")
            self.error.emit(str(e))