# audio_processor.py
import tempfile
from pathlib import Path
from pydub import AudioSegment
import sys
import os
import configparser
import numpy as np
import scipy.signal
import soundfile as sf
import librosa

# Konfiguracja ffmpeg dla pydub
ffmpeg_dir = Path(sys.executable).parent
os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ["PATH"]


class AudioProcessor:
    def __init__(self, eng, matlab_paths: list = None):
        self.eng = eng
        self.matlab_paths = matlab_paths if matlab_paths else []
        self.temp_files = []

    def load_config(self, config_path: Path):
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')

        root = config_path.parent
        # konwersja na ścieżki absolutne
        self.matlab_paths = [str((root / p).resolve()) for p in config['MATLAB_PATHS'].values()]
        self._configure_engine()

    def _configure_engine(self):
        if not self.eng:
            return

        for path in self.matlab_paths:
            self.eng.addpath(path, nargout=0)
        print(f"skonfigurowano {len(self.matlab_paths)} ścieżek dla matlaba.")

    def analyze_pair(self, ref_file, deg_file):
        if not self.eng:
            return {}

        try:
            # normalizacja glosnosci do -23 LUFS, 48kHz czestotliwosc probkowania, konwersja na mono dla visqol
            ref_wav, deg_wav_raw = self._normalize_audio(ref_file, deg_file)

            # wyrównanie czasowe (wymagane dla PEAQ)
            deg_aligned = self._align_signals(ref_wav, deg_wav_raw)

            mos = self.eng.runVisqolForPair(ref_wav, deg_aligned)
            odg, _ = self.eng.PEAQTest(ref_wav, deg_aligned, nargout=2)

            return {
                'mos_lqo': float(mos),
                'odg': float(odg)
            }
        except Exception as e:
            print(f"Błąd analizy w Matlabie: {e}")
            return {}
        finally:
            self.cleanup_temp_files()

    def _align_signals(self, ref_path, deg_path):
        y_ref, sr = librosa.load(ref_path, sr=48000, mono=True)
        y_deg, sz = librosa.load(deg_path, sr=48000, mono=True)

        # przesunięcie metodą korelacji (bierzemy max 7s tak jak sa fragmenty)
        n = min(len(y_ref), len(y_deg), 48000 * 30)
        corr = scipy.signal.correlate(y_ref[:n], y_deg[:n], mode='full', method='fft')
        lag = corr.argmax() - (n - 1)

        # Korygujemy przesunięcie
        if lag > 0:
            y_deg = np.concatenate((np.zeros(lag), y_deg))
        elif lag < 0:
            y_deg = y_deg[abs(lag):]

        # Bezwzględne dopasowanie długości do oryginału (wymóg PEAQ)
        if len(y_deg) > len(y_ref):
            y_deg = y_deg[:len(y_ref)]
        elif len(y_deg) < len(y_ref):
            # Uzupełniamy zerami jeśli za krótki
            padding = np.zeros(len(y_ref) - len(y_deg))
            y_deg = np.concatenate((y_deg, padding))

        with tempfile.NamedTemporaryFile(suffix="_aligned.wav", delete=False) as tf:
            out_path = tf.name

        sf.write(out_path, y_deg, sr)
        self.temp_files.append(out_path)

        return out_path

    def _normalize_audio(self, ref_path: str, deg_path: str):
        # Konwersja plików do standardu WAV 48kHz Mono
        TARGET_SR = 48000
        paths = [ref_path, deg_path]
        out_paths = []

        for p in paths:
            seg = AudioSegment.from_file(p)
            needs_export = False

            if seg.frame_rate != TARGET_SR:
                seg = seg.set_frame_rate(TARGET_SR)
                needs_export = True

            if seg.channels > 1:
                seg = seg.set_channels(1)
                needs_export = True

            # jeśli to MP3, musimy rozpakować do wav, matlab tego potrzebuje
            if p.endswith('.mp3'):
                needs_export = True

            if needs_export:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    new_path = tf.name
                seg.export(new_path, format="wav")
                self.temp_files.append(new_path)
                out_paths.append(new_path)
            else:
                out_paths.append(p)

        return out_paths[0], out_paths[1]

    def cleanup_temp_files(self):
        # sprzątanie plików tymczasowych
        for f in self.temp_files:
            try:
                Path(f).unlink(missing_ok=True)
            except:
                pass
        self.temp_files.clear()