from PyQt6.QtCore import QObject, pyqtSignal
from pathlib import Path
from app.Models.audio_processor import AudioProcessor
from app.Models.pytorch_processor import PyTorchProcessor


class AnalysisWorker(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, matlab_engine, files_list, selected_models):
        super().__init__()
        self.eng = matlab_engine
        self.files = files_list
        self.models = selected_models
        self.is_running = True

    def run(self):
        print("[WORKER] Start analizy...")

        try:
            # Konfiguracja ścieżek
            base = Path(__file__).parent.parent

            # MATLAB Processor
            mat_proc = AudioProcessor(self.eng)
            mat_proc.load_config(base / 'config.ini')

            # PyTorch Processor
            torch_proc = PyTorchProcessor(base.parent / 'models')

            # Podział modeli
            mat_models = [m for m in self.models if m in ['visqol', 'peaq']]
            torch_models = [m for m in self.models if m not in mat_models]

            for ref, deg in self.files:
                if not self.is_running: break

                print(f"[WORKER] Przetwarzam: {Path(deg).name}")
                result = {'deg_path': deg, 'status': 'Zakończono'}

                # 1. Analiza MATLAB
                if mat_models:
                    # Funkcja zwraca słownik {'mos_lqo': ..., 'odg': ...}
                    mat_res = mat_proc.analyze_pair(ref, deg)

                    # Filtrujemy tylko to, co chciał użytkownik
                    if 'visqol' in mat_models: result['mos_lqo'] = mat_res.get('mos_lqo')
                    if 'peaq' in mat_models:   result['odg'] = mat_res.get('odg')

                # 2. Analiza PyTorch
                if torch_models:
                    # Funkcja zwraca {'cnn_1d_score': ..., ...}
                    torch_res = torch_proc.analyze(deg, torch_models)
                    result.update(torch_res)

                # Emitujemy wynik do widoku
                self.progress.emit(result)

        except Exception as e:
            print(f"[WORKER] Błąd krytyczny: {e}")
            self.error.emit(str(e))

        self.finished.emit()

    def stop(self):
        self.is_running = False