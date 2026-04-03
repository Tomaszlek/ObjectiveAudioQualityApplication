# app_controller.py
import sys
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox
import matlab.engine

from app.Views.main_window import MainWindow
from app.Views.preprocessing.preprocessing_view import PreprocessingView
from app.Views.playback.playback_view import SubjectiveView
from app.Views.objective.objective_view import ObjectiveView

from app.Models.database_manager import DatabaseManager
from app.Services.file_generation_service import FileGenerationService


class AppController:
    def __init__(self, app):
        self.app = app
        self.view = MainWindow()

        # Ustalanie ścieżek do folderu z wynikami
        self.output_dir = Path(__file__).parent.parent.parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        db_path = self.output_dir / "projekt.db"

        try:
            self.db_manager = DatabaseManager(db_path)
        except Exception as e:
            print(f"Błąd krytyczny bazy danych: {e}")
            sys.exit(1)

        self.file_service = FileGenerationService(self.output_dir)

        self.project_data = self.db_manager.get_all_pairs_as_dataframe()

        # Start MATLABa - to może chwilę potrwać
        self.matlab_engine = self.start_matlab()
        if not self.matlab_engine:
            print("Nie udało się uruchomić matlaba.")
            sys.exit(1)

        self.preprocessing_screen = PreprocessingView()
        self.subjective_screen = SubjectiveView()

        # ekran analizy potrzebuje silnika i ścieżki do zapisu tymczasowego
        self.objective_screen = ObjectiveView(self.matlab_engine)
        self.objective_screen.output_dir = self.output_dir

        self.setup_ui()
        self.connect_signals()

        self.refresh_data()

    def run(self):
        self.view.show()
        self.app.exec()

    def start_matlab(self):
        try:
            eng = matlab.engine.start_matlab()
            return eng
        except Exception as e:
            print(f"Błąd uruchamiania silnika MATLAB: {e}")
            return None

    def setup_ui(self):
        #dodawanie na stos
        self.view.stackedWidget.addWidget(self.preprocessing_screen)
        self.view.stackedWidget.addWidget(self.subjective_screen)
        self.view.stackedWidget.addWidget(self.objective_screen)

    def connect_signals(self):
        # przełączanie widoków w menu
        self.view.nav_preprocessing.clicked.connect(lambda: self.view.stackedWidget.setCurrentIndex(0))
        self.view.nav_subjective.clicked.connect(lambda: self.view.stackedWidget.setCurrentIndex(1))
        self.view.nav_objective.clicked.connect(lambda: self.view.stackedWidget.setCurrentIndex(2))

        self.preprocessing_screen.processing_requested.connect(self.handle_preprocessing_request)

        self.objective_screen.pair_added.connect(self.add_pair)
        self.objective_screen.results_ready.connect(self.save_results)
        self.objective_screen.clear_requested.connect(self.clear_results)

        self.view.closeEvent = self.on_close

    def refresh_data(self):
        self.project_data = self.db_manager.get_all_pairs_as_dataframe()
        self.objective_screen.set_data(self.project_data)
        self.subjective_screen.set_data(self.project_data)

    def handle_preprocessing_request(self, audio_data, params):
        # pliki fizycznie na dysku, żeby łatwo było je obsluzyc
        new_pair = self.file_service.generate_pair(audio_data, params)
        self.add_pair(new_pair)

    def add_pair(self, pair_data):
        new_id = self.db_manager.add_pair(pair_data)
        if new_id:
            print(f"Dodano parę: {Path(pair_data['deg_path']).name}")
            QMessageBox.information(self.view, "Sukces",
                                    "Pomyślnie wygenerowano i dodano nową parę plików do analizy.")
            self.refresh_data()
        else:
            print("Błąd dodawania pary (możliwy duplikat).")
            QMessageBox.warning(self.view, "Błąd",
                                "Nie udało się dodać pary. Plik prawdopodobnie już istnieje w bazie.")

    def save_results(self, result_dict):
        deg_path = result_dict.pop('deg_path', None)
        result_dict.pop('ref_file', None)

        if deg_path:
            self.db_manager.update_analysis_results(deg_path, result_dict)
            self.refresh_data()

    def clear_results(self):
        if self.db_manager.clear_analysis_results():
            self.refresh_data()
            QMessageBox.information(self.view, "Sukces", "Wyczyszczono wyniki.")

    def on_close(self, event):
        self.objective_screen.stop_worker_thread()

        if self.db_manager:
            self.db_manager.close()

        if self.matlab_engine:
            self.matlab_engine.quit()

        event.accept()