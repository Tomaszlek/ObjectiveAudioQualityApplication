import sys
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox
import matlab.engine

# Widoki
from app.Views.main_window import MainWindow
from app.Views.preprocessing.preprocessing_view import PreprocessingView
from app.Views.playback.playback_view import SubjectiveView
from app.Views.objective.objective_view import ObjectiveView

# Modele
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

        # Próba połączenia z bazą danych
        try:
            self.db_manager = DatabaseManager(db_path)
        except Exception as e:
            print(f"Błąd krytyczny bazy danych: {e}")
            sys.exit(1)

        # Logika tworzenia plików
        self.file_service = FileGenerationService(self.output_dir)

        # Pobieramy dane na start
        self.project_data = self.db_manager.get_all_pairs_as_dataframe()

        # Start MATLABa - to może chwilę potrwać
        self.matlab_engine = self.start_matlab()
        if not self.matlab_engine:
            print("Nie udało się uruchomić MATLABa. Zamykanie.")
            sys.exit(1)

        # Inicjalizacja ekranów
        self.preprocessing_screen = PreprocessingView()
        self.subjective_screen = SubjectiveView()

        # Ekran analizy potrzebuje silnika i ścieżki do zapisu tymczasowego
        self.objective_screen = ObjectiveView(self.matlab_engine)
        self.objective_screen.output_dir = self.output_dir

        self.setup_ui()
        self.connect_signals()

        # Pierwsze odświeżenie danych
        self.refresh_data()

    def run(self):
        self.view.show()
        self.app.exec()

    def start_matlab(self):
        # Próba uruchomienia silnika Matlaba w tle, dlugo trwa...
        try:
            print("Uruchamianie silnika MATLAB...")
            eng = matlab.engine.start_matlab()
            return eng
        except Exception as e:
            print(f"Błąd silnika MATLAB: {e}")
            return None

    def setup_ui(self):
        # Dodajemy ekrany do stosu
        self.view.stackedWidget.addWidget(self.preprocessing_screen)
        self.view.stackedWidget.addWidget(self.subjective_screen)
        self.view.stackedWidget.addWidget(self.objective_screen)

    def connect_signals(self):
        # Przełączanie widoków w menu
        self.view.nav_preprocessing.clicked.connect(lambda: self.view.stackedWidget.setCurrentIndex(0))
        self.view.nav_subjective.clicked.connect(lambda: self.view.stackedWidget.setCurrentIndex(1))
        self.view.nav_objective.clicked.connect(lambda: self.view.stackedWidget.setCurrentIndex(2))

        # Komunikacja: Widok -> Kontroler
        self.preprocessing_screen.processing_requested.connect(self.handle_preprocessing_request)

        # Komunikacja z ekranu analizy
        self.objective_screen.pair_added.connect(self.add_pair)
        self.objective_screen.results_ready.connect(self.save_results)
        self.objective_screen.clear_requested.connect(self.clear_results)

        self.view.closeEvent = self.on_close

    def refresh_data(self):
        # Pobieramy świeże dane i wysyłamy do widoków
        self.project_data = self.db_manager.get_all_pairs_as_dataframe()
        self.objective_screen.set_data(self.project_data)
        self.subjective_screen.set_data(self.project_data)

    def handle_preprocessing_request(self, audio_data, params):
        # Tworzymy pliki fizycznie na dysku, żeby łatwo było je obsluzyc
        new_pair = self.file_service.generate_pair(audio_data, params)
        # Dodajemy wpis do bazy
        self.add_pair(new_pair)

    def add_pair(self, pair_data):
        new_id = self.db_manager.add_pair(pair_data)
        if new_id:
            print(f"Dodano parę: {Path(pair_data['deg_path']).name}")
            self.refresh_data()
        else:
            print("Błąd dodawania pary (możliwy duplikat).")

    def save_results(self, result_dict):
        # Wyciągamy ścieżkę, bo ona jest kluczem najwazniejszym, a nie id
        deg_path = result_dict.pop('deg_path', None)
        # Usuwamy zbędne klucze jeśli istnieją
        result_dict.pop('ref_file', None)

        if deg_path:
            self.db_manager.update_analysis_results(deg_path, result_dict)
            self.refresh_data()

    def clear_results(self):
        if self.db_manager.clear_analysis_results():
            self.refresh_data()
            QMessageBox.information(self.view, "Sukces", "Wyczyszczono wyniki.")

    def on_close(self, event):
        print("Zamykanie aplikacji")
        self.objective_screen.stop_worker_thread()

        if self.db_manager:
            self.db_manager.close()

        if self.matlab_engine:
            self.matlab_engine.quit()

        event.accept()