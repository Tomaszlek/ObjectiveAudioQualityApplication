# objective_view.py
from PyQt6 import QtWidgets, uic
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QTableWidgetItem, QMessageBox, QFileDialog, QMenu
import pandas as pd

from app.Workers.worker import AnalysisWorker
from app.Workers.single_file_worker import SingleFileProcessorWorker


class ObjectiveView(QWidget):
    # sygnały dla kontrolera
    results_ready = pyqtSignal(dict)
    pair_added = pyqtSignal(dict)
    clear_requested = pyqtSignal()

    def __init__(self, matlab_engine):
        super().__init__()
        ui_path = Path(__file__).parent / "objective_view.ui"
        uic.loadUi(ui_path, self)

        self.matlab_engine = matlab_engine
        self.output_dir = None
        self.project_data = pd.DataFrame()

        # zmienne - wątki
        self.worker_thread = None
        self.worker = None
        self.auto_thread = None
        self.auto_worker = None
        self.queue = []

        self.setup_ui()
        self.apply_styles()
        self.connect_signals()

    def apply_styles(self):
        # wlasne style do widgetow, w miare je dopasowywalem do stylu dark_teal
        self.setStyleSheet("""
            QLabel { font-size: 10pt; color: #E0E0E0; }
            QGroupBox { font-size: 10pt; color: #E0E0E0; }
            QGroupBox::title { font-weight: bold; color: #FFFFFF; }
            QCheckBox { font-size: 10pt; color: #E0E0E0; }

            QTableWidget { 
                font-size: 10pt; 
                color: #E0E0E0; 
                gridline-color: #555555; 
            }
            QHeaderView::section { 
                font-size: 10pt; 
                font-weight: bold; 
                background-color: #354a48; 
                color: #FFFFFF; 
                padding: 4px; 
            }

            QPushButton { font-size: 10pt; }
            QMenu { 
                background-color: #354a48; 
                color: #ffffff; 
                border: 1px solid #555555; 
            }
            QMenu::item { padding: 5px 20px; }
            QMenu::item:selected { background-color: #4a6664; }
        """)

    def setup_ui(self):
        # Konfiguracja tabeli
        cols = ["Plik referencyjny", "Plik do oceny jakości", "VISQOL", "PEAQ", "CNN 1D", "EfficientNet", "Inception", "VGG19"]
        self.table_results.setColumnCount(len(cols))
        self.table_results.setHorizontalHeaderLabels(cols)
        self.table_results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        # Menu wczytywania
        menu = QMenu(self)
        menu.addAction("Pojedynczy plik").triggered.connect(self.load_single)
        menu.addAction("Wiele plików").triggered.connect(self.load_batch)
        menu.addSeparator()
        menu.addAction("Import parami referencyjny - do oceny").triggered.connect(self.import_pair)
        self.btn_auto_process.setMenu(menu)

        self.btn_exportCsv.setEnabled(False)

    def connect_signals(self):
        self.btn_startAnalysis.clicked.connect(self.start_analysis)
        self.btn_exportCsv.clicked.connect(self.export_results)
        self.btn_clearResults.clicked.connect(self.clear_results)

        # synchronizacja zaznaczenia na listach, tak aby zawsze zaznaczalo odpowiadajace sobie pary
        self.list_ref_files.currentRowChanged.connect(
            lambda r: self.list_deg_files.setCurrentRow(r) if r != -1 else None
        )
        self.list_deg_files.currentRowChanged.connect(
            lambda r: self.list_ref_files.setCurrentRow(r) if r != -1 else None
        )

    def set_data(self, df: pd.DataFrame):
        #jak odbierze dane, to rysuje tabele od zera
        self.project_data = df

        self.list_ref_files.blockSignals(True)
        self.list_deg_files.blockSignals(True)

        self.list_ref_files.clear()
        self.list_deg_files.clear()
        self.table_results.setRowCount(0)

        if df.empty:
            self.list_ref_files.blockSignals(False)
            self.list_deg_files.blockSignals(False)
            self.btn_exportCsv.setEnabled(False)
            return

        for row in df.itertuples():
            # Listy boczne
            self.list_ref_files.addItem(Path(row.ref_path).name)
            self.list_deg_files.addItem(Path(row.deg_path).name)

            # Tabela
            r = self.table_results.rowCount()
            self.table_results.insertRow(r)
            self.table_results.setItem(r, 0, QTableWidgetItem(Path(row.ref_path).name))
            self.table_results.setItem(r, 1, QTableWidgetItem(Path(row.deg_path).name))

            # szybkie formatowanko tabeli
            def fmt(val): return f"{float(val):.4f}" if pd.notna(val) else "-"

            self.table_results.setItem(r, 2, QTableWidgetItem(fmt(getattr(row, 'mos_lqo', None))))
            self.table_results.setItem(r, 3, QTableWidgetItem(fmt(getattr(row, 'odg', None))))
            self.table_results.setItem(r, 4, QTableWidgetItem(fmt(getattr(row, 'cnn_1d_score', None))))
            self.table_results.setItem(r, 5, QTableWidgetItem(fmt(getattr(row, 'efficientnet_score', None))))
            self.table_results.setItem(r, 6, QTableWidgetItem(fmt(getattr(row, 'inception_score', None))))
            self.table_results.setItem(r, 7, QTableWidgetItem(fmt(getattr(row, 'vgg19_score', None))))

        self.list_ref_files.blockSignals(False)
        self.list_deg_files.blockSignals(False)
        self.btn_exportCsv.setEnabled(True)

    def start_analysis(self):
        models = []

        if self.check_visqol.isChecked(): models.append('visqol')
        if self.check_peaq.isChecked(): models.append('peaq')
        if self.check_cnn_1d.isChecked(): models.append('cnn_1d')
        if self.check_efficientnet.isChecked(): models.append('efficientnet')
        if self.check_inception.isChecked(): models.append('inception')
        if self.check_vgg19.isChecked(): models.append('vgg19')

        if not models:
            QMessageBox.warning(self, "Błąd", "Wybierz min. jeden model.")
            return

        if self.project_data.empty:
            QMessageBox.warning(self, "Błąd", "Brak plików do analizy.")
            return

        files = list(zip(self.project_data['ref_path'], self.project_data['deg_path']))

        self.btn_startAnalysis.setEnabled(False)
        self.progressBar.setMaximum(len(files))
        self.progressBar.setValue(0)

        self.worker_thread = QThread()
        self.worker = AnalysisWorker(self.matlab_engine, files, models)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker.progress.connect(self.update_project_data)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(lambda e: print(f"Błąd workera: {e}"))

        self.worker_thread.start()

    def update_project_data(self, result):
        # wysyla wyniki do bazy, żeby była na bieżąco
        self.results_ready.emit(result)
        self.progressBar.setValue(self.progressBar.value() + 1)

        deg_path = result.get('deg_path')
        if not deg_path:
            return
        # dla danej sciezki wyciagam indeks z tabeli i dopisuje na niej wyniki z metod
        name = Path(deg_path).name
        row_idx = -1
        for r in range(self.table_results.rowCount()):
            if self.table_results.item(r, 1).text() == name:
                row_idx = r
                break

        if row_idx != -1:
            mapping = {
                'mos_lqo': 2, 'odg': 3, 'cnn_1d_score': 4,
                'efficientnet_score': 5, 'inception_score': 6, 'vgg19_score': 7
            }
            for key, col in mapping.items():
                if val := result.get(key):
                    self.table_results.setItem(row_idx, col, QTableWidgetItem(f"{float(val):.4f}"))

    def on_analysis_finished(self):
        self.btn_startAnalysis.setEnabled(True)
        QMessageBox.information(self, "Info", "Analiza zakończona.")

    def load_single(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Plik", "", "Audio (*.wav *.flac *.mp3 *.aiff)")
        if path:
            self.queue = [path]
            self.process_queue()

    def load_batch(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Pliki", "", "Audio (*.wav *.flac *.mp3 *.aiff)")
        if paths:
            self.queue = list(paths)
            self.process_queue()

    def process_queue(self):
        if not self.queue:
            self.btn_auto_process.setEnabled(True)
            self.btn_auto_process.setText("Wczytaj...")
            return

        self.btn_auto_process.setEnabled(False)
        self.btn_auto_process.setText(f"Zostało: {len(self.queue)}")

        next_file = self.queue.pop(0)

        self.auto_thread = QThread()
        self.auto_worker = SingleFileProcessorWorker(next_file, self.output_dir)
        self.auto_worker.moveToThread(self.auto_thread)

        self.auto_thread.started.connect(self.auto_worker.run)
        self.auto_worker.finished.connect(self.on_file_processed)
        self.auto_worker.error.connect(self.on_file_error)

        self.auto_thread.start()

    def on_file_processed(self, data):
        self.pair_added.emit(data)
        self.auto_thread.quit()
        self.auto_thread.wait()
        self.process_queue()

    def on_file_error(self, err_msg):
        QMessageBox.error(self, "Błąd", err_msg)
        self._cleanup_thread()
        self.process_queue()

    def import_pair(self):
        ref, _ = QFileDialog.getOpenFileName(self, "Ref", "", "Audio (*.wav *.flac *.aiff *.aif *.aalc)")
        if not ref:
            return

        degs, _ = QFileDialog.getOpenFileNames(self, "Degs", "", "Audio (*.wav *.flac *.mp3 *.aiff *.aif *.aalc)")
        if not degs:
            return

        for d in degs:
            self.pair_added.emit({
                'ref_path': ref, 'deg_path': d, 'bitrate': 'Imported',
                'noise_level': 0, 'filter_cutoff': 0, 'status': 'Gotowy'
            })
        QMessageBox.information(self, "Info", f"Dodano {len(degs)} par.")

    def export_results(self):
        if self.project_data.empty: return
        path, _ = QFileDialog.getSaveFileName(self, "Eksport", "", "Excel (*.xlsx);;CSV (*.csv)")
        if not path: return

        try:
            df = self.project_data.copy()
            if 'id' in df.columns: df = df.drop(columns=['id'])

            if path.endswith('.xlsx'):
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False, encoding='utf-8-sig')

            QMessageBox.information(self, "Sukces", "Zapisano.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", str(e))

    def clear_results(self):
        ans = QMessageBox.question(self, 'Pytanie', 'Wyczyścić wszystkie wyniki?')
        if ans == QMessageBox.StandardButton.Yes:
            self.clear_requested.emit()

    def stop_worker_thread(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()