# preprocessing_view.py
from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QFont
from pathlib import Path
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import pyqtSignal, QThread
import pyqtgraph as pg
import numpy as np
import librosa
import sounddevice as sd
import warnings

from app.Workers.find_fragment_worker import FragmentFinderWorker


class PreprocessingView(QtWidgets.QWidget):
    processing_requested = pyqtSignal(dict, dict)  # (audio_data, params)

    def __init__(self):
        super().__init__()
        uic.loadUi(Path(__file__).parent / "preprocessing_view.ui", self)

        self.mainHorizontalLayout.setStretch(0, 1)
        self.mainHorizontalLayout.setStretch(1, 3)

        self.audio_data_cache = {}
        self.current_selection = {'path': None, 'data': None, 'samplerate': None}
        self.finder_thread = None
        self.finder_worker = None

        self.waveform_plot = pg.PlotWidget()
        layout = QtWidgets.QVBoxLayout(self.waveform_container)
        layout.addWidget(self.waveform_plot)
        self.waveform_plot.setBackground(None)

        #konfiguracja wykresu bez zmian
        self.waveform_plot.getPlotItem().getAxis('left').setWidth(60)
        self.waveform_plot.setLabel('left', 'Amplituda')
        self.waveform_plot.setLabel('bottom', 'Czas (s)')

        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.waveform_plot.addItem(self.region, ignoreBounds=True)

        self._connect_signals()

        self.spin_fragment_duration.setValue(7.0)
        self.spin_fragment_duration.setEnabled(False)
        self.spin_fragment_duration.setToolTip("Długość fragmentu jest stała i wynosi 7 sekund.")

        self._style_plot()
        self._apply_styles()

        self.btn_play_full.setEnabled(False)
        self.btn_play_selection.setEnabled(False)
        self.btn_stop_playback_left.setEnabled(False)
        self.btn_stop_playback_right.setEnabled(False)


    def _apply_styles(self):
        style_sheet = """
               QLabel, QCheckBox { font-size: 10pt; color: #E0E0E0; }
               QGroupBox { font-size: 10pt; }
               QGroupBox::title { font-size: 11pt; font-weight: bold; color: #FFFFFF; }
               QPushButton, QComboBox, QDoubleSpinBox, QSpinBox { font-size: 10pt; }
           """
        self.setStyleSheet(style_sheet)

    def _style_plot(self):
        font = QFont("Segoe UI", 10)
        label_style = {'color': '#E0E0E0', 'font-size': '12pt'}

        plot_item = self.waveform_plot.getPlotItem()
        plot_item.getAxis('left').setTickFont(font)
        plot_item.getAxis('left').setTextPen('#E0E0E0')

        plot_item.getAxis('bottom').setTickFont(font)
        plot_item.getAxis('bottom').setTextPen('#E0E0E0')

        plot_item.setLabel('left', 'Amplituda', **label_style)
        plot_item.setLabel('bottom', 'Czas (s)', **label_style)
        plot_item.showGrid(x=True, y=True, alpha=0.2)

    def _connect_signals(self):
        self.btn_load_files.clicked.connect(self.load_files)
        self.media_list.currentItemChanged.connect(self.on_file_selected)
        self.region.sigRegionChanged.connect(self.on_region_changed)

        self.spin_fragment_start.valueChanged.connect(self._on_spinbox_changed)
        self.spin_fragment_duration.valueChanged.connect(self._on_spinbox_changed)

        self.btn_play_full.clicked.connect(self.play_full_audio)
        self.btn_play_selection.clicked.connect(self.play_selected_fragment)
        self.btn_stop_playback_left.clicked.connect(self.stop_playback)
        self.btn_stop_playback_right.clicked.connect(self.stop_playback)

        self.btn_find_best_fragment.clicked.connect(self.find_best_fragment)

        # tutaj zmiana zaszla,  metoda teraz nazywa się on_process_click
        self.btn_process_and_add.clicked.connect(self.on_process_click)

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Wybierz pliki audio", "",
                                                "Pliki Audio (*.wav *.flac *.aiff *.aif)")
        if files:
            self.media_list.addItems(files)
            QMessageBox.information(self, "Sukces", f"Wczytano {len(files)} plików do biblioteki.")

    def on_file_selected(self, current_item, previous_item):
        self.stop_playback()
        if not current_item:
            self.current_selection = {'path': None, 'data': None, 'samplerate': None}
            self.waveform_plot.clear()
            self._set_playback_buttons(False)
            return

        file_path = current_item.text()
        try:
            if file_path not in self.audio_data_cache:
                data, samplerate = librosa.load(file_path, sr=48000, mono=True)
                self.audio_data_cache[file_path] = (data, samplerate)

            data, samplerate = self.audio_data_cache[file_path]
            self.current_selection = {'path': file_path, 'data': data, 'samplerate': samplerate}
            self.plot_waveform(data, samplerate)
            self._set_playback_buttons(True)
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się wczytać pliku: {e}")

    def _set_playback_buttons(self, enabled: bool):
        self.btn_play_full.setEnabled(enabled)
        self.btn_play_selection.setEnabled(enabled)
        self.btn_stop_playback_left.setEnabled(enabled)
        self.btn_stop_playback_right.setEnabled(enabled)

    def plot_waveform(self, data, samplerate):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_points = 50000

            if data.dtype != np.float32:
                data = data.astype(np.float32)

            plot_data = data[::len(data) // max_points] if len(data) > max_points else data
            duration = len(data) / samplerate
            time_axis = np.linspace(0., duration, len(plot_data))

            self.waveform_plot.clear()
            self.waveform_plot.plot(time_axis, plot_data, pen=pg.mkPen('c', width=1.5))
            self.waveform_plot.setLimits(xMin=0, xMax=duration)
            self.waveform_plot.setXRange(0, duration, padding=0)

            min_val, max_val = (np.min(data), np.max(data)) if len(data) > 0 else (-1.0, 1.0)
            self.waveform_plot.setYRange(min_val - 0.1, max_val + 0.1, padding=0)

            self.spin_fragment_start.setMaximum(duration + 1)
            self.spin_fragment_duration.setMaximum(duration + 1)
            self.spin_fragment_end.setMaximum(duration + 1)

            default_duration = 7.0
            end_time = min(default_duration, duration)

            self.region.setRegion([0.0, end_time])
            self.spin_fragment_start.setValue(0.0)
            self.spin_fragment_duration.setValue(end_time)
            self.spin_fragment_end.setValue(end_time)

    def on_region_changed(self):
        start, end = self.region.getRegion()
        duration = end - start

        self.spin_fragment_start.blockSignals(True)
        self.spin_fragment_duration.blockSignals(True)
        self.spin_fragment_end.blockSignals(True)

        self.spin_fragment_start.setValue(start)
        self.spin_fragment_duration.setValue(duration)
        self.spin_fragment_end.setValue(end)

        self.spin_fragment_start.blockSignals(False)
        self.spin_fragment_duration.blockSignals(False)
        self.spin_fragment_end.blockSignals(False)

    def _on_spinbox_changed(self):
        start = self.spin_fragment_start.value()
        duration = self.spin_fragment_duration.value()
        end = start + duration

        self.spin_fragment_end.blockSignals(True)
        self.spin_fragment_end.setValue(end)
        self.spin_fragment_end.blockSignals(False)

        self.region.blockSignals(True)
        self.region.setRegion([start, end])
        self.region.blockSignals(False)

    def stop_playback(self):
        sd.stop()

    def play_full_audio(self):
        if self.current_selection['data'] is not None:
            self.stop_playback()
            sd.play(self.current_selection['data'], self.current_selection['samplerate'])

    def play_selected_fragment(self):
        if self.current_selection['data'] is not None:
            y = self.current_selection['data']
            sr = self.current_selection['samplerate']

            start_time = self.spin_fragment_start.value()
            duration = self.spin_fragment_duration.value()

            start_sample = int(start_time * sr)
            end_sample = int((start_time + duration) * sr)

            if start_sample < len(y) and end_sample <= len(y):
                fragment = y[start_sample:end_sample]
                self.stop_playback()
                sd.play(fragment, sr)

    def find_best_fragment(self):
        if self.current_selection['data'] is None:
            QMessageBox.warning(self, "Brak pliku", "Najpierw wybierz plik z listy.")
            return

        self.btn_find_best_fragment.setEnabled(False)
        self.btn_find_best_fragment.setText("Szukam...")

        y = self.current_selection['data']
        sr = self.current_selection['samplerate']
        duration = self.spin_fragment_duration.value()

        self.finder_thread = QThread()
        self.finder_worker = FragmentFinderWorker(y, sr, duration)

        self.finder_worker.moveToThread(self.finder_thread)

        self.finder_thread.started.connect(self.finder_worker.run)
        self.finder_worker.finished.connect(self.on_finding_finished)
        self.finder_worker.error.connect(self.on_finding_error)
        self.finder_worker.finished.connect(self.finder_thread.quit)
        self.finder_worker.finished.connect(self.finder_worker.deleteLater)
        self.finder_thread.finished.connect(self.finder_thread.deleteLater)

        self.finder_thread.start()

    def on_finding_finished(self, result):
        start_sample, end_sample = result
        sr = self.current_selection['samplerate']

        start_time = start_sample / sr
        end_time = end_sample / sr

        self.region.setRegion([start_time, end_time])
        self.spin_fragment_start.setValue(start_time)
        self.spin_fragment_duration.setValue(end_time - start_time)
        self.spin_fragment_end.setValue(end_time)
        self.btn_find_best_fragment.setText("Znajdź najlepszy automatycznie")
        self.btn_find_best_fragment.setEnabled(True)

        QMessageBox.information(self, "Sukces", "Znaleziono i zaznaczono optymalny fragment.")

    def on_finding_error(self, error_message):
        QMessageBox.critical(self, "Błąd", error_message)
        self.btn_find_best_fragment.setText("Znajdź najlepszy automatycznie")
        self.btn_find_best_fragment.setEnabled(True)

    def on_process_click(self):
        if self.current_selection['data'] is None:
            QMessageBox.warning(self, "Brak pliku", "Najpierw wybierz plik z listy.")
            return

        # pakuj parametry w słownik
        params = {
            'start_time': self.spin_fragment_start.value(),
            'duration': self.spin_fragment_duration.value(),
            'bitrate': self.combo_bitrate.currentText(),
            'apply_filter': self.check_filter.isChecked(),
            'filter_cutoff': self.spin_filter_cutoff.value(),
            'apply_noise': self.check_noise.isChecked(),
            'noise_level': self.spin_noise_level.value()
        }

        self.processing_requested.emit(self.current_selection, params)