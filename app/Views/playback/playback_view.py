import numpy as np
import librosa
import sounddevice as sd
import pyqtgraph as pg
from pathlib import Path
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTableWidgetItem, QAbstractItemView)
import pandas as pd
import warnings


class SubjectiveView(QWidget):
    def __init__(self):
        super().__init__()

        ui_path = Path(__file__).parent / "playback_view.ui"
        uic.loadUi(ui_path, self)

        self.project_data = pd.DataFrame()

        # Cache audio, żeby nie ładować ciągle tego samego
        self.audio_data_cache = {}
        self.current_ref_data = None
        self.current_deg_data = None

        self.waveform_plot_A = self._setup_plot_widget()
        self.waveform_plot_B = self._setup_plot_widget()

        layout_A = QVBoxLayout(self.waveform_container_A)
        layout_A.setContentsMargins(0, 0, 0, 0)
        layout_A.addWidget(self.waveform_plot_A)

        layout_B = QVBoxLayout(self.waveform_container_B)
        layout_B.setContentsMargins(0, 0, 0, 0)
        layout_B.addWidget(self.waveform_plot_B)

        self._setup_table()
        self._connect_signals()

        # inicjalne odświeżenie (pustą tabelą)
        self.refresh_data()

    def set_data(self, df: pd.DataFrame):
        self.project_data = df
        self.refresh_data()

    def _setup_plot_widget(self) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setBackground(None)
        plot.getPlotItem().getAxis('left').setWidth(50)
        plot.setLabel('left', 'Amplituda')
        plot.setLabel('bottom', 'Czas (s)')
        plot.showGrid(x=True, y=True, alpha=0.3)
        return plot

    def _setup_table(self):
        self.pairs_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.pairs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.pairs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def _connect_signals(self):
        self.pairs_table.currentCellChanged.connect(self.on_pair_selected)

        self.btn_play_A.clicked.connect(lambda: self.play_audio(self.current_ref_data))
        self.btn_stop_A.clicked.connect(self.stop_playback)
        self.btn_play_B.clicked.connect(lambda: self.play_audio(self.current_deg_data))
        self.btn_stop_B.clicked.connect(self.stop_playback)

    def refresh_data(self):
        df = self.project_data

        self.pairs_table.blockSignals(True)
        self.pairs_table.setRowCount(0)

        if df.empty:
            self.pairs_table.blockSignals(False)
            return

        for row in df.itertuples():
            row_pos = self.pairs_table.rowCount()
            self.pairs_table.insertRow(row_pos)
            self.pairs_table.setItem(row_pos, 0, QTableWidgetItem(Path(row.ref_path).name))
            self.pairs_table.setItem(row_pos, 1, QTableWidgetItem(Path(row.deg_path).name))

        self.pairs_table.blockSignals(False)
        if self.pairs_table.rowCount() > 0:
            self.pairs_table.setCurrentCell(0, 0)

    def on_pair_selected(self, currentRow):
        if currentRow < 0:
            return

        self.stop_playback()

        df = self.project_data
        if currentRow >= len(df): return

        ref_path = df.iloc[currentRow]['ref_path']
        deg_path = df.iloc[currentRow]['deg_path']

        # Cache audio loading
        if ref_path in self.audio_data_cache:
            self.current_ref_data = self.audio_data_cache[ref_path]
        else:
            try:
                data, sr = librosa.load(ref_path, sr=48000, mono=True)
                self.current_ref_data = {'data': data, 'samplerate': sr}
                self.audio_data_cache[ref_path] = self.current_ref_data
            except Exception as e:
                print(f"Nie udało się wczytać {ref_path}: {e}")
                self.current_ref_data = None

        if deg_path in self.audio_data_cache:
            self.current_deg_data = self.audio_data_cache[deg_path]
        else:
            try:
                data, sr = librosa.load(deg_path, sr=48000, mono=True)
                self.current_deg_data = {'data': data, 'samplerate': sr}
                self.audio_data_cache[deg_path] = self.current_deg_data
            except Exception as e:
                print(f"Nie udało się wczytać {deg_path}: {e}")
                self.current_deg_data = None

        self.filename_label_A.setText(Path(ref_path).name)
        if self.current_ref_data:
            self.plot_waveform(self.waveform_plot_A, self.current_ref_data['data'], self.current_ref_data['samplerate'])
        else:
            self.waveform_plot_A.clear()

        self.filename_label_B.setText(Path(deg_path).name)
        if self.current_deg_data:
            self.plot_waveform(self.waveform_plot_B, self.current_deg_data['data'], self.current_deg_data['samplerate'])
        else:
            self.waveform_plot_B.clear()

    def plot_waveform(self, plot_widget: pg.PlotWidget, data, samplerate):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            plot_widget.clear()

            # optymalizacja: wybieram 50k maks punktow dla bardzo długich plików
            max_points = 50000
            step = max(1, len(data) // max_points)
            plot_data = data[::step]

            duration = len(data) / samplerate
            time_axis = np.linspace(0., duration, len(plot_data))

            plot_widget.plot(time_axis, plot_data, pen=pg.mkPen('c', width=1.5))
            plot_widget.setLimits(xMin=0, xMax=duration)
            plot_widget.setXRange(0, duration, padding=0)

    def play_audio(self, audio_data):
        if audio_data and audio_data['data'] is not None:
            self.stop_playback()
            sd.play(audio_data['data'], audio_data['samplerate'])

    def stop_playback(self):
        sd.stop()