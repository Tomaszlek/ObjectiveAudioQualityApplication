import numpy as np
import librosa
import time
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import tempfile
import os
import pyloudnorm as pyln


def add_noise(y, noise_level):
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise


def lowpass_filter(y, sr, cutoff):
    nyq = 0.5 * sr
    # Zabezpieczenie przed nieprawidłowymi wartościami cutoff
    cutoff = min(max(10, cutoff), nyq - 10)
    normal_cutoff = cutoff / nyq
    b, a = butter(13, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, y)


def save_mp3(y, sr, path, bitrate):
    # pydub nie potrafi bezpośrednio czytać NumPy, więc robimy to przez tymczasowy plik WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_wav_path = tmp_file.name

    try:
        sf.write(temp_wav_path, y, sr)
        audio = AudioSegment.from_wav(temp_wav_path)
        audio.export(path, format='mp3', bitrate=bitrate)
    finally:
        # Zawsze usuwamy plik tymczasowy
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def spectral_entropy(fragment):
    # Sprawdzenie, czy fragment nie jest pusty lub zbyt krótki
    if fragment.shape[0] < 2048:  # Domyślny rozmiar ramki STFT
        return 0.0

    S = np.abs(librosa.stft(fragment))

    s_sum = np.sum(S)
    if s_sum == 0:     # Uniknięcie dzielenia przez zero, jeśli sygnał jest ciszą
        return 0.0

    ps = S / s_sum
    # Dodanie małej stałej (epsilon) dla stabilności numerycznej logarytmu
    return float(-np.sum(ps * np.log2(ps + 1e-12)))


def find_best_fragment(y, sr, duration_seconds):
    samples_per_fragment = int(duration_seconds * sr)

    if len(y) < samples_per_fragment:
        print("Plik za krótki, bierzemy cały do oceny.")
        return 0, len(y)

    max_score = -1.0
    best_start_sample = 0

    step = sr # Krok co sekundę tak jak widać
    num_steps = (len(y) - samples_per_fragment) // step
    print(f"Rozpoczynam pętlę wyszukiwania ({num_steps} kroków).")

    start_time = time.time()

    # Używamy prostej pętli while dla lepszej kontroli
    current_start = 0
    while current_start <= len(y) - samples_per_fragment:
        fragment = y[current_start: current_start + samples_per_fragment]
        score = spectral_entropy(fragment)

        if score > max_score:
            max_score = score
            best_start_sample = current_start

        current_start += step

    end_time = time.time()
    print(f"Pętla zakończona w {end_time - start_time:.2f}s. Najlepszy wynik: {max_score:.2f}")

    return best_start_sample, best_start_sample + samples_per_fragment


def normalize_loudness(y, sr, target_lufs=-23.0):
    try:
        # Pomiar początkowej głośności
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)

        # Obliczenie i zastosowanie wzmocnienia
        normalized_audio = pyln.normalize.loudness(y, loudness, target_lufs)
        print(f"Znormalizowano głośność z {loudness:.2f} LUFS do {target_lufs:.2f} LUFS.")
        return normalized_audio
    except Exception as e:
        print(f"Błąd podczas normalizacji głośności: {e}. Zwracam oryginalny sygnał.")
        return y