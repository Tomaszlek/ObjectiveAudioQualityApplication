import os
import re
import shutil
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import pyloudnorm as pyln
from tqdm import tqdm

# ==============================================================================
# --- KONFIGURACJA ŚCIEŻEK ---
# ==============================================================================

# ZMIANA: Podaj ścieżkę do folderu z TWOIMI RĘCZNIE WYCIĘTYMI, krótkimi plikami
PRECUT_AUDIO_DIR = r"C:\Users\Tomek\Music\Praca_inzynierska\Dataset_Finalny\subjective_test\long_audio"

# Foldery wyjściowe (zostaną utworzone automatycznie)
BASE_OUTPUT_DIR = r"C:\Users\Tomek\Music\Praca_inzynierska\Dataset_Finalny\subjective_test\final_processed_stereo_12s"
REF_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "1_ref_normalized")
DEGRADED_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "2_degraded")
SPECTROGRAMS_DIR = os.path.join(BASE_OUTPUT_DIR, "3_spectrograms")

# ==============================================================================
# --- PARAMETRY PRZETWARZANIA ---
# ==============================================================================

TARGET_LOUDNESS_LUFS = -23.0
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 256

# ==============================================================================
# --- LISTA PLIKÓW DO UTWORZENIA ---
# ==============================================================================

FILENAMES_TO_PROCESS = [
    "05__LACA_32k_szum_0.00000_filtr_14435.8.mp3",
    "05__LACA_32k_szum_0.00509_filtr_9840.5.mp3",
    "05__LACA_128k_szum_0.00000_filtr_0.0.mp3",
    "05__LACA_192k_szum_0.00000_filtr_0.0.mp3",
    "05__LACA_192k_szum_0.00000_filtr_12937.5.mp3",
    "05__LACA_192k_szum_0.00866_filtr_0.0.mp3",
    "05__LACA_320k_szum_0.00699_filtr_0.0.mp3",
    "05__LACA_320k_szum_0.00790_filtr_10180.6.mp3",
    "AndersIlar-Traveller_32k_szum_0.00913_filtr_14333.6.mp3",
    "AndersIlar-Traveller_64k_szum_0.00000_filtr_0.0.mp3",
    "AndersIlar-Traveller_64k_szum_0.00774_filtr_0.0.mp3",
    "AndersIlar-Traveller_128k_szum_0.00000_filtr_0.0.mp3",
    "AndersIlar-Traveller_128k_szum_0.00000_filtr_11496.1.mp3",
    "AndersIlar-Traveller_128k_szum_0.00831_filtr_0.0.mp3",
    "AndersIlar-Traveller_128k_szum_0.00899_filtr_8845.4.mp3",
    "AndersIlar-Traveller_192k_szum_0.00000_filtr_0.0.mp3",
    "AndersIlar-Traveller_192k_szum_0.00818_filtr_0.0.mp3",
    "AndersIlar-Traveller_320k_szum_0.00000_filtr_0.0.mp3",
    "AndersIlar-Traveller_320k_szum_0.00505_filtr_0.0.mp3",
    "AndersIlar-Traveller_320k_szum_0.00541_filtr_3231.3.mp3",
    "BedichSmetana-MVlast-Vltava_32k_szum_0.00785_filtr_3908.2.mp3",
    "BedichSmetana-MVlast-Vltava_64k_szum_0.00000_filtr_0.0.mp3",
    "BedichSmetana-MVlast-Vltava_64k_szum_0.00000_filtr_11558.2.mp3",
    "BedichSmetana-MVlast-Vltava_64k_szum_0.00987_filtr_3090.7.mp3",
    "BedichSmetana-MVlast-Vltava_128k_szum_0.00000_filtr_0.0.mp3",
    "BedichSmetana-MVlast-Vltava_128k_szum_0.00680_filtr_0.0.mp3",
    "BedichSmetana-MVlast-Vltava_192k_szum_0.00000_filtr_0.0.mp3",
    "BedichSmetana-MVlast-Vltava_192k_szum_0.00990_filtr_14142.7.mp3",
    "BedichSmetana-MVlast-Vltava_320k_szum_0.00000_filtr_3621.5.mp3",
    "Boogie_Belgique_-_01_-_Forever_and_Ever_32k_szum_0.00574_filtr_0.0.mp3",
    "Boogie_Belgique_-_01_-_Forever_and_Ever_64k_szum_0.00589_filtr_5554.8.mp3",
    "Boogie_Belgique_-_01_-_Forever_and_Ever_128k_szum_0.00631_filtr_0.0.mp3",
    "Boogie_Belgique_-_01_-_Forever_and_Ever_192k_szum_0.00671_filtr_0.0.mp3",
    "Boogie_Belgique_-_01_-_Forever_and_Ever_320k_szum_0.00000_filtr_14905.4.mp3",
    "Boogie_Belgique_-_01_-_Forever_and_Ever_320k_szum_0.00860_filtr_0.0.mp3",
    "ComingHome_64k_szum_0.00000_filtr_0.0.mp3",
    "ComingHome_64k_szum_0.00000_filtr_7590.2.mp3",
    "ComingHome_64k_szum_0.00748_filtr_6686.0.mp3",
    "ComingHome_128k_szum_0.00000_filtr_0.0.mp3",
    "ComingHome_128k_szum_0.00979_filtr_12969.0.mp3",
    "ComingHome_192k_szum_0.00000_filtr_0.0.mp3",
    "ComingHome_320k_szum_0.00000_filtr_0.0.mp3",
    "ComingHome_320k_szum_0.00000_filtr_14531.6.mp3",
    "ComingHome_320k_szum_0.00859_filtr_12486.4.mp3",
    "ComingHome_320k_szum_0.00959_filtr_0.0.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_64k_szum_0.00000_filtr_0.0.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_64k_szum_0.00000_filtr_12580.2.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_64k_szum_0.00728_filtr_0.0.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_128k_szum_0.00000_filtr_12726.4.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_128k_szum_0.00510_filtr_0.0.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_192k_szum_0.00000_filtr_11958.8.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_192k_szum_0.00904_filtr_6846.0.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_320k_szum_0.00577_filtr_10936.3.mp3",
    "EdvardGrieg-PeerGyntSuiteNo.1Op.46-01-Morning_320k_szum_0.00772_filtr_0.0.mp3",
    "JohannesBrahms-TragicOverture_32k_szum_0.00000_filtr_0.0.mp3",
    "JohannesBrahms-TragicOverture_32k_szum_0.00000_filtr_5249.4.mp3",
    "JohannesBrahms-TragicOverture_64k_szum_0.00000_filtr_0.0.mp3",
    "JohannesBrahms-TragicOverture_64k_szum_0.00000_filtr_12627.4.mp3",
    "JohannesBrahms-TragicOverture_64k_szum_0.00613_filtr_0.0.mp3",
    "JohannesBrahms-TragicOverture_64k_szum_0.00620_filtr_7952.9.mp3",
    "JohannesBrahms-TragicOverture_128k_szum_0.00000_filtr_0.0.mp3",
    "JohannesBrahms-TragicOverture_128k_szum_0.00000_filtr_12705.9.mp3",
    "JohannesBrahms-TragicOverture_128k_szum_0.00518_filtr_5644.3.mp3",
    "JohannesBrahms-TragicOverture_192k_szum_0.00000_filtr_0.0.mp3",
    "JohannesBrahms-TragicOverture_192k_szum_0.00785_filtr_6057.5.mp3",
    "JohannesBrahms-TragicOverture_192k_szum_0.00835_filtr_0.0.mp3",
    "JohannesBrahms-TragicOverture_320k_szum_0.00000_filtr_0.0.mp3",
    "JohannesBrahms-TragicOverture_320k_szum_0.00700_filtr_4627.7.mp3",
    "JohannesBrahms-TragicOverture_320k_szum_0.00874_filtr_8704.7.mp3",
    "Life Crystals_64k_szum_0.00000_filtr_0.0.mp3",
    "Life Crystals_64k_szum_0.00869_filtr_13044.7.mp3",
    "Life Crystals_128k_szum_0.00000_filtr_0.0.mp3",
    "Life Crystals_128k_szum_0.00000_filtr_10035.2.mp3",
    "Life Crystals_192k_szum_0.00000_filtr_0.0.mp3",
    "Life Crystals_320k_szum_0.00000_filtr_0.0.mp3",
    "Life Crystals_320k_szum_0.00000_filtr_5928.6.mp3",
    "Life Crystals_320k_szum_0.00931_filtr_13384.2.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_32k_szum_0.00000_filtr_0.0.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_32k_szum_0.00000_filtr_13852.6.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_128k_szum_0.00000_filtr_11898.8.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_128k_szum_0.00655_filtr_13072.8.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_128k_szum_0.00996_filtr_0.0.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_192k_szum_0.00000_filtr_8837.4.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_192k_szum_0.00717_filtr_10683.6.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_320k_szum_0.00000_filtr_0.0.mp3",
    "Ribbonmouthrabbit_-_02_-_Trust_Your_Instincts_feat._Groove_Cereal_320k_szum_0.00862_filtr_12487.2.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_32k_szum_0.00746_filtr_0.0.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_64k_szum_0.00000_filtr_0.0.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_64k_szum_0.00888_filtr_0.0.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_128k_szum_0.00000_filtr_10017.8.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_128k_szum_0.00538_filtr_9854.1.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_128k_szum_0.00723_filtr_0.0.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_192k_szum_0.00000_filtr_4729.1.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_192k_szum_0.00559_filtr_0.0.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_192k_szum_0.00790_filtr_4500.5.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_320k_szum_0.00000_filtr_11487.1.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_320k_szum_0.00874_filtr_11042.1.mp3",
    "Various (1950s) - This is How it All Began vol 2 (BLP)-cr-08_320k_szum_0.00962_filtr_0.0.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_32k_szum_0.00782_filtr_14902.3.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_64k_szum_0.00000_filtr_7447.9.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_64k_szum_0.00867_filtr_0.0.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_128k_szum_0.00000_filtr_0.0.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_128k_szum_0.00763_filtr_14072.8.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_128k_szum_0.00939_filtr_0.0.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_192k_szum_0.00000_filtr_0.0.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_192k_szum_0.00000_filtr_3038.1.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_320k_szum_0.00000_filtr_0.0.mp3",
    "Various Artists - Creator Jam - Rise of Atlantis (Official Soundtrack) - 05 DoKashiteru - Watching Seabirds Soar (Harmony) (ft. Snowflake)_320k_szum_0.00674_filtr_0.0.mp3",
    "WolfgangAmadeusMozart-MagicFluteOverture_32k_szum_0.00757_filtr_3535.0.mp3",
    "WolfgangAmadeusMozart-MagicFluteOverture_64k_szum_0.00000_filtr_0.0.mp3",
    "WolfgangAmadeusMozart-MagicFluteOverture_64k_szum_0.00946_filtr_0.0.mp3",
    "WolfgangAmadeusMozart-MagicFluteOverture_128k_szum_0.00000_filtr_0.0.mp3",
    "WolfgangAmadeusMozart-MagicFluteOverture_128k_szum_0.00737_filtr_11299.3.mp3",
]


# ==============================================================================
# --- FUNKCJE POMOCNICZE (bez zmian) ---
# ==============================================================================
def add_noise(y, noise_level):
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise


def lowpass_filter(y, sr, cutoff):
    nyq = 0.5 * sr
    cutoff = min(max(10, cutoff), nyq - 10)
    normal_cutoff = cutoff / nyq
    b, a = butter(13, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, y, axis=1)


def save_mp3(y, sr, path, bitrate):
    temp_wav = path.replace('.mp3', '_temp.wav')
    sf.write(temp_wav, y.T, sr)
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(path, format='mp3', bitrate=bitrate)
    os.remove(temp_wav)


def create_and_save_spectrogram(audio_path, output_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
        if y.ndim > 1: y = np.mean(y, axis=0)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, window='hann'
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        np.save(output_path, log_mel_spectrogram)
        return True
    except Exception as e:
        print(f"  BŁĄD spektrogramu dla {audio_path}: {e}")
        return False


def get_params_from_filename(filename):
    pattern = re.compile(r'(.+?)_(\d+k)_szum_([\d\.]+)_filtr_([\d\.]+)\.mp3')
    match = pattern.match(filename)
    if not match:
        raise ValueError(f"Nie można sparsować nazwy pliku: {filename}")

    base_name, bitrate, noise_level_str, cutoff_str = match.groups()
    params = {
        "base_name": base_name,
        "bitrate": bitrate,
        "noise_level": float(noise_level_str),
        "cutoff": float(cutoff_str)
    }
    return params


def find_source_file_path(base_name, source_dir):
    for ext in ['.wav', '.flac', '.aiff', '.aif', '.mp3']:
        potential_path = os.path.join(source_dir, base_name + ext)
        if os.path.exists(potential_path):
            return potential_path
    return None


# ==============================================================================
# --- GŁÓWNA LOGIKA SKRYPTU (Z POPRAWIONĄ LOGIKĄ) ---
# ==============================================================================

def main():
    print("=" * 80)
    print("--- PRZETWARZANIE RĘCZNIE PRZYGOTOWANYCH FRAGMENTÓW (WERSJA STEREO) ---")
    print("=" * 80)

    os.makedirs(REF_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEGRADED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SPECTROGRAMS_DIR, exist_ok=True)

    audio_cache = {}

    print(f"Rozpoczynam przetwarzanie {len(FILENAMES_TO_PROCESS)} plików...")

    for filename_to_create in tqdm(FILENAMES_TO_PROCESS, desc="Generowanie plików"):
        try:
            params = get_params_from_filename(filename_to_create)
            base_name = params['base_name']

            if base_name not in audio_cache:
                source_path = find_source_file_path(base_name, PRECUT_AUDIO_DIR)
                if not source_path:
                    print(f"OSTRZEŻENIE: Nie znaleziono pliku dla '{base_name}' w '{PRECUT_AUDIO_DIR}'.")
                    audio_cache[base_name] = None
                    continue

                y, sr = librosa.load(source_path, sr=SAMPLE_RATE, mono=False, res_type='kaiser_fast')

                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(y.T)
                normalized_y = pyln.normalize.loudness(y.T, loudness, TARGET_LOUDNESS_LUFS).T

                audio_cache[base_name] = normalized_y

                # === LOGIKA PRZENIESIONA TUTAJ ===
                # Generuj plik referencyjny i kotwice tylko RAZ

                # Zapisz plik referencyjny
                ref_filename = f"{base_name}_ref_normalized_stereo.wav"
                ref_path = os.path.join(REF_OUTPUT_DIR, ref_filename)
                sf.write(ref_path, normalized_y.T, SAMPLE_RATE)

                # Generuj pliki kotwic
                anchors_to_create = [
                    {'name': 'anchor35', 'cutoff': 3500},
                    {'name': 'anchor70', 'cutoff': 7000}
                ]
                for anchor in anchors_to_create:
                    anchor_filename = f"{base_name}_{anchor['name']}_stereo.mp3"
                    anchor_path = os.path.join(DEGRADED_OUTPUT_DIR, anchor_filename)
                    anchor_data = lowpass_filter(normalized_y.copy(), SAMPLE_RATE, cutoff=anchor['cutoff'])
                    save_mp3(anchor_data, SAMPLE_RATE, anchor_path, bitrate="320k")

            ref_data = audio_cache.get(base_name)
            if ref_data is None:
                continue

            # Generowanie plików zdegradowanych z listy (ta część jest już poprawna)
            degraded_data = ref_data.copy()
            if params['cutoff'] > 0.0:
                degraded_data = lowpass_filter(degraded_data, SAMPLE_RATE, cutoff=params['cutoff'])
            if params['noise_level'] > 0.0:
                degraded_data = add_noise(degraded_data, noise_level=params['noise_level'])

            base_mp3, ext_mp3 = os.path.splitext(filename_to_create)
            suffixed_mp3_filename = f"{base_mp3}_stereo{ext_mp3}"
            degraded_mp3_path = os.path.join(DEGRADED_OUTPUT_DIR, suffixed_mp3_filename)
            save_mp3(degraded_data, SAMPLE_RATE, degraded_mp3_path, params['bitrate'])

            base_npy, _ = os.path.splitext(suffixed_mp3_filename)
            suffixed_npy_filename = f"{base_npy}.npy"
            spectrogram_output_path = os.path.join(SPECTROGRAMS_DIR, suffixed_npy_filename)
            create_and_save_spectrogram(degraded_mp3_path, spectrogram_output_path)

        except Exception as e:
            print(f"\nBŁĄD przy pliku {filename_to_create}: {e}")

    print("\n" + "=" * 80)
    print("--- ZAKOŃCZONO ---")


if __name__ == '__main__':
    main()