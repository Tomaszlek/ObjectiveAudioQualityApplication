import hashlib
import soundfile as sf
from pathlib import Path
from app.Models import audio_tools


class FileGenerationService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_pair(self, audio_data: dict, params: dict) -> dict:
        y = audio_data['data']
        sr = audio_data['samplerate']
        original_path = Path(audio_data['path'])

        # 1. Wycinanie fragmentu
        start_time = params['start_time']
        duration = params['duration']
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)

        # Zabezpieczenie zakresu
        end_sample = min(end_sample, len(y))
        fragment = y[start_sample:end_sample]

        # 2. Normalizacja głośności (zawsze)
        normalized_fragment = audio_tools.normalize_loudness(fragment, sr, target_lufs=-23.0)

        # 3. Kopia do degradacji
        degraded = normalized_fragment.copy()
        is_degraded = False

        # 4. Aplikowanie efektów (jeśli wybrano)
        cutoff_val = 0
        noise_val = 0

        if params['apply_filter']:
            cutoff_val = params['filter_cutoff']
            degraded = audio_tools.lowpass_filter(degraded, sr, cutoff=cutoff_val)
            is_degraded = True

        if params['apply_noise']:
            noise_val = params['noise_level']
            degraded = audio_tools.add_noise(degraded, noise_level=noise_val)
            is_degraded = True

        bitrate = params['bitrate']
        if bitrate != "WAV (bez kompresji)":
            is_degraded = True

        # 5. Generowanie Hasha (unikalna nazwa pliku)
        params_str = f"{original_path.name}{start_sample}{end_sample}{bitrate}{noise_val}{cutoff_val}"
        file_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        # 6. Zapis plików
        ref_frag_name = f"{original_path.stem}_{file_hash}_ref.wav"
        ref_frag_path = self.output_dir / ref_frag_name

        # Zapis REF (zawsze WAV)
        sf.write(str(ref_frag_path), normalized_fragment, sr)

        if is_degraded:
            deg_frag_name = f"{original_path.stem}_{file_hash}_deg.mp3"
            deg_frag_path = self.output_dir / deg_frag_name
            # Zapis DEG (MP3)
            audio_tools.save_mp3(degraded, sr, str(deg_frag_path), bitrate)
        else:
            # Brak degradacji -> Zapisz kopię jako WAV
            deg_frag_name = f"{original_path.stem}_{file_hash}_deg.wav"
            deg_frag_path = self.output_dir / deg_frag_name
            sf.write(str(deg_frag_path), normalized_fragment, sr)
            bitrate = "WAV"

        print(f"Wygenerowano parę: {ref_frag_name} | {deg_frag_name}")

        # 7. Zwracanie danych dla bazy
        return {
            'ref_path': str(ref_frag_path),
            'deg_path': str(deg_frag_path),
            'bitrate': bitrate,
            'noise_level': noise_val if params['apply_noise'] else 0,
            'filter_cutoff': cutoff_val if params['apply_filter'] else 0,
            'status': 'Gotowy do analizy'
        }