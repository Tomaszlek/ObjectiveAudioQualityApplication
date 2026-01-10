import torch
import librosa
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from . import model_architecture


class PyTorchProcessor:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}

        # Konfiguracja modeli - pliki i typy normalizacji
        # CNN1D wymaga konkretnego rozmiaru wejścia
        self.configs = {
            'cnn_1d': {
                'file': 'cnn_1d_unipolared.pth',
                'cls': model_architecture.CNN1D_Paper,
                'norm': 'unipolar',
                'resize': (256, 657)
            },
            'efficientnet': {
                'file': 'efficientnet_v2_s_bipolared.pth',
                'cls': model_architecture.EfficientNetV2_S_Paper,
                'norm': 'bipolar'
            },
            'inception': {
                'file': 'inception_v3_unipolared.pth',
                'cls': model_architecture.InceptionV3_Paper,
                'norm': 'unipolar'
            },
            'vgg19': {
                'file': 'vgg19_bipolared.pth',
                'cls': model_architecture.VGG19_Paper,
                'norm': 'bipolar'
            }
        }
        self.load_models()

    def load_models(self):
        print(f"Używam urządzenia: {self.device}")

        for name, cfg in self.configs.items():
            path = self.models_dir / cfg['file']

            if path.exists():
                try:
                    # Inicjalizacja i ładowanie wag
                    model = cfg['cls']()
                    model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
                    model.to(self.device)
                    model.eval()

                    self.models[name] = model
                    print(f"Załadowano model: {name}")
                except Exception as e:
                    print(f"Błąd ładowania {name}: {e}")
            else:
                print(f"Brak pliku modelu: {path.name}")

    def get_tensor(self, y, sr, norm_type):
        # Generowanie Mel-spektrogramu
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=256, n_mels=256)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Normalizacja Min-Max do zakresu [0, 1]
        # Przyjmujemy min_db = -80
        S_db = np.clip(S_db, -80.0, 0.0)
        S_db = (S_db + 80.0) / 80.0

        # Normalizacja Bipolar [-1, 1]
        if norm_type == 'bipolar':
            S_db = (S_db * 2.0) - 1.0

        # Konwersja do Tensora [1, 256, Time]
        tensor = torch.from_numpy(S_db).float().unsqueeze(0)
        return tensor.to(self.device)

    def analyze(self, path, selected_models):
        if not selected_models:
            return {}

        y, sr = librosa.load(path, sr=48000, mono=True)

        results = {}
        tensor_cache = {}  # Cache żeby nie liczyć tego samego tensora 2 razy

        with torch.no_grad():
            for name in selected_models:
                if name not in self.models: continue

                cfg = self.configs[name]
                norm = cfg['norm']

                # Pobranie lub wygenerowanie tensora
                if norm not in tensor_cache:
                    tensor_cache[norm] = self.get_tensor(y, sr, norm)

                input_tensor = tensor_cache[norm]

                # Resize dla modeli które tego wymagają (np. CNN1D)
                if 'resize' in cfg:
                    resizer = transforms.Resize(cfg['resize'], antialias=True)
                    input_tensor = resizer(input_tensor)

                # Dodanie wymiaru batcha [1, 1, H, W]
                input_tensor = input_tensor.unsqueeze(0)

                # Predykcja
                output = self.models[name](input_tensor)
                raw_val = output.item()

                # Skalowanie z (0-1) na skalę MOS (1-5)
                score = (raw_val * 4.0) + 1.0
                score = max(1.0, min(5.0, score))

                results[f"{name}_score"] = score
                print(f"{name}: {score:.4f}")

        return results