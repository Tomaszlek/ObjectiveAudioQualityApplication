# app/Models/model_architecture.py

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# --- Model 1: Sieć CNN 1D ---
class CNN1D_Paper(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=11),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.01),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.01),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 78, 256),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Oczekiwane wejście x: [batch, 1, 256, czas]
        x = x.squeeze(1)  # -> [batch, 256, czas]
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class InceptionV3_Paper(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_adapter = nn.Sequential(
            # 1. Konwolucja na pełnym spektrogramie (1 -> 3 kanały)
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=11, stride=1, padding=0),
            # 2. Resize do 299 wewnątrz modelu
            transforms.Resize((299, 299), antialias=True)
        )

        # Włączamy aux_logits=True (zgodnie z treningiem)
        self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_adapter(x)

        # Obsługa różnicy między train/eval dla Inception
        if self.training:
            return self.backbone(x)[0]
        else:
            return self.backbone(x)


class VGG19_Paper(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_adapter = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=11, stride=1, padding=0),
            transforms.Resize((224, 224), antialias=True)
        )

        self.backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.regressor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_adapter(x)
        x = self.backbone(x)
        x = self.regressor_head(x)
        return x

class EfficientNetV2_S_Paper(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_adapter = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=11, stride=1, padding=0),
            transforms.Resize((384, 384), antialias=True)
        )

        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Dostosowanie classifiera
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 12 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_adapter(x)
        x = self.backbone.features(x)
        x = self.backbone.classifier(x)
        return x