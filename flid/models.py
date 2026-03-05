"""
FLiD — Model Architectures

Defines the frozen MobileNetV3-Small backbone used for feature extraction
and the lightweight MLP classifiers for each attack scenario.

Architecture:
    Backbone : MobileNetV3-Small (ImageNet-pretrained, FROZEN)  → 576-D
    Face MLP : 576 → 256 → 128 → 64 → 32 → 1   (190 977 params)
    Text MLP : 576 → 256 → 128 → 64 → 32 → 1   (190 977 params)
    Both MLP : 1152 → 512 → 256 → 128 → 64 → 1  (762 881 params)

Total trainable parameters (Face or Text): ~191K
"""

import torch
import torch.nn as nn
from torchvision import models, transforms


# ═══════════════════════════════════════════════════════════════
# Feature extractor (frozen)
# ═══════════════════════════════════════════════════════════════
class MobileNetV3Extractor(nn.Module):
    """MobileNetV3-Small backbone for 576-D embedding extraction."""

    def __init__(self):
        super().__init__()
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.features = m.features
        self.avgpool = m.avgpool
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)  # (B, 576)


# ═══════════════════════════════════════════════════════════════
# MLP classifiers
# ═══════════════════════════════════════════════════════════════
class FaceClassifier(nn.Module):
    """MLP head for face-attack detection (576-D input)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextClassifier(nn.Module):
    """MLP head for text-attack detection (576-D input)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BothClassifier(nn.Module):
    """MLP head for both-attack detection (1152-D concatenated input)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1152, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# Generic MLP builder (used in ablation / fair comparison)
# ═══════════════════════════════════════════════════════════════
def make_mlp(in_dim: int, hidden_dims: list[int] = [256, 128, 64, 32]) -> nn.Sequential:
    """Build a generic MLP with ReLU + Dropout."""
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════
# Alternative backbones (for ablation study)
# ═══════════════════════════════════════════════════════════════
class EfficientNetExtractor(nn.Module):
    """EfficientNet-B0 backbone (1280-D)."""

    def __init__(self):
        super().__init__()
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)  # (B, 1280)


class ResNet18Extractor(nn.Module):
    """ResNet-18 backbone (512-D)."""

    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feat = nn.Sequential(*list(m.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat(x).flatten(1)  # (B, 512)


# ═══════════════════════════════════════════════════════════════
# Standard transforms
# ═══════════════════════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
