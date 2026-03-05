"""
Baseline — MobileNetV2 PAD Model (Gonzalez & Tapia, Pattern Recognition 2025)

Re-implementation of:
    "Forged Presentation Attack Detection for ID Cards on Remote Verification Systems"

Architecture:
    MobileNetV2 trained from scratch (NO ImageNet pretrained weights)
    Input:  448 × 448 × 3
    Output: 2-class softmax (Real / Fake)
    Initialization: Kaiming normal
    Trainable parameters: ~2.55 M
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════

class MobileNetV2PAD(nn.Module):
    """
    MobileNetV2 for Presentation Attack Detection.

    Trained from scratch (no ImageNet pretrained weights) as per the paper.
    Binary classification: Real (bona fide) vs Fake (attack).

    Args:
        num_classes: Number of output classes (default: 2).
        pretrained:  If True, use ImageNet weights (for pretrained variant).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=pretrained)
        num_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        if not pretrained:
            self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities for scoring."""
        return torch.softmax(self.forward(x), dim=1)


# ═══════════════════════════════════════════════════════════════
# Custom augmentation
# ═══════════════════════════════════════════════════════════════

IMG_SIZE = 448


class GaussianNoise:
    """Add Gaussian noise to a tensor image."""

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_train_transforms():
    """Aggressive data augmentation matching the paper."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                scale=(0.85, 1.15), shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        GaussianNoise(std=0.02),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Validation / test transforms — resize and normalize only."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
