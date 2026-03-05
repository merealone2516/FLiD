"""
FLiD — Data Loading Utilities

Functions to load pre-extracted MobileNetV3-Small embeddings for each
attack scenario, as well as raw image paths for YOLO / ablation pipelines.
"""

import json
import re
import numpy as np
from pathlib import Path
from PIL import Image
import torch

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import (PAIR_DATA, PAIR_DATA_YOLO, FACE_EMB_DIR,
                            TEXT_EMB_PATH, BOTH_EMB_DIR, FACE_FULL_EMB)
from flid.models import TRANSFORM


# ═══════════════════════════════════════════════════════════════
# Embedding loaders (pre-extracted .npy / .json)
# ═══════════════════════════════════════════════════════════════

def load_face_embeddings():
    """
    Load augmented face-crop embeddings (document-level naming).

    Returns:
        X:         (N, 576) float32 embedding array.
        y:         (N,) int array  — 0 = Real, 1 = Fake.
        doc_names: list[str] — base document name (augmentation suffix removed).
    """
    X, y, names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = FACE_EMB_DIR / cat
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing directory: {cat_dir}")
        for f in sorted(cat_dir.glob('*.npy')):
            X.append(np.load(f).astype(np.float32).flatten())
            y.append(label)
            names.append(f.stem)
    doc_names = [re.sub(r'_aug\d+$', '', n) for n in names]
    return np.array(X), np.array(y), doc_names


def load_text_embeddings():
    """
    Load patch-based text-crop embeddings from a single JSON file.

    Returns:
        X: (N, 576) float32 embedding array.
        y: (N,) int array.
    """
    with open(TEXT_EMB_PATH) as f:
        data = json.load(f)
    X = np.array([e['embedding'] for e in data], dtype=np.float32)
    y = np.array([int(e['label']) for e in data])
    return X, y


def load_both_embeddings():
    """
    Load concatenated face+text embeddings for both-attack detection.

    Returns:
        X: (N, 1152) float32 embedding array.
        y: (N,) int array.
    """
    X, y = [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = BOTH_EMB_DIR / cat
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing directory: {cat_dir}")
        for f in sorted(cat_dir.glob('*.npy')):
            X.append(np.load(f).astype(np.float32).flatten())
            y.append(label)
    return np.array(X), np.array(y)


def load_full_image_embeddings():
    """
    Load full-image MobileNetV3 embeddings for Face_attack (no ROI).
    Used in ablation study to compare whole-image vs ROI approach.

    Returns:
        X: (N, 576) float32 embedding array.
        y: (N,) int array.
    """
    X, y = [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = FACE_FULL_EMB / cat
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing directory: {cat_dir}")
        for f in sorted(cat_dir.glob('*.npy')):
            X.append(np.load(f).astype(np.float32).flatten())
            y.append(label)
    return np.array(X), np.array(y)


# ═══════════════════════════════════════════════════════════════
# Image-path loaders (for re-extraction with different backbones)
# ═══════════════════════════════════════════════════════════════

SKIP_DIRS = {
    'Mobilenetv3_small', 'data_aug_mobilenet', 'data_aug_efficient',
    'Efficientnetbo', 'patch', 'data_aug',
}


def load_coord_face_images():
    """Coordinate-based face-crop image paths (no augmentation)."""
    paths, labels, doc_names = [], [], []
    face_crop_dir = PAIR_DATA / 'Face_attack_crop'
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = face_crop_dir / cat
        if not cat_dir.exists():
            continue
        for country_dir in sorted(cat_dir.iterdir()):
            if not country_dir.is_dir() or country_dir.name in SKIP_DIRS:
                continue
            for doc_dir in sorted(country_dir.iterdir()):
                if not doc_dir.is_dir():
                    continue
                fc = doc_dir / 'face_crop.png'
                if fc.exists():
                    paths.append(fc)
                    labels.append(label)
                    doc_names.append(doc_dir.name)
    return paths, np.array(labels), doc_names


def load_coord_text_images():
    """Coordinate-based text-crop image paths (no augmentation)."""
    paths, labels, doc_names = [], [], []
    text_crop_dir = PAIR_DATA / 'Text_attack_crop'
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = text_crop_dir / cat
        if not cat_dir.exists():
            continue
        for country_dir in sorted(cat_dir.iterdir()):
            if not country_dir.is_dir() or country_dir.name in SKIP_DIRS:
                continue
            for doc_dir in sorted(country_dir.iterdir()):
                if not doc_dir.is_dir():
                    continue
                tc = doc_dir / 'text_crop.png'
                if tc.exists():
                    paths.append(tc)
                    labels.append(label)
                    doc_names.append(doc_dir.name)
    return paths, np.array(labels), doc_names


def load_yolo_face_images():
    """YOLO-detected face-crop image paths."""
    paths, labels, doc_names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = PAIR_DATA_YOLO / 'face_attack_crop' / cat
        if not cat_dir.exists():
            continue
        for d in sorted(cat_dir.iterdir()):
            if d.is_dir():
                fc = d / 'face_crop.png'
                if fc.exists():
                    paths.append(fc)
                    labels.append(label)
                    doc_names.append(d.name)
    return paths, np.array(labels), doc_names


def load_yolo_text_images():
    """YOLO-detected text-crop image paths."""
    paths, labels, doc_names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = PAIR_DATA_YOLO / 'text_attack' / cat
        if not cat_dir.exists():
            continue
        for d in sorted(cat_dir.iterdir()):
            if d.is_dir():
                tc = d / 'text_crop.png'
                if tc.exists():
                    paths.append(tc)
                    labels.append(label)
                    doc_names.append(d.name)
    return paths, np.array(labels), doc_names


# ═══════════════════════════════════════════════════════════════
# On-the-fly embedding extraction
# ═══════════════════════════════════════════════════════════════

def extract_embeddings_from_images(image_paths, extractor, device=None,
                                    batch_size: int = 32) -> np.ndarray:
    """
    Extract embeddings from a list of image file paths using a frozen backbone.

    Args:
        image_paths: List of Path objects pointing to images.
        extractor:   nn.Module backbone (must be in eval mode on `device`).
        device:      torch.device (defaults to extractor's device).
        batch_size:  Images per forward pass.

    Returns:
        (N, D) float32 numpy array of embeddings.
    """
    if device is None:
        device = next(extractor.parameters()).device

    embeddings = []
    extractor.eval()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(TRANSFORM(img))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            embs = extractor(batch).cpu().numpy()
        embeddings.append(embs)
    return np.concatenate(embeddings, axis=0).astype(np.float32)
