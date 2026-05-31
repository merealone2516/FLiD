#!/usr/bin/env python3
"""
Extract MobileNetV3-Small embeddings from the raw test-train_data dataset.

Reads JPG images + paired JSON annotations, crops face and text regions using
bounding boxes from the 'regions' field, runs them through a frozen
MobileNetV3-Small backbone, and writes three embedding JSON files:

    embeddings/Face_attack.json   — one entry per document (face bbox crop)
    embeddings/Text_attack.json   — one entry per text-field patch
    embeddings/Both_attack.json   — one entry per document (concat face+text)

Each entry: {"embedding": [...], "label": 0|1, "doc_id": "<face_id>", "split": "train|test"}

For Both_attack/Fake images that have no JSON annotation, the script finds
the matching Real JSON (same face_id, same document template) and borrows its
bounding boxes. If no match is found, the full image is used for both regions.

Usage:
    python scripts/extract_embeddings.py [--attack all|Face|Text|Both] [--device auto]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import TRAIN_TEST_DATA, EMB_DIR, get_device

# ─── Transform ───────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Backbone ────────────────────────────────────────────────────────────────
def build_extractor(device):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    extractor = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten(1))
    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval().to(device)
    return extractor


@torch.no_grad()
def embed(img: Image.Image, extractor, device) -> np.ndarray:
    t = TRANSFORM(img.convert('RGB')).unsqueeze(0).to(device)
    return extractor(t).squeeze(0).cpu().numpy().astype(np.float32)


# ─── Region helpers ──────────────────────────────────────────────────────────
def _crop(img: Image.Image, bbox: dict) -> Image.Image:
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    iw, ih = img.size
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def get_face_crop(img: Image.Image, regions: list):
    """Return the largest face-bbox crop, or the full image if none found."""
    face_bboxes = [
        r['shape_attributes'] for r in regions
        if str(r.get('region_attributes', {}).get('field_name', '')).lower() == 'face'
        and r['shape_attributes'].get('name') == 'rect'
    ]
    if not face_bboxes:
        return img
    best = max(face_bboxes, key=lambda b: b['width'] * b['height'])
    return _crop(img, best)


def get_text_crops(img: Image.Image, regions: list) -> list:
    """Return list of crops for every non-face text field."""
    crops = []
    for r in regions:
        fn = str(r.get('region_attributes', {}).get('field_name', '')).lower()
        if fn == 'face' or fn == '':
            continue
        if r['shape_attributes'].get('name') != 'rect':
            continue
        crops.append(_crop(img, r['shape_attributes']))
    return crops if crops else [img]


# ─── Doc-ID helpers ──────────────────────────────────────────────────────────
def face_id_from_stem(stem: str) -> str:
    """Extract face_id from filename stem: 'country-face_id' → 'face_id'."""
    if '-' in stem:
        return stem.split('-', 1)[1]
    return stem


def load_json_safe(path: Path):
    try:
        return json.load(open(path))
    except Exception:
        return None


# ─── Per-attack extraction ────────────────────────────────────────────────────
def collect_files(attack: str):
    """Return list of (img_path, json_path_or_None, label, split) for an attack."""
    base = TRAIN_TEST_DATA / attack
    entries = []
    for label_name, label in [('Real', 0), ('Fake', 1)]:
        for split in ['train', 'test']:
            d = base / label_name / split
            if not d.exists():
                continue
            for img_path in sorted(d.glob('*.jpg')):
                json_path = img_path.with_suffix('.json')
                entries.append((
                    img_path,
                    json_path if json_path.exists() else None,
                    label,
                    split,
                ))
    return entries


def build_fallback_regions_map(attack: str) -> dict:
    """Build face_id → regions map from Real JSONs for use when Fake JSON is missing."""
    mapping = {}
    base = TRAIN_TEST_DATA / attack / 'Real'
    for split in ['train', 'test']:
        d = base / split
        if not d.exists():
            continue
        for jf in sorted(d.glob('*.json')):
            data = load_json_safe(jf)
            if data and 'person_info' in data:
                fid = data['person_info'].get('face_id', face_id_from_stem(jf.stem))
                mapping[fid] = data.get('regions', [])
    return mapping


def extract_face(attack_dir: str, extractor, device):
    print(f"\n[Face] Extracting from {attack_dir} ...")
    entries = collect_files(attack_dir)
    fallback = build_fallback_regions_map(attack_dir)
    results = []
    for img_path, json_path, label, split in entries:
        data = load_json_safe(json_path) if json_path else None
        if data and 'person_info' in data:
            doc_id = data['person_info'].get('face_id', face_id_from_stem(img_path.stem))
            regions = data.get('regions', [])
        else:
            doc_id = face_id_from_stem(img_path.stem)
            regions = fallback.get(doc_id, [])

        img = Image.open(img_path)
        crop = get_face_crop(img, regions)
        emb = embed(crop, extractor, device)
        results.append({
            'embedding': emb.tolist(),
            'label': label,
            'doc_id': doc_id,
            'split': split,
            'stem': img_path.stem,
        })
    print(f"  {len(results)} samples  "
          f"({sum(r['label']==0 for r in results)} Real, "
          f"{sum(r['label']==1 for r in results)} Fake)")
    return results


def extract_text(attack_dir: str, extractor, device):
    print(f"\n[Text] Extracting from {attack_dir} ...")
    entries = collect_files(attack_dir)
    fallback = build_fallback_regions_map(attack_dir)
    results = []
    for img_path, json_path, label, split in entries:
        data = load_json_safe(json_path) if json_path else None
        if data and 'person_info' in data:
            doc_id = data['person_info'].get('face_id', face_id_from_stem(img_path.stem))
            regions = data.get('regions', [])
        else:
            doc_id = face_id_from_stem(img_path.stem)
            regions = fallback.get(doc_id, [])

        img = Image.open(img_path)
        crops = get_text_crops(img, regions)
        for i, crop in enumerate(crops):
            emb = embed(crop, extractor, device)
            results.append({
                'embedding': emb.tolist(),
                'label': label,
                'doc_id': doc_id,
                'split': split,
                'stem': img_path.stem,
                'patch_idx': i,
            })
    print(f"  {len(results)} patches from "
          f"{len(entries)} documents  "
          f"({sum(r['label']==0 for r in results)} Real patches, "
          f"{sum(r['label']==1 for r in results)} Fake patches)")
    return results


def extract_both(attack_dir: str, extractor, device):
    print(f"\n[Both] Extracting from {attack_dir} ...")
    entries = collect_files(attack_dir)
    fallback = build_fallback_regions_map(attack_dir)
    results = []
    for img_path, json_path, label, split in entries:
        data = load_json_safe(json_path) if json_path else None
        if data and 'person_info' in data:
            doc_id = data['person_info'].get('face_id', face_id_from_stem(img_path.stem))
            regions = data.get('regions', [])
        else:
            doc_id = face_id_from_stem(img_path.stem)
            regions = fallback.get(doc_id, [])

        img = Image.open(img_path)

        # Face embedding
        face_crop = get_face_crop(img, regions)
        face_emb = embed(face_crop, extractor, device)

        # Text embedding — mean-pool across all text patches
        text_crops = get_text_crops(img, regions)
        text_embs = np.stack([embed(c, extractor, device) for c in text_crops])
        text_emb = text_embs.mean(axis=0)

        both_emb = np.concatenate([face_emb, text_emb])  # (1152,)
        results.append({
            'embedding': both_emb.tolist(),
            'label': label,
            'doc_id': doc_id,
            'split': split,
            'stem': img_path.stem,
        })
    print(f"  {len(results)} samples  "
          f"({sum(r['label']==0 for r in results)} Real, "
          f"{sum(r['label']==1 for r in results)} Fake)")
    return results


# ─── Full-image extraction (no cropping) ─────────────────────────────────────
def extract_full_image(attack_dir: str, extractor, device, is_both=False):
    """Embed the entire document image without any region cropping."""
    print(f"\n[{'Both' if is_both else attack_dir.split('_')[0]}] Full-image from {attack_dir} ...")
    entries = collect_files(attack_dir)
    fallback = build_fallback_regions_map(attack_dir)
    results = []
    for img_path, json_path, label, split in entries:
        data = load_json_safe(json_path) if json_path else None
        if data and 'person_info' in data:
            doc_id = data['person_info'].get('face_id', face_id_from_stem(img_path.stem))
        else:
            doc_id = face_id_from_stem(img_path.stem)

        img = Image.open(img_path)
        emb = embed(img, extractor, device)

        # Both classifier expects 1152-D — duplicate the 576-D full-image embedding
        if is_both:
            emb = np.concatenate([emb, emb])

        results.append({
            'embedding': emb.tolist(),
            'label': label,
            'doc_id': doc_id,
            'split': split,
            'stem': img_path.stem,
        })
    print(f"  {len(results)} samples  "
          f"({sum(r['label']==0 for r in results)} Real, "
          f"{sum(r['label']==1 for r in results)} Fake)")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='all',
                        choices=['all', 'Face', 'Text', 'Both'])
    parser.add_argument('--full_image', action='store_true',
                        help='Embed full document image without region cropping')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}  |  Mode: {'full-image' if args.full_image else 'roi-crop'}")

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(device)

    attack_dirs = {'Face': 'Face_attack', 'Text': 'Text_attack', 'Both': 'Both_attack'}
    to_run = list(attack_dirs.keys()) if args.attack == 'all' else [args.attack]

    for key in to_run:
        attack_dir = attack_dirs[key]
        if args.full_image:
            results = extract_full_image(attack_dir, extractor, device, is_both=(key == 'Both'))
            suffix = '_full'
        else:
            fn = {'Face': extract_face, 'Text': extract_text, 'Both': extract_both}[key]
            results = fn(attack_dir, extractor, device)
            suffix = ''

        out = EMB_DIR / f'{key}_attack{suffix}.json'
        with open(out, 'w') as f:
            json.dump(results, f)
        print(f"  Saved → {out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
