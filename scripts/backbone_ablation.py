import argparse
import json
import warnings
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import (TRAIN_TEST_DATA, EMB_DIR, EFFNET_B0_DIR,
                            RESNET50_DIR, get_device)
from flid.models import MobileNetV3Extractor, EfficientNetExtractor, ResNet50Extractor
from flid.metrics import compute_metrics

SEED = 42
N_FOLDS = 5
N_BOOT = 1000


# ═══════════════════════════════════════════════════════════════
# Backbone registry
# ═══════════════════════════════════════════════════════════════

BACKBONES = [
    {
        'name':          'MobileNetV3-Small',
        'cls':           MobileNetV3Extractor,
        'dim':           576,
        'params':        '2.5M',
        'flops':         '119M',
        'emb_dir':       EMB_DIR,        # already extracted — skip extraction
        'pre_extracted': True,
    },
    {
        'name':          'EfficientNet-B0',
        'cls':           EfficientNetExtractor,
        'dim':           1280,
        'params':        '5.3M',
        'flops':         '390M',
        'emb_dir':       EFFNET_B0_DIR,
        'pre_extracted': False,
    },
    {
        'name':          'ResNet-50',
        'cls':           ResNet50Extractor,
        'dim':           2048,
        'params':        '25.6M',
        'flops':         '4110M',
        'emb_dir':       RESNET50_DIR,
        'pre_extracted': False,
    },
]


# ═══════════════════════════════════════════════════════════════
# Transforms (same as main pipeline)
# ═══════════════════════════════════════════════════════════════

from torchvision import transforms
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════
# Extraction helpers  (mirrors scripts/extract_embeddings.py)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def _embed(img: Image.Image, extractor, device) -> np.ndarray:
    t = TRANSFORM(img.convert('RGB')).unsqueeze(0).to(device)
    return extractor(t).squeeze(0).cpu().numpy().astype(np.float32)


def _crop(img: Image.Image, bbox: dict) -> Image.Image:
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    iw, ih = img.size
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def _face_crop(img, regions):
    bboxes = [
        r['shape_attributes'] for r in regions
        if str(r.get('region_attributes', {}).get('field_name', '')).lower() == 'face'
        and r['shape_attributes'].get('name') == 'rect'
    ]
    if not bboxes:
        return img
    return _crop(img, max(bboxes, key=lambda b: b['width'] * b['height']))


def _text_crops(img, regions, label=None):

    crops = []
    for r in regions:
        fn = str(r.get('region_attributes', {}).get('field_name', '')).lower()
        if fn in ('face', '') or r['shape_attributes'].get('name') != 'rect':
            continue
        provenance = r.get('region_attributes', {}).get('region_provenance', 'original')
        if label == 1 and provenance != 'altered':
            continue
        crops.append(_crop(img, r['shape_attributes']))
    return crops if crops else [img]


def _face_id(stem):
    return stem.split('-', 1)[1] if '-' in stem else stem


def _load_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _collect(attack_dir):
    entries = []
    for label_name, label in [('Real', 0), ('Fake', 1)]:
        for split in ['train', 'test']:
            d = TRAIN_TEST_DATA / attack_dir / label_name / split
            if not d.exists():
                continue
            for img_path in sorted(d.glob('*.jpg')):
                jpath = img_path.with_suffix('.json')
                entries.append((img_path, jpath if jpath.exists() else None, label, split))
    return entries


def _fallback_regions(attack_dir):
    mapping = {}
    for split in ['train', 'test']:
        d = TRAIN_TEST_DATA / attack_dir / 'Real' / split
        if not d.exists():
            continue
        for jf in sorted(d.glob('*.json')):
            data = _load_json(jf)
            if data and 'person_info' in data:
                fid = data['person_info'].get('face_id', _face_id(jf.stem))
                mapping[fid] = data.get('regions', [])
    return mapping


def _extract(attack_dir, extractor, device, full_image=False):
    """Extract embeddings for one attack × mode combination."""
    entries  = _collect(attack_dir)
    fallback = _fallback_regions(attack_dir)
    is_both  = 'Both' in attack_dir
    is_text  = 'Text' in attack_dir
    results  = []

    for img_path, jpath, label, split in entries:
        data = _load_json(jpath) if jpath else None
        if data and 'person_info' in data:
            doc_id  = data['person_info'].get('face_id', _face_id(img_path.stem))
            regions = data.get('regions', [])
        else:
            doc_id  = _face_id(img_path.stem)
            regions = fallback.get(doc_id, [])

        img = Image.open(img_path)

        if full_image:
            emb = _embed(img, extractor, device)
            if is_both:
                emb = np.concatenate([emb, emb])   # duplicate so cascade can split halves
            results.append({'embedding': emb.tolist(), 'label': label,
                            'doc_id': doc_id, 'split': split})

        elif is_text:
            for i, crop in enumerate(_text_crops(img, regions, label=label)):
                emb = _embed(crop, extractor, device)
                results.append({'embedding': emb.tolist(), 'label': label,
                                'doc_id': doc_id, 'split': split, 'patch_idx': i})

        elif is_both:
            face_emb  = _embed(_face_crop(img, regions), extractor, device)
            # No label filter here — Fake Both docs borrow Real JSONs (only
            # 'original' provenance), so filtering would wrongly discard all crops.
            text_crops = _text_crops(img, regions)
            text_emb  = np.stack([_embed(c, extractor, device)
                                   for c in text_crops]).mean(0)
            emb = np.concatenate([face_emb, text_emb])
            results.append({'embedding': emb.tolist(), 'label': label,
                            'doc_id': doc_id, 'split': split})

        else:  # Face
            emb = _embed(_face_crop(img, regions), extractor, device)
            results.append({'embedding': emb.tolist(), 'label': label,
                            'doc_id': doc_id, 'split': split})

    return results


def extract_backbone(cfg, device):
    """Extract all 6 embedding files for one backbone, skipping existing files."""
    emb_dir = cfg['emb_dir']
    emb_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {cfg['name']} weights...")
    extractor = cfg['cls']().to(device).eval()

    for attack_key, attack_dir, full_image in [
        ('Face', 'Face_attack', False),
        ('Text', 'Text_attack', False),
        ('Both', 'Both_attack', False),
        ('Face', 'Face_attack', True),
        ('Text', 'Text_attack', True),
        ('Both', 'Both_attack', True),
    ]:
        suffix   = '_full' if full_image else ''
        out_path = emb_dir / f'{attack_key}_attack{suffix}.json'
        mode_tag = 'full-img' if full_image else 'roi-crop'

        if out_path.exists():
            print(f"    [{attack_key} {mode_tag}] already exists — skipping")
            continue

        print(f"    [{attack_key} {mode_tag}] extracting ...", end=' ', flush=True)
        results = _extract(attack_dir, extractor, device, full_image=full_image)
        with open(out_path, 'w') as f:
            json.dump(results, f)
        n_real = sum(1 for r in results if r['label'] == 0)
        n_fake = sum(1 for r in results if r['label'] == 1)
        print(f"{len(results)} samples ({n_real}R, {n_fake}F) → {out_path.name}")

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def _load(path: Path, expected_dim: int):
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}\n"
            "Run without --cv_only to extract embeddings first."
        )
    with open(path) as f:
        data = json.load(f)
    X, y, docs = [], [], []
    for e in data:
        emb = np.array(e['embedding'], dtype=np.float32)
        if emb.shape[0] != expected_dim:
            raise ValueError(f"Expected {expected_dim}-D, got {emb.shape[0]} in {path.name}")
        X.append(emb)
        y.append(int(e['label']))
        docs.append(str(e['doc_id']))
    return np.array(X), np.array(y), docs
