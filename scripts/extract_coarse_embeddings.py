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
from configs.paths import BASE, TRAIN_TEST_DATA, EMB_DIR, get_device

CROP_FULL_DIR = BASE / 'crop-full' / 'Text_attack_crop'

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_extractor(backbone: str, device):
    if backbone == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        ext = torch.nn.Sequential(m.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
    elif backbone == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        ext = torch.nn.Sequential(*list(m.children())[:-1], torch.nn.Flatten(1))
    else:
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        ext = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten(1))
    for p in ext.parameters():
        p.requires_grad = False
    return ext.eval().to(device)


@torch.no_grad()
def embed(img: Image.Image, extractor, device) -> np.ndarray:
    t = TRANSFORM(img.convert('RGB')).unsqueeze(0).to(device)
    return extractor(t).squeeze(0).cpu().numpy().astype(np.float32)


def face_id_from_stem(stem: str) -> str:
    if '-' in stem:
        return stem.split('-', 1)[1]
    return stem


def build_split_map() -> dict:
    """Build {(stem, label_name) -> split} from test-train_data file tree."""
    mapping = {}
    base = TRAIN_TEST_DATA / 'Text_attack'
    for label_name in ['Real', 'Fake']:
        for split in ['train', 'test']:
            d = base / label_name / split
            if not d.exists():
                continue
            for img in d.glob('*.jpg'):
                mapping[(img.stem, label_name)] = split
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',   default='auto')
    parser.add_argument('--backbone', default='mobilenet',
                        choices=['mobilenet', 'efficientnet_b0', 'resnet50'],
                        help='Feature extractor backbone')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}  |  Backbone: {args.backbone}  |  Mode: coarse text_crop (Option A)")

    extractor = build_extractor(args.backbone, device)
    split_map = build_split_map()

    results = []
    missing_splits = []

    for label_name, label in [('Real', 0), ('Fake', 1)]:
        label_dir = CROP_FULL_DIR / label_name
        if not label_dir.exists():
            print(f"  WARNING: {label_dir} not found, skipping")
            continue

        crop_paths = sorted(label_dir.rglob('text_crop.png'))
        print(f"\n[{label_name}] found {len(crop_paths)} text_crop.png files")

        for crop_path in crop_paths:
            stem = crop_path.parent.name          # e.g. arabic-01_0076_000-fd5000df_1
            doc_id = face_id_from_stem(stem)      # e.g. 01_0076_000-fd5000df_1

            split = split_map.get((stem, label_name))
            if split is None:
                missing_splits.append((stem, label_name))
                split = 'unknown'

            img = Image.open(crop_path)
            emb = embed(img, extractor, device)

            results.append({
                'embedding': emb.tolist(),
                'label': label,
                'doc_id': doc_id,
                'split': split,
                'stem': stem,
            })

    if missing_splits:
        print(f"\n  WARNING: {len(missing_splits)} docs had no split info:")
        for s in missing_splits[:5]:
            print(f"    {s}")

    real_n = sum(r['label'] == 0 for r in results)
    fake_n = sum(r['label'] == 1 for r in results)
    print(f"\nTotal: {len(results)} documents  ({real_n} Real, {fake_n} Fake)")

    out_dir = EMB_DIR if args.backbone == 'mobilenet' else BASE / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'Text_attack_coarse.json'
    with open(out, 'w') as f:
        json.dump(results, f)
    print(f"Saved → {out}")
    print("Done.")


if __name__ == '__main__':
    main()
