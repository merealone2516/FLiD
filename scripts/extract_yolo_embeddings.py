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

YOLO_WEIGHTS = BASE / 'yolo_finetuned' / 'field_detector' / 'weights' / 'best.pt'

CLASS_FACE   = 0
TEXT_CLASSES = {1: 'name', 2: 'dob', 3: 'doe', 4: 'doi'}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


BACKBONE_DIMS = {'mobilenet': 576, 'efficientnet_b0': 1280, 'resnet50': 2048}

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
    return stem.split('-', 1)[1] if '-' in stem else stem


def collect_files(attack_dir: str):
    base = TRAIN_TEST_DATA / attack_dir
    entries = []
    for label_name, label in [('Real', 0), ('Fake', 1)]:
        for split in ['train', 'test']:
            d = base / label_name / split
            if not d.exists():
                continue
            for img_path in sorted(d.glob('*.jpg')):
                entries.append((img_path, label, split))
    return entries


def detect(yolo, img_path: str, conf: float, yolo_device: str):
    """Run YOLO and return (boxes_xyxy, class_ids) as numpy arrays."""
    det = yolo.predict(img_path, conf=conf, device=yolo_device, verbose=False)[0]
    boxes  = det.boxes.xyxy.cpu().numpy()
    cls_ids = det.boxes.cls.cpu().numpy().astype(int)
    return boxes, cls_ids


def get_face_crop(img: Image.Image, boxes, cls_ids) -> Image.Image:
    """Return largest YOLO-detected face crop, or full image as fallback."""
    iw, ih = img.size
    best, best_area = None, 0
    for box, cls in zip(boxes, cls_ids):
        if cls != CLASS_FACE:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best, best_area = (x1, y1, x2, y2), area
    return img.crop(best) if best else img


def get_text_crops(img: Image.Image, boxes, cls_ids):
    """Return list of (crop, field_name) for all YOLO-detected text fields."""
    iw, ih = img.size
    crops = []
    for box, cls in zip(boxes, cls_ids):
        if cls not in TEXT_CLASSES:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append((img.crop((x1, y1, x2, y2)), TEXT_CLASSES[cls]))
    return crops if crops else [(img, 'full')]


# ─── Per-attack extractors ────────────────────────────────────────────────────

def extract_face(yolo, extractor, device, yolo_device, conf):
    print("\n[Face] Extracting with YOLO11 ...")
    entries = collect_files('Face_attack')
    results = []
    no_det = 0
    for img_path, label, split in entries:
        stem   = img_path.stem
        doc_id = face_id_from_stem(stem)
        img    = Image.open(img_path).convert('RGB')
        boxes, cls_ids = detect(yolo, str(img_path), conf, yolo_device)
        face_crop = get_face_crop(img, boxes, cls_ids)
        if face_crop is img:
            no_det += 1
        emb = embed(face_crop, extractor, device)
        results.append({'embedding': emb.tolist(), 'label': label,
                        'doc_id': doc_id, 'split': split, 'stem': stem})
    _print_stats(results, entries, no_det, 'face')
    return results


def extract_text(yolo, extractor, device, yolo_device, conf):
    print("\n[Text] Extracting with YOLO11 ...")
    entries = collect_files('Text_attack')
    results = []
    no_det = 0
    for img_path, label, split in entries:
        stem   = img_path.stem
        doc_id = face_id_from_stem(stem)
        img    = Image.open(img_path).convert('RGB')
        boxes, cls_ids = detect(yolo, str(img_path), conf, yolo_device)
        crops = get_text_crops(img, boxes, cls_ids)
        if crops[0][1] == 'full':
            no_det += 1
        for i, (crop, fn) in enumerate(crops):
            emb = embed(crop, extractor, device)
            results.append({'embedding': emb.tolist(), 'label': label,
                            'doc_id': doc_id, 'split': split, 'stem': stem,
                            'patch_idx': i, 'field': fn})
    _print_stats(results, entries, no_det, 'text')
    return results


def extract_both(yolo, extractor, device, yolo_device, conf):
    print("\n[Both] Extracting with YOLO11 ...")
    entries = collect_files('Both_attack')
    results = []
    no_face_det = 0
    no_text_det = 0
    for img_path, label, split in entries:
        stem   = img_path.stem
        doc_id = face_id_from_stem(stem)
        img    = Image.open(img_path).convert('RGB')
        boxes, cls_ids = detect(yolo, str(img_path), conf, yolo_device)

        face_crop = get_face_crop(img, boxes, cls_ids)
        if face_crop is img:
            no_face_det += 1

        text_crops = get_text_crops(img, boxes, cls_ids)
        if text_crops[0][1] == 'full':
            no_text_det += 1

        face_emb  = embed(face_crop, extractor, device)
        text_embs = np.stack([embed(c, extractor, device) for c, _ in text_crops])
        text_emb  = text_embs.mean(axis=0)

        both_emb = np.concatenate([face_emb, text_emb])
        results.append({'embedding': both_emb.tolist(), 'label': label,
                        'doc_id': doc_id, 'split': split, 'stem': stem})

    print(f"  {len(results)} samples  "
          f"({sum(r['label']==0 for r in results)} Real, "
          f"{sum(r['label']==1 for r in results)} Fake)")
    if no_face_det:
        print(f"  WARNING: {no_face_det} docs had no YOLO face detection → used full image")
    if no_text_det:
        print(f"  WARNING: {no_text_det} docs had no YOLO text detection → used full image")
    return results


def _print_stats(results, entries, no_det, kind):
    unique = len(set(r['stem'] for r in results))
    real   = sum(r['label'] == 0 for r in results)
    fake   = sum(r['label'] == 1 for r in results)
    print(f"  {len(results)} patches from {unique} docs  ({real} Real, {fake} Fake)")
    if no_det:
        print(f"  WARNING: {no_det} docs had no YOLO {kind} detection → used full image")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack',   default='all',
                        choices=['all', 'Face', 'Text', 'Both'])
    parser.add_argument('--conf',     type=float, default=0.3)
    parser.add_argument('--device',   default='auto')
    parser.add_argument('--backbone', default='mobilenet',
                        choices=['mobilenet', 'efficientnet_b0', 'resnet50'],
                        help='Feature extractor backbone')
    args = parser.parse_args()

    device = get_device(args.device)
    yolo_device = str(device).replace('cuda', '0')
    print(f"Device: {device}  |  Backbone: {args.backbone}  |  YOLO11 field detection")

    if not YOLO_WEIGHTS.exists():
        print(f"ERROR: YOLO weights not found at {YOLO_WEIGHTS}")
        sys.exit(1)

    from ultralytics import YOLO
    yolo      = YOLO(str(YOLO_WEIGHTS))
    extractor = build_extractor(args.backbone, device)

    out_dir = EMB_DIR if args.backbone == 'mobilenet' else BASE / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    to_run = ['Face', 'Text', 'Both'] if args.attack == 'all' else [args.attack]

    fns = {'Face': extract_face, 'Text': extract_text, 'Both': extract_both}

    for key in to_run:
        results = fns[key](yolo, extractor, device, yolo_device, args.conf)
        out = out_dir / f'{key}_attack_yolo.json'
        with open(out, 'w') as f:
            json.dump(results, f)
        print(f"  Saved → {out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
