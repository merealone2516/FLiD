#!/usr/bin/env python3


import argparse
import json
import sys
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import PAIR_DATA, PAIR_DATA_YOLO, YOLO_MODEL_PATH

CLASS_NAMES = {0: 'face', 1: 'text'}


def detect_and_crop_face(model, image_path, conf=0.5):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    results = model.predict(source=image_path, conf=conf, verbose=False)
    faces = []
    if results:
        for box in results[0].boxes:
            if CLASS_NAMES.get(int(box.cls), '') != 'face':
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size > 0:
                faces.append({'image': crop, 'bbox': [x1, y1, x2, y2],
                              'confidence': float(box.conf)})
    return faces or None


def main():
    parser = argparse.ArgumentParser(description='Crop faces with YOLO')
    parser.add_argument('--input_dir', default=str(PAIR_DATA / 'Face_attack'))
    parser.add_argument('--output_dir', default=str(PAIR_DATA_YOLO / 'face_attack_crop'))
    parser.add_argument('--model_path', default=str(YOLO_MODEL_PATH))
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model_path)

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    stats = {'total': 0, 'detected': 0, 'failed': []}

    for cat in ['Real', 'Fake']:
        cat_dir = inp / cat
        if not cat_dir.exists():
            continue
        (out / cat).mkdir(parents=True, exist_ok=True)
        imgs = [f for f in sorted(cat_dir.iterdir())
                if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
                and 'crop' not in f.name.lower()]

        for img_file in tqdm(imgs, desc=f"Face {cat}"):
            stats['total'] += 1
            faces = detect_and_crop_face(model, img_file, args.conf)
            if not faces:
                stats['failed'].append(str(img_file))
                continue
            stats['detected'] += 1
            best = max(faces, key=lambda x: x['confidence'])
            img_out = out / cat / img_file.stem
            img_out.mkdir(exist_ok=True)
            Image.fromarray(best['image']).save(img_out / 'face_crop.png')
            with open(img_out / 'detection_metadata.json', 'w') as f:
                json.dump({'source': str(img_file), 'bbox': best['bbox'],
                           'confidence': best['confidence']}, f, indent=2)

    print(f"\nProcessed {stats['total']} images, detected faces in {stats['detected']}")
    if stats['failed']:
        print(f"Failed: {len(stats['failed'])} images")
    print(f"Output: {out}")


if __name__ == '__main__':
    main()
