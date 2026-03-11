#!/usr/bin/env python3


import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import PAIR_DATA, PAIR_DATA_YOLO

CLASSES = {'face': 0, 'text': 1}


def find_crop_regions(non_crop_img_path, crop_type, dataset_name, crop_base):
    """Use template matching to locate face/text crops in the original image."""
    bboxes = []
    image_stem = non_crop_img_path.stem
    crop_base_path = crop_base / f"{crop_type}_crop" / dataset_name

    if not crop_base_path.exists():
        return bboxes

    original = cv2.imread(str(non_crop_img_path))
    if original is None:
        return bboxes
    h, w = original.shape[:2]

    # Try direct and nested directory structures
    crop_dir = crop_base_path / image_stem
    if not crop_dir.exists():
        for country in crop_base_path.iterdir():
            if country.is_dir():
                candidate = country / image_stem
                if candidate.exists():
                    crop_dir = candidate
                    break

    if not crop_dir.exists():
        return bboxes

    for field, crop_name in [('face', 'face_crop.png'), ('text', 'text_crop.png')]:
        crop_path = crop_dir / crop_name
        if not crop_path.exists():
            continue
        try:
            crop_img = cv2.imread(str(crop_path))
            if crop_img is None:
                continue
            ch, cw = crop_img.shape[:2]
            gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray_orig, gray_crop, cv2.TM_CCOEFF)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > 1e6:
                x, y = max_loc
                xc = max(0, min(1, (x + cw / 2) / w))
                yc = max(0, min(1, (y + ch / 2) / h))
                bw = max(0.01, min(1, cw / w))
                bh = max(0.01, min(1, ch / h))
                bboxes.append((field, xc, yc, bw, bh))
        except Exception as e:
            print(f"  Warning: template match failed for {image_stem}/{field}: {e}")

    return bboxes


def process_attack_type(attack_type, non_crop_base, crop_base, out_dir):
    train_count, val_count = 0, 0
    for cat in ['Real', 'Fake']:
        img_dir = non_crop_base / attack_type / cat
        if not img_dir.exists():
            continue
        imgs = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            imgs.extend(img_dir.glob(ext))

        for img_file in tqdm(imgs, desc=f"{attack_type}/{cat}"):
            bboxes = find_crop_regions(img_file, attack_type, cat, crop_base)
            if not bboxes:
                continue
            split = 'train' if np.random.rand() < 0.8 else 'val'
            shutil.copy2(img_file, out_dir / 'images' / split / img_file.name)
            with open(out_dir / 'labels' / split / f"{img_file.stem}.txt", 'w') as f:
                for cls, xc, yc, bw, bh in bboxes:
                    f.write(f"{CLASSES[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            if split == 'train':
                train_count += 1
            else:
                val_count += 1
    return train_count, val_count


def main():
    parser = argparse.ArgumentParser(description='Generate YOLO annotations')
    parser.add_argument('--output_dir', default=str(PAIR_DATA_YOLO / 'yolo_finetuning_dataset_v2'))
    args = parser.parse_args()

    out = Path(args.output_dir)
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (out / sub).mkdir(parents=True, exist_ok=True)

    total_train, total_val = 0, 0
    for atk in ['Face_attack', 'Text_attack', 'Both_attack']:
        tr, va = process_attack_type(atk, PAIR_DATA, PAIR_DATA, out)
        total_train += tr
        total_val += va

    # data.yaml
    yaml_path = out / 'data.yaml'
    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n\nnc: 2\nnames: ['face', 'text']\n"
    )

    print(f"\nDataset: {total_train} train, {total_val} val")
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
