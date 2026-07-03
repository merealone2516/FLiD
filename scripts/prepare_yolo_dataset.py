import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import BASE, TRAIN_TEST_DATA

CROP_FULL_IMG_DIR = BASE / 'crop-full' / 'Text_attack' / 'Real'
ANNOT_DIR         = TRAIN_TEST_DATA / 'Text_attack' / 'Real'
YOLO_DIR          = BASE / 'yolo_dataset'

CLASS_NAMES = ['face', 'name', 'dob', 'doe', 'doi']

FIELD_TO_CLASS = {
    # face
    'face': 0,
    # name variants
    'name': 1, 'name_en': 1, 'Name': 1, 'name_cn': 1,
    'name_ua': 1, 'name_ru': 1, 'نام': 1, 'name2': 1,
    'name_other': 1, 'alternative': 1, '/الإسم': 1,
    'surname': 1, 'Surname': 1, 'surname_en': 1,
    'surname_ua': 1, 'surname_ru': 1, 'نام خانوادگی': 1,
    # dob variants
    'dob': 2, 'تاريخ الميلاد': 2,
    # doe variants
    'doe': 3, 'صالح لغاية': 3,
    # doi variants
    'doi': 4, 'تاريخ الإصدار': 4,
}


def regions_to_yolo(regions: list, img_w: int, img_h: int) -> list[str]:
    """Convert JSON regions to YOLO label lines."""
    lines = []
    for r in regions:
        fn = str(r.get('region_attributes', {}).get('field_name', '')).strip()
        cls = FIELD_TO_CLASS.get(fn)
        if cls is None:
            continue
        sa = r['shape_attributes']
        if sa.get('name') != 'rect':
            continue
        x, y, w, h = sa['x'], sa['y'], sa['width'], sa['height']
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        # clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        lines.append(f'{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--val-ratio',  type=float, default=0.2)
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect all Real JSON files
    json_files = []
    for split in ['train', 'test']:
        json_files.extend(sorted((ANNOT_DIR / split).glob('*.json')))
    print(f"Found {len(json_files)} Real annotation JSON files")

    # Shuffle and split
    random.shuffle(json_files)
    n_val = max(1, int(len(json_files) * args.val_ratio))
    val_set  = set(jf.stem for jf in json_files[:n_val])
    train_set = set(jf.stem for jf in json_files[n_val:])
    print(f"  Train: {len(train_set)} docs  |  Val: {len(val_set)} docs")

    # Create output dirs
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)

    missing_imgs = []
    skipped_no_labels = []
    stats = {'train': 0, 'val': 0}

    for jf in json_files:
        stem = jf.stem
        split = 'val' if stem in val_set else 'train'

        # Find matching image in crop-full
        img_path = CROP_FULL_IMG_DIR / f'{stem}.jpg'
        if not img_path.exists():
            missing_imgs.append(stem)
            continue

        data = json.load(open(jf))
        regions = data.get('regions', [])

        # Get image dimensions from cropping_info
        ci = data.get('cropping_info', {})
        img_w = ci.get('resulted_cropped_image_width')
        img_h = ci.get('resulted_cropped_image_height')
        if img_w is None or img_h is None:
            # Fall back to reading image dimensions
            from PIL import Image
            img_w, img_h = Image.open(img_path).size

        lines = regions_to_yolo(regions, img_w, img_h)
        if not lines:
            skipped_no_labels.append(stem)
            continue

        # Copy image
        dst_img = YOLO_DIR / 'images' / split / f'{stem}.jpg'
        shutil.copy2(img_path, dst_img)

        # Write label file
        dst_lbl = YOLO_DIR / 'labels' / split / f'{stem}.txt'
        dst_lbl.write_text('\n'.join(lines))

        stats[split] += 1

    # Write data.yaml
    yaml_content = f"""path: {YOLO_DIR}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (YOLO_DIR / 'data.yaml').write_text(yaml_content)

    print(f"\nDataset written to {YOLO_DIR}")
    print(f"  Train images: {stats['train']}")
    print(f"  Val   images: {stats['val']}")
    if missing_imgs:
        print(f"  WARNING: {len(missing_imgs)} docs had no matching image in crop-full")
    if skipped_no_labels:
        print(f"  WARNING: {len(skipped_no_labels)} docs had no mappable field annotations")

    # Label distribution
    print("\nClass distribution in train labels:")
    from collections import Counter
    cls_counter = Counter()
    for lbl in (YOLO_DIR / 'labels' / 'train').glob('*.txt'):
        for line in lbl.read_text().splitlines():
            cls_counter[int(line.split()[0])] += 1
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name:10s}: {cls_counter[i]:4d} boxes")

    print("\nDone.")


if __name__ == '__main__':
    main()
