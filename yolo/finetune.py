#!/usr/bin/env python3


import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import YOLO_DATASET_YAML, PAIR_DATA_YOLO


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLOv8m for ID field detection')
    parser.add_argument('--data_yaml', type=str, default=str(YOLO_DATASET_YAML),
                        help='Path to YOLO dataset YAML')
    parser.add_argument('--output_dir', type=str,
                        default=str(PAIR_DATA_YOLO / 'yolo_finetuned_models_v2'))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, 0 (GPU), mps')
    args = parser.parse_args()

    from ultralytics import YOLO

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("  YOLOv8m Fine-Tuning for ID Field Detection")
    print("=" * 70)

    model = YOLO('yolov8m.pt')

    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        patience=args.patience,
        save=True,
        device=args.device,
        project=str(output_dir),
        name='id_field_detector_v2',
        pretrained=True,
        optimizer='SGD',
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=15,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True,
    )

    print(f"\nFine-tuning completed. Best model: {output_dir}/id_field_detector_v2/weights/best.pt")


if __name__ == '__main__':
    main()
