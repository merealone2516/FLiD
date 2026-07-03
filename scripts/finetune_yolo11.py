import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import BASE, get_device

YOLO_DIR  = BASE / 'yolo_dataset'
MODEL_DIR = BASE / 'yolo_finetuned'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default='yolo11m',
                        help='YOLO11 variant: yolo11n/s/m/l/x (default: yolo11m)')
    parser.add_argument('--epochs',  type=int, default=50)
    parser.add_argument('--imgsz',   type=int, default=1280,
                        help='Input image size (default 1280 for high-res ID docs)')
    parser.add_argument('--batch',   type=int, default=8)
    parser.add_argument('--device',  default='auto')
    args = parser.parse_args()

    from ultralytics import YOLO

    device = str(get_device(args.device)).replace('cuda', '0')  # ultralytics wants '0' not 'cuda'

    data_yaml = YOLO_DIR / 'data.yaml'
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run prepare_yolo_dataset.py first.")
        sys.exit(1)

    model_pt = f'{args.model}.pt'
    print(f"Loading {model_pt} ...")
    model = YOLO(model_pt)

    print(f"Fine-tuning on {data_yaml}")
    print(f"  epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}  device={device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(MODEL_DIR),
        name='field_detector',
        exist_ok=True,
        # Augmentation — moderate, ID docs have fixed layouts
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        degrees=2.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.0,   # ID docs are not horizontally flipped
        mosaic=0.5,
        # Training
        patience=15,
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        verbose=True,
    )

    best_pt = MODEL_DIR / 'field_detector' / 'weights' / 'best.pt'
    print(f"\nBest model saved at: {best_pt}")

    # Quick validation on val set
    print("\nRunning validation with best weights ...")
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(data=str(data_yaml), imgsz=args.imgsz, device=device, verbose=False)
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Per-class mAP50:")
    class_names = ['face', 'name', 'dob', 'doe', 'doi']
    for i, name in enumerate(class_names):
        try:
            print(f"    {name:10s}: {metrics.box.ap50[i]:.4f}")
        except Exception:
            pass

    print("\nDone.")


if __name__ == '__main__':
    main()
