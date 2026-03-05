"""
FLiD — Path Configuration

All dataset and model paths are configured here. Update BASE to point
to the root directory of your data. The expected directory layout is
documented below.

Expected directory structure:
    BASE/
    ├── pair_data/
    │   ├── Face_attack/
    │   │   ├── Real/           # Bona fide face-crop images
    │   │   └── Fake/           # Forged face-crop images
    │   ├── Face_attack_crop/
    │   │   ├── Real/<country>/<doc_id>/face_crop.png
    │   │   └── Fake/<country>/<doc_id>/face_crop.png
    │   │       └── data_aug/face_attack_crop_embeddings_aug/
    │   │           ├── Real/<name>.npy
    │   │           └── Fake/<name>.npy
    │   ├── Text_attack/
    │   │   ├── Real/
    │   │   └── Fake/
    │   ├── Text_attack_crop/
    │   │   ├── Real/<country>/<doc_id>/text_crop.png
    │   │   └── Fake/<country>/<doc_id>/text_crop.png
    │   │       └── patch/text_attack_crop_embeddings_patch/embeddings.json
    │   ├── Both_attack/
    │   │   ├── Real/
    │   │   └── Fake/
    │   └── Both_attack_crop/
    │       └── Mobilenetv3_small/Both_attack_crop_embeddings/
    │           ├── Real/<name>.npy
    │           └── Fake/<name>.npy
    ├── pair_data_yolo/
    │   ├── yolo_finetuned_models_v2/id_field_detector_v2/weights/best.pt
    │   ├── yolo_finetuning_dataset_v2/data.yaml
    │   ├── face_attack_crop/
    │   │   ├── Real/<doc_id>/face_crop.png
    │   │   └── Fake/<doc_id>/face_crop.png
    │   └── text_attack/
    │       ├── Real/<doc_id>/text_crop.png
    │       └── Fake/<doc_id>/text_crop.png
    ├── test-train_data/
    │   ├── Face_attack/
    │   │   ├── Real/{train,test}/<image>.png
    │   │   └── Fake/{train,test}/<image>.png
    │   ├── Text_attack/
    │   │   ├── Real/{train,test}/<image>.png
    │   │   └── Fake/{train,test}/<image>.png
    │   └── Both_attack/
    │       ├── Real/{train,test}/<image>.png
    │       └── Fake/{train,test}/<image>.png
    └── gonzalez_tapia_reimpl/
        ├── results/                   # Baseline model checkpoints
        └── results_pretrained/        # Pretrained baseline checkpoints
"""

from pathlib import Path

# ════════════════════════════════════════════════════════════════
# ▸ Update BASE to point to your data root
# ════════════════════════════════════════════════════════════════
BASE = Path('/path/to/your/data')   # <-- CHANGE THIS

# Derived paths (no changes needed below this line)
PAIR_DATA        = BASE / 'pair_data'
PAIR_DATA_YOLO   = BASE / 'pair_data_yolo'
TRAIN_TEST_DATA  = BASE / 'test-train_data'
BASELINE_DIR     = BASE / 'gonzalez_tapia_reimpl'
BASELINE_RESULTS = BASELINE_DIR / 'results'

# FLiD embedding directories
FACE_EMB_DIR  = PAIR_DATA / 'Face_attack_crop' / 'data_aug' / 'face_attack_crop_embeddings_aug'
TEXT_EMB_PATH  = PAIR_DATA / 'Text_attack_crop' / 'patch' / 'text_attack_crop_embeddings_patch' / 'embeddings.json'
BOTH_EMB_DIR  = PAIR_DATA / 'Both_attack_crop' / 'Mobilenetv3_small' / 'Both_attack_crop_embeddings'
FACE_FULL_EMB  = PAIR_DATA / 'Face_attack' / 'Mobilenetv3_small' / 'face_attack_embeddings'

# YOLO
YOLO_MODEL_PATH  = PAIR_DATA_YOLO / 'yolo_finetuned_models_v2' / 'id_field_detector_v2' / 'weights' / 'best.pt'
YOLO_DATASET_YAML = PAIR_DATA_YOLO / 'yolo_finetuning_dataset_v2' / 'data.yaml'

# Output directories (created automatically)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
KFOLD_OUTPUT = OUTPUT_DIR / 'kfold_results'
ABLATION_OUTPUT = OUTPUT_DIR / 'ablation_results'
EFFICIENCY_OUTPUT = OUTPUT_DIR / 'efficiency_results'

# ════════════════════════════════════════════════════════════════
# Device selection helper
# ════════════════════════════════════════════════════════════════
import torch

def get_device(preference: str = 'auto') -> torch.device:
    """Select compute device.

    Args:
        preference: 'auto', 'cpu', 'mps', or 'cuda'.
    """
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(preference)
