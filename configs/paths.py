

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
