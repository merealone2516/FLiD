from pathlib import Path

# ════════════════════════════════════════════════════════════════
# Root of the Turing workspace
# ════════════════════════════════════════════════════════════════
BASE = Path('/Users/akumar/Downloads/Turing')

# Raw dataset (JPG + JSON annotations)
TRAIN_TEST_DATA = BASE / 'test-train_data'

# Pre-extracted embedding files (written by scripts/extract_embeddings.py)
EMB_DIR          = BASE / 'embeddings'
FACE_EMB_PATH    = EMB_DIR / 'Face_attack.json'
TEXT_EMB_PATH    = EMB_DIR / 'Text_attack.json'
BOTH_EMB_PATH    = EMB_DIR / 'Both_attack.json'

# Legacy paths kept for ablation scripts that still reference them
PAIR_DATA        = BASE / 'pair_data'
PAIR_DATA_YOLO   = BASE / 'pair_data_yolo'
BASELINE_DIR     = BASE / 'gonzalez_tapia_reimpl'
BASELINE_RESULTS = BASELINE_DIR / 'results'
FACE_EMB_DIR     = EMB_DIR / 'face_npy'   # unused — JSON used instead
BOTH_EMB_DIR     = EMB_DIR / 'both_npy'   # unused — JSON used instead
FACE_FULL_EMB    = EMB_DIR / 'face_full_npy'

# YOLO
YOLO_MODEL_PATH   = PAIR_DATA_YOLO / 'yolo_finetuned_models_v2' / 'id_field_detector_v2' / 'weights' / 'best.pt'
YOLO_DATASET_YAML = PAIR_DATA_YOLO / 'yolo_fineturing_dataset_v2' / 'data.yaml'

# Output directories (created automatically)
OUTPUT_DIR       = Path(__file__).resolve().parent.parent / 'outputs'
KFOLD_OUTPUT     = OUTPUT_DIR / 'kfold_results'
ABLATION_OUTPUT  = OUTPUT_DIR / 'ablation_results'
EFFICIENCY_OUTPUT = OUTPUT_DIR / 'efficiency_results'

# ════════════════════════════════════════════════════════════════
# Device selection helper
# ════════════════════════════════════════════════════════════════
import torch

def get_device(preference: str = 'auto') -> torch.device:
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(preference)
