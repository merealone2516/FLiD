from pathlib import Path


BASE = Path('/home/user1/Flid')

# Raw dataset (JPG + JSON annotations)
TRAIN_TEST_DATA = BASE / 'test-train_data'

# Pre-extracted embedding files (written by scripts/extract_embeddings.py)
EMB_DIR          = BASE / 'embeddings'
EFFNET_B0_DIR    = BASE / 'efficientnet_b0'
RESNET50_DIR     = BASE / 'resnet50'
FACE_EMB_PATH    = EMB_DIR / 'Face_attack.json'
TEXT_EMB_PATH    = EMB_DIR / 'Text_attack.json'
BOTH_EMB_PATH    = EMB_DIR / 'Both_attack.json'

# Baseline
BASELINE_DIR     = BASE / 'gonzalez_tapia_reimpl'
BASELINE_RESULTS = BASELINE_DIR / 'results'

# YOLO11 field detector (fine-tuned, used by scripts/extract_yolo_embeddings.py)
YOLO11_WEIGHTS   = BASE / 'yolo_finetuned' / 'field_detector' / 'weights' / 'best.pt'
YOLO_DATASET_DIR = BASE / 'yolo_dataset'

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
