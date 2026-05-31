import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import FACE_EMB_PATH, TEXT_EMB_PATH, BOTH_EMB_PATH
from flid.models import TRANSFORM


# ─── Embedding loaders ────────────────────────────────────────────────────────
# All loaders return (X, y, doc_names) where doc_names is a list of base
# document ids (face_id from the raw JSON annotations). This enables a
# leakage-free document-level split in train_kfold.py.

def _load_emb_json(path: Path, expected_dim: int):
    """Load embeddings JSON and return (X, y, doc_names)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}\n"
            "Run:  python scripts/extract_embeddings.py --attack all"
        )
    with open(path) as f:
        data = json.load(f)

    X, y, doc_names = [], [], []
    for e in data:
        emb = np.array(e['embedding'], dtype=np.float32)
        if emb.shape[0] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}-D embedding in {path}, got {emb.shape[0]}. "
                "Re-run extract_embeddings.py."
            )
        X.append(emb)
        y.append(int(e['label']))
        doc_names.append(str(e['doc_id']))

    return np.array(X), np.array(y), doc_names


def load_face_embeddings():
    """
    Load face-crop embeddings (576-D).

    Returns:
        X:         (N, 576) float32
        y:         (N,) int  — 0=Real, 1=Fake
        doc_names: list[str] — face_id per sample
    """
    return _load_emb_json(FACE_EMB_PATH, expected_dim=576)


def load_text_embeddings():
    """
    Load text-patch embeddings (576-D, one entry per text-field crop).

    Returns:
        X:         (N, 576) float32
        y:         (N,) int
        doc_names: list[str] — face_id per patch (many patches share one doc_id)
    """
    return _load_emb_json(TEXT_EMB_PATH, expected_dim=576)


def load_both_embeddings():
    """
    Load concatenated face+text embeddings (1152-D).

    Returns:
        X:         (N, 1152) float32
        y:         (N,) int
        doc_names: list[str] — face_id per sample
    """
    return _load_emb_json(BOTH_EMB_PATH, expected_dim=1152)


def load_full_image_embeddings():
    """Whole-image embeddings for ablation. Falls back to load_face_embeddings()."""
    return load_face_embeddings()


# ─── Image-path loaders (on-the-fly re-extraction / ablations) ───────────────
from configs.paths import TRAIN_TEST_DATA

SKIP_DIRS = {
    'Mobilenetv3_small', 'data_aug_mobilenet', 'data_aug_efficient',
    'Efficientnetbo', 'patch', 'data_aug',
}


def _iter_img_paths(attack_subdir: str):
    """Yield (img_path, label, doc_id) across Real/Fake train+test."""
    base = TRAIN_TEST_DATA / attack_subdir
    for label_name, label in [('Real', 0), ('Fake', 1)]:
        for split in ['train', 'test']:
            d = base / label_name / split
            if not d.exists():
                continue
            for img_path in sorted(d.glob('*.jpg')):
                json_path = img_path.with_suffix('.json')
                if json_path.exists():
                    try:
                        meta = json.load(open(json_path))
                        doc_id = meta.get('person_info', {}).get(
                            'face_id', img_path.stem.split('-', 1)[-1])
                    except Exception:
                        doc_id = img_path.stem.split('-', 1)[-1]
                else:
                    doc_id = img_path.stem.split('-', 1)[-1]
                yield img_path, label, doc_id


def load_coord_face_images():
    paths, labels, doc_names = [], [], []
    for p, lbl, doc in _iter_img_paths('Face_attack'):
        paths.append(p); labels.append(lbl); doc_names.append(doc)
    return paths, np.array(labels), doc_names


def load_coord_text_images():
    paths, labels, doc_names = [], [], []
    for p, lbl, doc in _iter_img_paths('Text_attack'):
        paths.append(p); labels.append(lbl); doc_names.append(doc)
    return paths, np.array(labels), doc_names


def load_yolo_face_images():
    from configs.paths import PAIR_DATA_YOLO
    paths, labels, doc_names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = PAIR_DATA_YOLO / 'face_attack_crop' / cat
        if not cat_dir.exists():
            continue
        for d in sorted(cat_dir.iterdir()):
            if d.is_dir():
                fc = d / 'face_crop.png'
                if fc.exists():
                    paths.append(fc); labels.append(label); doc_names.append(d.name)
    return paths, np.array(labels), doc_names


def load_yolo_text_images():
    from configs.paths import PAIR_DATA_YOLO
    paths, labels, doc_names = [], [], []
    for label, cat in enumerate(['Real', 'Fake']):
        cat_dir = PAIR_DATA_YOLO / 'text_attack' / cat
        if not cat_dir.exists():
            continue
        for d in sorted(cat_dir.iterdir()):
            if d.is_dir():
                tc = d / 'text_crop.png'
                if tc.exists():
                    paths.append(tc); labels.append(label); doc_names.append(d.name)
    return paths, np.array(labels), doc_names


# ─── On-the-fly embedding extraction ────────────────────────────────────────
def extract_embeddings_from_images(image_paths, extractor, device=None,
                                   batch_size: int = 32) -> np.ndarray:
    if device is None:
        device = next(extractor.parameters()).device
    embeddings = []
    extractor.eval()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(TRANSFORM(img))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            embs = extractor(batch).cpu().numpy()
        embeddings.append(embs)
    return np.concatenate(embeddings, axis=0).astype(np.float32)
