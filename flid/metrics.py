"""
FLiD — ISO/IEC 30107-3 Compliant Metrics

Computes EER, BPCER@APCER thresholds (BPCER10, BPCER20, BPCER50, BPCER100),
AUC, accuracy, precision, recall, and F1.

Convention:
    label 0 = Bona fide (Real)
    label 1 = Attack (Fake)
    bf_scores  = P(Real) — higher means more likely bona fide
"""

import numpy as np


def compute_metrics(labels: np.ndarray, bf_scores: np.ndarray,
                    n_thresh: int = 5000) -> dict:
    """
    Compute full ISO/IEC 30107-3 metrics from bona-fide scores.

    Args:
        labels:    Ground truth array  (0 = bona fide, 1 = attack).
        bf_scores: Bona-fide likelihood scores in [0, 1].
        n_thresh:  Number of threshold steps.

    Returns:
        Dictionary with keys: auc, eer, accuracy, f1, bpcer10, bpcer20,
        bpcer50, bpcer100.
    """
    bf  = bf_scores[labels == 0]
    pa  = bf_scores[labels == 1]

    if len(bf) == 0 or len(pa) == 0:
        return {
            'auc': 0.5, 'eer': 50.0, 'accuracy': 50.0, 'f1': 0.0,
            'bpcer10': 100.0, 'bpcer20': 100.0, 'bpcer50': 100.0,
            'bpcer100': 100.0,
        }

    tau   = np.linspace(0, 1, n_thresh)
    apcer = np.array([np.mean(pa > t) for t in tau])   # attack misclassified as bona fide
    bpcer = np.array([np.mean(bf <= t) for t in tau])   # bona fide misclassified as attack

    # ── EER ──
    diff = np.abs(bpcer - apcer)
    eidx = np.argmin(diff)
    eer  = (bpcer[eidx] + apcer[eidx]) / 2 * 100
    eer_tau = tau[eidx]

    # ── AUC ──
    tpr = 1 - bpcer
    fpr = apcer
    si  = np.argsort(fpr)
    auc = abs(np.trapz(tpr[si], fpr[si]))
    auc = max(0.0, min(1.0, auc))

    # ── Accuracy / F1 at EER threshold ──
    preds    = np.where(bf_scores > eer_tau, 0, 1)
    accuracy = np.mean(preds == labels) * 100
    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # ── BPCER@APCER operating points ──
    def bpcer_at_apcer(target_apcer: float) -> float:
        """BPCER (%) when APCER ≤ target_apcer."""
        valid = apcer <= target_apcer
        if not valid.any():
            return 100.0
        valid_bpcer = bpcer.copy()
        valid_bpcer[~valid] = np.inf
        return float(valid_bpcer.min()) * 100

    return {
        'auc':      auc,
        'eer':      eer,
        'accuracy': accuracy,
        'f1':       f1 * 100,
        'bpcer10':  bpcer_at_apcer(0.10),
        'bpcer20':  bpcer_at_apcer(0.05),
        'bpcer50':  bpcer_at_apcer(0.02),
        'bpcer100': bpcer_at_apcer(0.01),
    }


def compute_pad_metrics(labels, scores, num_thresholds: int = 1000) -> dict:
    """
    Full ISO/IEC 30107-3 PAD metrics (matches baseline evaluation format).

    Args:
        labels: Ground truth (0 = bona fide, 1 = attack).
        scores: Bona-fide scores P(Real).
        num_thresholds: Number of thresholds.

    Returns:
        Dictionary with eer, auc, bpcer_ap dict, and threshold info.
    """
    import torch

    labels_t = torch.tensor(labels, dtype=torch.float32) if not isinstance(labels, torch.Tensor) else labels
    scores_t = torch.tensor(scores, dtype=torch.float32) if not isinstance(scores, torch.Tensor) else scores

    bf_scores = scores_t[labels_t == 0]
    pa_scores = scores_t[labels_t == 1]

    if len(bf_scores) == 0 or len(pa_scores) == 0:
        return {"error": "Need both bona fide and attack samples"}

    thresholds = torch.linspace(0, 1, num_thresholds)

    bpcers, apcers = [], []
    for tau in thresholds:
        bpcers.append((bf_scores <= tau).float().mean().item())
        apcers.append((pa_scores > tau).float().mean().item())

    bpcers = torch.tensor(bpcers)
    apcers = torch.tensor(apcers)

    # EER
    diff = torch.abs(bpcers - apcers)
    eer_idx = diff.argmin()
    eer = ((bpcers[eer_idx] + apcers[eer_idx]) / 2).item()
    eer_threshold = thresholds[eer_idx].item()

    # BPCER at various APCER operating points
    bpcer_ap = {}
    for ap_name, ap_target in [('BPCER10', 0.10), ('BPCER20', 0.05),
                                ('BPCER50', 0.02), ('BPCER100', 0.01),
                                ('BPCER200', 0.005), ('BPCER500', 0.002),
                                ('BPCER1000', 0.001), ('BPCER10000', 0.0001)]:
        valid = apcers <= ap_target
        if valid.any():
            valid_bpcers = bpcers.clone()
            valid_bpcers[~valid] = float('inf')
            best_idx = valid_bpcers.argmin()
            bpcer_ap[ap_name] = {
                'bpcer': bpcers[best_idx].item() * 100,
                'apcer_target': ap_target * 100,
                'apcer_actual': apcers[best_idx].item() * 100,
                'threshold': thresholds[best_idx].item(),
            }
        else:
            bpcer_ap[ap_name] = {
                'bpcer': 100.0,
                'apcer_target': ap_target * 100,
                'apcer_actual': None,
                'threshold': None,
            }

    # AUC
    sorted_idx = apcers.argsort()
    sorted_apcer = apcers[sorted_idx]
    sorted_tpr = 1 - bpcers[sorted_idx]
    auc = torch.trapezoid(sorted_tpr, sorted_apcer).item()
    auc = max(0, min(1, auc))

    return {
        'eer': eer * 100,
        'eer_threshold': eer_threshold,
        'apcer_at_eer': apcers[eer_idx].item() * 100,
        'bpcer_at_eer': bpcers[eer_idx].item() * 100,
        'auc': auc,
        'bpcer_ap': bpcer_ap,
    }
