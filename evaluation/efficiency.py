#!/usr/bin/env python3
"""
Evaluation — Computational Efficiency Analysis

Compares FLiD vs Baseline in terms of:
    - Parameter count (total & trainable)
    - FLOPs (forward pass)
    - End-to-end CPU latency

Usage:
    python -m evaluation.efficiency
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import EFFICIENCY_OUTPUT, YOLO_MODEL_PATH
from flid.models import (MobileNetV3Extractor, FaceClassifier, TextClassifier,
                          BothClassifier)
from baseline.model import MobileNetV2PAD


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_mlp_flops(mlp, in_dim):
    """Approximate FLOPs for a Sequential MLP (multiply-add per Linear)."""
    flops = 0
    prev = in_dim
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * prev * m.out_features
            prev = m.out_features
    return flops


def count_model_flops_cnn(model, input_shape):
    """Count FLOPs via forward hooks for conv/linear layers."""
    total = [0]
    hooks = []

    def hook_fn(module, inp, out):
        if isinstance(module, nn.Conv2d):
            bs, _, oh, ow = out.shape
            total[0] += 2 * module.in_channels * module.out_channels * \
                        module.kernel_size[0] * module.kernel_size[1] * oh * ow // module.groups
        elif isinstance(module, nn.Linear):
            total[0] += 2 * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape))
    for h in hooks:
        h.remove()
    return total[0]


def measure_latency_cpu(model, dummy, warmup=20, repeats=100):
    model.eval().cpu()
    dummy = dummy.cpu()
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    EFFICIENCY_OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}

    print("=" * 75)
    print("  EFFICIENCY ANALYSIS")
    print("=" * 75)

    # ── FLiD backbone ──
    print("\n  FLiD Backbone: MobileNetV3-Small (frozen)")
    bb = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval()
    total_bb, _ = count_parameters(bb)
    flops_bb = count_model_flops_cnn(bb, (1, 3, 224, 224))
    lat_bb, lat_bb_std = measure_latency_cpu(bb, torch.randn(1, 3, 224, 224))
    print(f"    Params: {total_bb:,}  FLOPs: {flops_bb/1e6:.1f}M  Latency: {lat_bb:.2f}ms")

    results['backbone'] = {
        'params': total_bb, 'flops': flops_bb,
        'latency_ms': round(lat_bb, 2), 'latency_std_ms': round(lat_bb_std, 2),
    }

    # ── FLiD MLPs ──
    for name, cls, dim in [('Face', FaceClassifier, 576),
                            ('Text', TextClassifier, 576),
                            ('Both', BothClassifier, 1152)]:
        m = cls()
        t, tr = count_parameters(m)
        fl = count_mlp_flops(m, dim)
        print(f"    {name} MLP: {t:,} params ({tr:,} trainable), {fl/1e3:.1f}K FLOPs")
        results[f'mlp_{name}'] = {'params': t, 'trainable': tr, 'flops': fl}

    # ── FLiD pipeline totals ──
    face_mlp = FaceClassifier()
    text_mlp = TextClassifier()
    both_mlp = BothClassifier()
    text_n_patches = 4  # average text patches

    face_total_params = total_bb + count_parameters(face_mlp)[0]
    face_trainable = count_parameters(face_mlp)[1]
    face_total_flops = flops_bb + count_mlp_flops(face_mlp, 576)

    text_total_params = total_bb + count_parameters(text_mlp)[0]
    text_trainable = count_parameters(text_mlp)[1]
    text_total_flops = text_n_patches * flops_bb + count_mlp_flops(text_mlp, 576)

    both_total_params = total_bb + count_parameters(both_mlp)[0]
    both_trainable = count_parameters(both_mlp)[1]
    both_total_flops = (1 + text_n_patches) * flops_bb + count_mlp_flops(both_mlp, 1152)

    for pname, tp, tr, fl, passes in [
        ('Face', face_total_params, face_trainable, face_total_flops, 1),
        ('Text', text_total_params, text_trainable, text_total_flops, text_n_patches),
        ('Both', both_total_params, both_trainable, both_total_flops, 1 + text_n_patches),
    ]:
        print(f"\n    FLiD {pname} pipeline ({passes} backbone pass{'es' if passes>1 else ''}):")
        print(f"      Params: {tp:,} (trainable: {tr:,})  FLOPs: {fl/1e6:.1f}M")
        results[f'flid_pipeline_{pname}'] = {
            'total_params': tp, 'trainable_params': tr,
            'total_flops': fl, 'backbone_passes': passes,
        }

    # ── Baseline ──
    print("\n" + "─" * 75)
    print("  BASELINE: MobileNetV2 from scratch (Gonzalez & Tapia)")
    bl = MobileNetV2PAD(pretrained=False).eval()
    total_bl, trainable_bl = count_parameters(bl)
    flops_bl = count_model_flops_cnn(bl, (1, 3, 448, 448))
    lat_bl, lat_bl_std = measure_latency_cpu(bl, torch.randn(1, 3, 448, 448))
    print(f"    Params: {total_bl:,}  FLOPs: {flops_bl/1e6:.1f}M  Latency: {lat_bl:.2f}ms")

    results['baseline_single'] = {
        'params_total': total_bl, 'params_trainable': trainable_bl,
        'flops': flops_bl, 'latency_ms': round(lat_bl, 2),
    }

    cascade_params = 2 * total_bl
    cascade_flops = 2 * flops_bl
    print(f"    Cascade (2×MobileNetV2): {cascade_params:,} params, {cascade_flops/1e6:.1f}M FLOPs")
    results['baseline_cascade'] = {
        'total_params': cascade_params, 'total_flops': cascade_flops,
    }

    # ── End-to-end latency ──
    print("\n" + "─" * 75)
    print("  END-TO-END LATENCY (CPU, batch=1)")

    bb_e2e = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval()
    face_mlp_e2e = FaceClassifier().eval()

    def flid_forward(x):
        feat = bb_e2e.features(x)
        feat = bb_e2e.avgpool(feat).flatten(1)
        return face_mlp_e2e(feat)

    dummy_224 = torch.randn(1, 3, 224, 224)
    for _ in range(20):
        flid_forward(dummy_224)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            flid_forward(dummy_224)
        times.append((time.perf_counter() - t0) * 1000)
    flid_lat = float(np.mean(times))

    bl_e2e = MobileNetV2PAD(pretrained=False).eval()
    dummy_448 = torch.randn(1, 3, 448, 448)
    for _ in range(20):
        with torch.no_grad():
            bl_e2e(dummy_448)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            bl_e2e(dummy_448)
        times.append((time.perf_counter() - t0) * 1000)
    bl_lat = float(np.mean(times))

    speedup = bl_lat / flid_lat
    print(f"    FLiD (224×224):     {flid_lat:.2f} ms")
    print(f"    Baseline (448×448): {bl_lat:.2f} ms")
    print(f"    Speedup: {speedup:.1f}×")

    results['e2e_latency'] = {
        'flid_ms': round(flid_lat, 2), 'baseline_ms': round(bl_lat, 2),
        'speedup': round(speedup, 1),
    }

    # ── Key ratios ──
    print(f"\n  KEY RATIOS:")
    print(f"    Param ratio:  {trainable_bl / face_trainable:.0f}× fewer trainable params (FLiD)")
    print(f"    FLOP ratio:   {flops_bl / face_total_flops:.1f}× fewer FLOPs (FLiD Face)")
    print(f"    Latency:      {speedup:.1f}× faster (FLiD)")

    # ── Save ──
    def ser(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    out_path = EFFICIENCY_OUTPUT / 'efficiency_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=ser)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
