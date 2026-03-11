#!/usr/bin/env python3
"""Benchmark CPU latency for all FLiD pipelines and baseline."""
import torch
import torch.nn as nn
import time
import numpy as np
from torchvision import models

# MobileNetV3-Small backbone
m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

# MLP heads
face_mlp = nn.Sequential(
    nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 1))
text_mlp = nn.Sequential(
    nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 1))
both_mlp = nn.Sequential(
    nn.Linear(1152, 512), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 1))

face_mlp.eval()
text_mlp.eval()
both_mlp.eval()

# Count params
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

bb_params = sum(p.numel() for p in backbone.parameters())
print(f"Backbone params: {bb_params:,}")
print(f"Face MLP trainable: {count_params(face_mlp):,}")
print(f"Text MLP trainable: {count_params(text_mlp):,}")
print(f"Both MLP trainable: {count_params(both_mlp):,}")
print(f"FLiD Face total: {bb_params + count_params(face_mlp):,}")
print(f"FLiD Text total: {bb_params + count_params(text_mlp):,}")
print(f"FLiD Both total: {bb_params + count_params(both_mlp):,}")

# Benchmark on CPU
device = torch.device("cpu")
backbone = backbone.to(device)
face_mlp = face_mlp.to(device)
text_mlp = text_mlp.to(device)
both_mlp = both_mlp.to(device)

x = torch.randn(1, 3, 224, 224, device=device)
N = 100

# Warmup
for _ in range(20):
    with torch.no_grad():
        backbone(x)

# Face: 1 backbone + MLP
times = []
for _ in range(N):
    t0 = time.perf_counter()
    with torch.no_grad():
        emb = backbone(x)
        face_mlp(emb)
    times.append(time.perf_counter() - t0)
print(f"\nFLiD Face latency (CPU): {np.median(times)*1000:.1f} ms")

# Text: 1 backbone + MLP (single-crop)
times = []
for _ in range(N):
    t0 = time.perf_counter()
    with torch.no_grad():
        emb = backbone(x)
        text_mlp(emb)
    times.append(time.perf_counter() - t0)
print(f"FLiD Text latency (CPU, single-crop): {np.median(times)*1000:.1f} ms")

# Both: 2 backbone + bigger MLP
times = []
for _ in range(N):
    t0 = time.perf_counter()
    with torch.no_grad():
        face_emb = backbone(x)
        text_emb = backbone(x)
        combined = torch.cat([face_emb, text_emb], dim=1)
        both_mlp(combined)
    times.append(time.perf_counter() - t0)
print(f"FLiD Both latency (CPU): {np.median(times)*1000:.1f} ms")

# Baseline MobileNetV2 448x448
mv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mv2.eval()
mv2 = mv2.to(device)
mv2_params = sum(p.numel() for p in mv2.parameters())
print(f"\nBaseline MobileNetV2 params: {mv2_params:,}")

x448 = torch.randn(1, 3, 448, 448, device=device)
for _ in range(20):
    with torch.no_grad():
        mv2(x448)

times = []
for _ in range(N):
    t0 = time.perf_counter()
    with torch.no_grad():
        mv2(x448)
    times.append(time.perf_counter() - t0)
baseline_single = np.median(times) * 1000
print(f"Baseline single model latency (CPU): {baseline_single:.1f} ms")

times = []
for _ in range(N):
    t0 = time.perf_counter()
    with torch.no_grad():
        mv2(x448)
        mv2(x448)
    times.append(time.perf_counter() - t0)
print(f"Baseline two-model latency (CPU): {np.median(times)*1000:.1f} ms")
