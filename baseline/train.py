#!/usr/bin/env python3
"""
Baseline — Train MobileNetV2 PAD (Gonzalez & Tapia re-implementation)

Trains MobileNetV2 from scratch on the Fantasy-ID dataset for
Face_attack, Text_attack, and Both_attack scenarios.  Includes a
two-stage cascade evaluation for Both_attack (min of face/text P(Real)).

Usage:
    python -m baseline.train --attack_type all
    python -m baseline.train --attack_type Face_attack
    python -m baseline.train --cascade_only
"""

import os
import sys
import json
import argparse
import time
import copy
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import TRAIN_TEST_DATA, BASELINE_RESULTS, get_device
from baseline.model import MobileNetV2PAD, get_train_transforms, get_val_transforms, IMG_SIZE
from flid.metrics import compute_pad_metrics

# ════════════════════════════════════════════════════════════
# Configuration (matching paper specifications)
# ════════════════════════════════════════════════════════════
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
PATIENCE = 20
NUM_WORKERS = 0
VAL_SPLIT = 0.2
SEED = 42


# ════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════
class IDCardPADDataset(Dataset):
    """Load images from Real/ and Fake/ folders.
    Label: 0 = Real (bona fide), 1 = Fake (attack).
    """

    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []

        for label, class_name in enumerate(['Real', 'Fake']):
            class_dir = os.path.join(root_dir, class_name, split)
            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} not found, skipping")
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

        print(f"  Loaded {len(self.samples)} images from {root_dir}/{split}")
        lc = Counter(self.labels)
        print(f"    Real: {lc.get(0, 0)}, Fake: {lc.get(1, 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, os.path.basename(self.samples[idx])


# ════════════════════════════════════════════════════════════
# Training helpers
# ════════════════════════════════════════════════════════════
def compute_class_weights(dataset):
    lc = Counter(dataset.labels)
    n = len(dataset)
    nc = len(lc)
    weights = {cls: n / (nc * cnt) for cls, cnt in lc.items()}
    print(f"  Class weights: Real={weights.get(0, 0):.4f}, Fake={weights.get(1, 0):.4f}")
    return weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_labels, all_scores, all_preds = [], [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        _, pred = outputs.max(1)
        loss_sum += loss.item() * images.size(0)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        all_labels.extend(labels.cpu().tolist())
        all_scores.extend(probs[:, 0].cpu().tolist())
        all_preds.extend(pred.cpu().tolist())
    return loss_sum / total, 100.0 * correct / total, all_labels, all_scores, all_preds


# ════════════════════════════════════════════════════════════
# Full training pipeline for one attack type
# ════════════════════════════════════════════════════════════
def train_model(attack_type, data_root, output_dir, device,
                epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    print(f"\n{'='*70}")
    print(f"Training MobileNetV2 PAD for: {attack_type}")
    print(f"{'='*70}")

    attack_dir = os.path.join(data_root, attack_type)
    model_dir = os.path.join(output_dir, attack_type)
    os.makedirs(model_dir, exist_ok=True)

    # Datasets
    train_ds = IDCardPADDataset(attack_dir, 'train', get_train_transforms())
    test_ds  = IDCardPADDataset(attack_dir, 'test',  get_val_transforms())
    if len(train_ds) == 0:
        print(f"ERROR: No training data for {attack_type}")
        return None

    n_val = int(len(train_ds) * VAL_SPLIT)
    n_train = len(train_ds) - n_val
    torch.manual_seed(SEED)
    train_sub, val_sub = torch.utils.data.random_split(train_ds, [n_train, n_val])

    val_ds = IDCardPADDataset(attack_dir, 'train', get_val_transforms())
    val_sub_proper = torch.utils.data.Subset(val_ds, val_sub.indices)

    # Class weights + sampler
    cw = compute_class_weights(train_ds)
    wt = torch.tensor([cw.get(0, 1), cw.get(1, 1)], dtype=torch.float32).to(device)
    train_labels = [train_ds.labels[i] for i in train_sub.indices]
    sample_w = [cw[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_sub_proper, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = MobileNetV2PAD(num_classes=2, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=wt)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_ctr = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if epoch % 5 == 0 or epoch <= 3 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | TrLoss {tl:.4f} TrAcc {ta:.1f}% | "
                  f"VaLoss {vl:.4f} VaAcc {va:.1f}% | LR {optimizer.param_groups[0]['lr']:.2e}")

        if vl < best_val_loss:
            best_val_loss, best_epoch, patience_ctr = vl, epoch, 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': vl, 'val_acc': va,
                'attack_type': attack_type,
            }, os.path.join(model_dir, 'best_model.pth'))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    elapsed = time.time() - t0

    # Evaluate
    ckpt = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    tl, ta, labels, scores, preds = evaluate(model, test_loader, criterion, device)
    pad = compute_pad_metrics(labels, scores)

    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    results = {
        'attack_type': attack_type,
        'architecture': 'MobileNetV2 (from scratch)',
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        'best_epoch': best_epoch,
        'training_time_seconds': elapsed,
        'test_accuracy': ta,
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
        'precision': prec * 100, 'recall': rec * 100, 'f1_score': f1 * 100,
        'iso_metrics': pad,
    }

    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Test Acc: {ta:.2f}%  EER: {pad.get('eer', 0):.2f}%  AUC: {pad.get('auc', 0):.4f}")
    return results


# ════════════════════════════════════════════════════════════
# Two-stage cascade (Both = min(Face P(Real), Text P(Real)))
# ════════════════════════════════════════════════════════════
def evaluate_two_stage_cascade(data_root, output_dir, device):
    print(f"\n{'='*70}")
    print("Two-Stage Cascade Evaluation (Face + Text → Both)")
    print(f"{'='*70}")

    face_path = os.path.join(output_dir, 'Face_attack', 'best_model.pth')
    text_path = os.path.join(output_dir, 'Text_attack', 'best_model.pth')
    for p in [face_path, text_path]:
        if not os.path.exists(p):
            print(f"ERROR: model not found at {p}")
            return None

    face_model = MobileNetV2PAD(2, False).to(device)
    text_model = MobileNetV2PAD(2, False).to(device)
    face_model.load_state_dict(torch.load(face_path, map_location=device)['model_state_dict'])
    text_model.load_state_dict(torch.load(text_path, map_location=device)['model_state_dict'])
    face_model.eval()
    text_model.eval()

    both_dir = os.path.join(data_root, 'Both_attack')
    test_ds = IDCardPADDataset(both_dir, 'test', get_val_transforms())
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_labels, all_scores = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            fp = torch.softmax(face_model(images), 1)[:, 0]
            tp = torch.softmax(text_model(images), 1)[:, 0]
            cascade = torch.min(fp, tp)
            all_labels.extend(labels.tolist())
            all_scores.extend(cascade.cpu().tolist())

    pad = compute_pad_metrics(all_labels, all_scores)
    preds = [0 if s > 0.5 else 1 for s in all_scores]
    acc = 100 * sum(l == p for l, p in zip(all_labels, preds)) / len(all_labels)

    print(f"  Acc: {acc:.2f}%  EER: {pad.get('eer', 0):.2f}%  AUC: {pad.get('auc', 0):.4f}")

    results = {'method': 'Two-Stage Cascade', 'test_accuracy': acc, 'iso_metrics': pad}
    with open(os.path.join(output_dir, 'two_stage_cascade_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Baseline MobileNetV2 PAD training')
    parser.add_argument('--attack_type', default='all',
                        choices=['Face_attack', 'Text_attack', 'Both_attack', 'all'])
    parser.add_argument('--data_root', default=str(TRAIN_TEST_DATA))
    parser.add_argument('--output_dir', default=str(BASELINE_RESULTS))
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cascade_only', action='store_true')
    args = parser.parse_args()

    device = get_device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.cascade_only:
        evaluate_two_stage_cascade(args.data_root, args.output_dir, device)
        return

    attacks = (['Face_attack', 'Text_attack', 'Both_attack']
               if args.attack_type == 'all' else [args.attack_type])
    results = {}
    for atk in attacks:
        r = train_model(atk, args.data_root, args.output_dir, device,
                        args.epochs, args.batch_size, args.lr)
        if r:
            results[atk] = r

    if 'Face_attack' in results and 'Text_attack' in results:
        evaluate_two_stage_cascade(args.data_root, args.output_dir, device)

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for name, r in results.items():
        iso = r.get('iso_metrics', {})
        print(f"  {name:<20s}  Acc={r['test_accuracy']:.2f}%  "
              f"EER={iso.get('eer', 0):.2f}%  AUC={iso.get('auc', 0):.4f}")


if __name__ == '__main__':
    main()
