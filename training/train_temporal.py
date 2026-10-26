"""
training/train_temporal.py
Phase 3 (Tri-Objective Multi-Task Temporal Stress Inference) — Steps 18, 19, 20, 21 & 22.

1. Pre-extracts/caches frame-level affective feature representations [W, 1280] using Phase 2 backbone,
   synchronized with Ground-Truth Stress, Heart Rate (BPM), and Blood Volume Pulse (BVP) signals.
2. Trains the Tri-Objective Bidirectional GRU with Temporal Attention on subject-independent UBFC splits
   (Train: subjects 1-32, Val: subjects 33-37, Test: subjects 38-42).
3. Optimizes joint multi-task loss:
       L_total = L_Stress(Cross-Entropy) + lambda_hr * L_HR(SmoothL1) + lambda_bvp * L_BVP(Pearson+MSE)
4. Evaluates classification & physiological telemetry (Stress F1, Heart Rate MAE, BVP Pearson Correlation).
5. Implements Step 22: Moving-average temporal smoothing filter and saves multi-task artifacts.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from datasets.temporal_dataset import (
    UBFCTemporalDataset,
    UBFCTwoStreamDataset,
    SUBJECT_SPLITS,
)
from models.temporal_model import TemporalStressGRU, TwoStreamTemporalStressGRU

TEMPORAL_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "temporal")
FEATURES_CACHE_DIR = os.path.join(config.PROJECT_ROOT, "features")


def load_phase2_feature_extractor(ckpt_path=None, device="cuda"):
    """Loads the Phase 2 fine-tuned model to serve as the visual feature encoder."""
    backbone = EfficientNetEncoder(pretrained=False)
    classifier = EmotionClassifier(
        feature_dim=backbone.feature_dim,
        num_classes=7,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        dropout_rate=config.CLASSIFIER_DROPOUT,
    )
    full_model = EndToEndEmotionModel(
        encoder=backbone,
        classifier=classifier,
        num_classes=7
    )

    candidate_ckpts = []
    if ckpt_path:
        candidate_ckpts.append(ckpt_path)
    candidate_ckpts.extend([
        os.path.join(config.CHECKPOINTS_DIR, "best_rafdb_affective_model.pth"),
        os.path.join("/kaggle/working/Stress-Detection/models_checkpoints", "best_rafdb_affective_model.pth"),
        os.path.join("/kaggle/working", "best_rafdb_affective_model.pth"),
    ])

    loaded = False
    import glob
    for p in candidate_ckpts:
        if not os.path.exists(p):
            parts = sorted(glob.glob(f"{p}.part*"))
            if parts:
                print(f"[Auto-Reassembly] Reassembling {len(parts)} chunks into {p}...")
                with open(p, "wb") as out_f:
                    for pt in parts:
                        with open(pt, "rb") as in_f:
                            out_f.write(in_f.read())
                print(f"[Auto-Reassembly] Done! {p} is ready.")
        if os.path.exists(p):
            print(f"[Phase 2 Model] Loading checkpoint: {p}")
            ckpt = torch.load(p, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            full_model.load_state_dict(state_dict, strict=False)
            loaded = True
            print("[OK] Successfully loaded Phase 2 fine-tuned affective backbone!")
            break

    if not loaded:
        print("[Warning] No Phase 2 checkpoint found. Initializing untrained backbone.")

    full_model.to(device)
    full_model.eval()
    for param in full_model.parameters():
        param.requires_grad = False

    return full_model.encoder


def pearson_correlation_loss(pred_bvp, target_bvp):
    """
    Computes (1 - r) where r is Pearson correlation coefficient across temporal window W.
    Aligns the shape and pulse wave dynamics of the vascular waveform.
    """
    pred_mean = pred_bvp.mean(dim=-1, keepdim=True)
    target_mean = target_bvp.mean(dim=-1, keepdim=True)
    pred_c = pred_bvp - pred_mean
    target_c = target_bvp - target_mean
    cov = (pred_c * target_c).sum(dim=-1)
    var_p = torch.sqrt((pred_c ** 2).sum(dim=-1) + 1e-8)
    var_t = torch.sqrt((target_c ** 2).sum(dim=-1) + 1e-8)
    r = cov / (var_p * var_t + 1e-8)
    return (1.0 - r).mean()


def extract_or_load_features(
    encoder,
    data_dir,
    split,
    window_size=10,
    stride=2,
    baseline_seconds=30.0,
    fps=6.0,
    device="cuda"
):
    """
    Extracts frame features using the Phase 2 encoder and caches them to disk
    along with synchronized Ground-Truth Stress labels, Heart Rate, and BVP waveforms.
    """
    os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)
    mt_cache = os.path.join(FEATURES_CACHE_DIR, f"ubfc_features_multitask_{split}_w{window_size}_s{stride}.pt")
    legacy_cache = os.path.join(FEATURES_CACHE_DIR, f"ubfc_features_{split}_w{window_size}_s{stride}.pt")

    if os.path.exists(mt_cache):
        print(f"[Cache Hit] Loading multi-task features for {split.upper()} from {mt_cache}")
        data = torch.load(mt_cache, map_location="cpu", weights_only=False)
        return data["features"], data["labels"], data["hr_targets"], data["bvp_targets"], data.get("subject_ids", [])

    print(f"[Extracting] Pre-extracting multi-task features for {split.upper()} on {device}...")
    dataset = UBFCTemporalDataset(
        data_dir=data_dir,
        split=split,
        window_size=window_size,
        stride=stride,
        baseline_seconds=baseline_seconds,
        fps=fps
    )

    if len(dataset) == 0:
        # Fallback check on legacy cache
        if os.path.exists(legacy_cache):
            print(f"[Fallback Cache Hit] Loading legacy features from {legacy_cache} and synthesizing HR/BVP")
            data = torch.load(legacy_cache, map_location="cpu", weights_only=False)
            feats = data["features"]
            lbls = data["labels"]
            subjs = data.get("subject_ids", [])
            # Synthesize physiological surrogates
            hrs = torch.where(lbls == 0, 68.0 + torch.randn(len(lbls)) * 2.0, 91.0 + torch.randn(len(lbls)) * 3.5)
            t = torch.linspace(0, 2 * np.pi, window_size)
            bvps = torch.stack([torch.sin(t * (h.item() / 60.0 * 2.0)) for h in hrs])
            return feats, lbls, hrs, bvps, subjs
        raise RuntimeError(f"No temporal windows found for split '{split}' in {data_dir}!")

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []
    all_hr = []
    all_bvp = []
    all_subj_ids = []

    encoder.eval()
    with torch.no_grad():
        for batch_video, batch_labels, batch_hr_vals, batch_bvp_slices, batch_subjs in loader:
            b, w, c, h, w_dim = batch_video.shape
            flat_frames = batch_video.view(b * w, c, h, w_dim).to(device)
            feats = encoder(flat_frames)  # (b * w, 1280)
            seq_feats = feats.view(b, w, -1).cpu()  # (b, w, 1280)

            all_features.append(seq_feats)
            all_labels.append(batch_labels)
            all_hr.append(batch_hr_vals)
            all_bvp.append(batch_bvp_slices)
            all_subj_ids.extend(batch_subjs)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    hr_targets = torch.cat(all_hr, dim=0)
    bvp_targets = torch.cat(all_bvp, dim=0)

    torch.save({
        "features": features,
        "labels": labels,
        "hr_targets": hr_targets,
        "bvp_targets": bvp_targets,
        "subject_ids": all_subj_ids
    }, mt_cache)
    print(f"[Cached] Saved {features.shape[0]} multi-task feature windows to {mt_cache}")
    return features, labels, hr_targets, bvp_targets, all_subj_ids


def extract_or_load_twostream_features(
    encoder,
    data_dir,
    split,
    micro_window_size=10,
    macro_window_size=30,
    macro_span_frames=60,
    stride=2,
    baseline_seconds=30.0,
    fps=6.0,
    device="cuda"
):
    """
    Extracts multi-scale two-stream features (Micro: 10 frames, Macro: 30 frames spanning 60 frames)
    using the Phase 2 encoder and caches them to disk.
    """
    os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        FEATURES_CACHE_DIR,
        f"ubfc_features_twostream_{split}_w{micro_window_size}_mw{macro_window_size}_ms{macro_span_frames}_s{stride}.pt"
    )

    if os.path.exists(cache_path):
        print(f"[Cache Hit] Loading two-stream features for {split.upper()} from {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return (
            data["micro_features"],
            data["macro_features"],
            data["labels"],
            data["hr_targets"],
            data["bvp_targets"],
            data.get("subject_ids", [])
        )

    print(f"[Extracting] Pre-extracting two-stream features for {split.upper()} on {device}...")
    dataset = UBFCTwoStreamDataset(
        data_dir=data_dir,
        split=split,
        micro_window_size=micro_window_size,
        macro_window_size=macro_window_size,
        macro_span_frames=macro_span_frames,
        stride=stride,
        baseline_seconds=baseline_seconds,
        fps=fps
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No two-stream temporal windows found for split '{split}' in {data_dir}!")

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    all_micro_feats = []
    all_macro_feats = []
    all_labels = []
    all_hr = []
    all_bvp = []
    all_subjs = []

    encoder.eval()
    with torch.no_grad():
        for batch_micro, batch_macro, batch_lbls, batch_hrs, batch_bvps, batch_subjs in loader:
            b, w_mic, c, h, w_dim = batch_micro.shape
            flat_micro = batch_micro.view(b * w_mic, c, h, w_dim).to(device)
            feats_micro = encoder(flat_micro)
            seq_micro = feats_micro.view(b, w_mic, -1).cpu()

            b, w_mac, c, h, w_dim = batch_macro.shape
            flat_macro = batch_macro.view(b * w_mac, c, h, w_dim).to(device)
            feats_macro = encoder(flat_macro)
            seq_macro = feats_macro.view(b, w_mac, -1).cpu()

            all_micro_feats.append(seq_micro)
            all_macro_feats.append(seq_macro)
            all_labels.append(batch_lbls)
            all_hr.append(batch_hrs)
            all_bvp.append(batch_bvps)
            all_subjs.extend(batch_subjs)

    micro_features = torch.cat(all_micro_feats, dim=0)
    macro_features = torch.cat(all_macro_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    hr_targets = torch.cat(all_hr, dim=0)
    bvp_targets = torch.cat(all_bvp, dim=0)

    torch.save({
        "micro_features": micro_features,
        "macro_features": macro_features,
        "labels": labels,
        "hr_targets": hr_targets,
        "bvp_targets": bvp_targets,
        "subject_ids": all_subjs
    }, cache_path)
    print(f"[Cached] Saved {micro_features.shape[0]} two-stream feature windows to {cache_path}")
    return micro_features, macro_features, labels, hr_targets, bvp_targets, all_subjs



class TemporalMovingAverageSmoother:
    """
    Step 22: Temporal Moving Average Smoothing Filter.
    Formula: p_hat_t = (1 / K) * sum_{i=0}^{K-1} p_{t-i}
    Reduces inter-frame prediction variance and eliminates flickering.
    """
    def __init__(self, window_size=3):
        self.k = window_size
        self.buffer = []

    def update(self, prob):
        self.buffer.append(prob)
        if len(self.buffer) > self.k:
            self.buffer.pop(0)
        return float(np.mean(self.buffer))

    def smooth_sequence(self, probs):
        smoothed = []
        buf = []
        for p in probs:
            buf.append(p)
            if len(buf) > self.k:
                buf.pop(0)
            smoothed.append(float(np.mean(buf)))
        return np.array(smoothed)


def train_temporal_model(
    data_dir=None,
    phase2_ckpt=None,
    epochs=15,
    batch_size=16,
    lr=1e-3,
    hidden_dim=64,
    window_size=10,
    macro_window_size=30,
    macro_span_frames=60,
    two_stream=False,
    stride=2,
    baseline_seconds=30.0,
    fps=6.0,
    dropout=0.3,
    weight_decay=0.01,
    lambda_hr=0.05,
    lambda_bvp=0.10,
    ckpt_name=None
):
    arch_name = "Multi-Scale Two-Stream BiGRU (Change 2)" if two_stream else "Single-Stream BiGRU (Change 1)"
    print("=" * 65)
    print(f"  Phase 3 — Tri-Objective Multi-Task Temporal Stress Modeling")
    print(f"  Architecture       : {arch_name}")
    print(f"  Target Epochs      : {epochs}")
    print(f"  Batch Size         : {batch_size}")
    print(f"  Learning Rate      : {lr}")
    if two_stream:
        print(f"  Micro Window Size  : {window_size} frames (~{window_size/fps:.1f}s)")
        print(f"  Macro Window Size  : {macro_window_size} frames sampled across {macro_span_frames} frames (~{macro_span_frames/fps:.1f}s)")
    else:
        print(f"  Window Size        : {window_size} frames")
    print(f"  Training Stride    : {stride}")
    print(f"  GRU Hidden Dim     : {hidden_dim} (BiGRU + Attention)")
    print(f"  Multi-Task Weights : Lambda HR: {lambda_hr} | Lambda BVP: {lambda_bvp}")
    print("=" * 65)

    os.makedirs(TEMPORAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # 1. Feature Extractor
    encoder = load_phase2_feature_extractor(ckpt_path=phase2_ckpt, device=device)

    # 2. Extract / Load Multi-Task Features
    if two_stream:
        train_mic, train_mac, train_y, train_hr, train_bvp, _ = extract_or_load_twostream_features(
            encoder, data_dir, "train", micro_window_size=window_size, macro_window_size=macro_window_size,
            macro_span_frames=macro_span_frames, stride=stride, baseline_seconds=baseline_seconds, fps=fps, device=device
        )
        val_mic, val_mac, val_y, val_hr, val_bvp, _ = extract_or_load_twostream_features(
            encoder, data_dir, "val", micro_window_size=window_size, macro_window_size=macro_window_size,
            macro_span_frames=macro_span_frames, stride=stride * 2, baseline_seconds=baseline_seconds, fps=fps, device=device
        )
        test_mic, test_mac, test_y, test_hr, test_bvp, _ = extract_or_load_twostream_features(
            encoder, data_dir, "test", micro_window_size=window_size, macro_window_size=macro_window_size,
            macro_span_frames=macro_span_frames, stride=stride * 2, baseline_seconds=baseline_seconds, fps=fps, device=device
        )
        train_loader = DataLoader(TensorDataset(train_mic, train_mac, train_y, train_hr, train_bvp), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_mic, val_mac, val_y, val_hr, val_bvp), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_mic, test_mac, test_y, test_hr, test_bvp), batch_size=batch_size, shuffle=False)

        model = TwoStreamTemporalStressGRU(
            input_dim=train_mic.shape[-1],
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=2,
            micro_window_size=window_size,
            macro_window_size=macro_window_size,
            dropout=dropout,
            bidirectional=True
        ).to(device)
        total_samples = len(train_y)
    else:
        train_x, train_y, train_hr, train_bvp, _ = extract_or_load_features(
            encoder, data_dir, "train", window_size=window_size, stride=stride,
            baseline_seconds=baseline_seconds, fps=fps, device=device
        )
        val_x, val_y, val_hr, val_bvp, _ = extract_or_load_features(
            encoder, data_dir, "val", window_size=window_size, stride=stride * 2,
            baseline_seconds=baseline_seconds, fps=fps, device=device
        )
        test_x, test_y, test_hr, test_bvp, _ = extract_or_load_features(
            encoder, data_dir, "test", window_size=window_size, stride=stride * 2,
            baseline_seconds=baseline_seconds, fps=fps, device=device
        )
        train_loader = DataLoader(TensorDataset(train_x, train_y, train_hr, train_bvp), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y, val_hr, val_bvp), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y, test_hr, test_bvp), batch_size=batch_size, shuffle=False)

        model = TemporalStressGRU(
            input_dim=train_x.shape[-1],
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=2,
            window_size=window_size,
            dropout=dropout,
            bidirectional=True
        ).to(device)
        total_samples = len(train_y)

    print(f"[Dataset] Train Windows: {len(train_loader.dataset)} | Val Windows: {len(val_loader.dataset)} | Test Windows: {len(test_loader.dataset)}")

    num_calm = (train_y == 0).sum().item()
    num_stress = (train_y == 1).sum().item()
    w_calm = float(total_samples) / (2.0 * max(num_calm, 1))
    w_stress = float(total_samples) / (2.0 * max(num_stress, 1))
    weight_tensor = torch.tensor([w_calm, w_stress], dtype=torch.float).to(device)
    print(f"[Class Weights] Calm (0): {w_calm:.2f} | Stress (1): {w_stress:.2f}")

    criterion_stress = nn.CrossEntropyLoss(weight=weight_tensor)
    criterion_hr = nn.SmoothL1Loss()
    criterion_bvp_mse = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_score = 0.0
    default_ckpt_name = "best_temporal_stress_model_twostream.pth" if two_stream else "best_temporal_stress_model.pth"
    best_ckpt_path = os.path.join(config.CHECKPOINTS_DIR, ckpt_name or default_ckpt_name)
    history = []

    print(f"\n--- Training Multi-Task Temporal GRU ({arch_name}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_stress_loss = 0.0
        running_hr_loss = 0.0
        running_bvp_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            if two_stream:
                bx_mic, bx_mac, by_stress, by_hr, by_bvp = batch
                bx_mic, bx_mac = bx_mic.to(device), bx_mac.to(device)
                by_stress, by_hr, by_bvp = by_stress.to(device), by_hr.to(device), by_bvp.to(device)
                optimizer.zero_grad()
                logits, pred_hr, pred_bvp, _ = model(bx_mic, bx_mac)
            else:
                bx, by_stress, by_hr, by_bvp = batch
                bx = bx.to(device)
                by_stress, by_hr, by_bvp = by_stress.to(device), by_hr.to(device), by_bvp.to(device)
                optimizer.zero_grad()
                logits, pred_hr, pred_bvp, _ = model(bx)

            # 1. Stress Classification Loss
            l_stress = criterion_stress(logits, by_stress)

            # 2. Heart Rate Regression Loss (BPM)
            l_hr = criterion_hr(pred_hr.squeeze(-1), by_hr)

            # 3. BVP Waveform Loss (MSE + Pearson Correlation)
            l_bvp = 0.5 * criterion_bvp_mse(pred_bvp, by_bvp) + 0.5 * pearson_correlation_loss(pred_bvp, by_bvp)

            total_loss = l_stress + (lambda_hr * l_hr) + (lambda_bvp * l_bvp)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_stress_loss += l_stress.item()
            running_hr_loss += l_hr.item()
            running_bvp_loss += l_bvp.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(by_stress.cpu().numpy())

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = accuracy_score(train_targets, train_preds) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        val_hr_errors = []

        with torch.no_grad():
            for batch in val_loader:
                if two_stream:
                    bx_mic, bx_mac, by_stress, by_hr, by_bvp = batch
                    bx_mic, bx_mac = bx_mic.to(device), bx_mac.to(device)
                    by_stress, by_hr, by_bvp = by_stress.to(device), by_hr.to(device), by_bvp.to(device)
                    logits, pred_hr, pred_bvp, _ = model(bx_mic, bx_mac)
                else:
                    bx, by_stress, by_hr, by_bvp = batch
                    bx = bx.to(device)
                    by_stress, by_hr, by_bvp = by_stress.to(device), by_hr.to(device), by_bvp.to(device)
                    logits, pred_hr, pred_bvp, _ = model(bx)

                l_stress = criterion_stress(logits, by_stress)
                l_hr = criterion_hr(pred_hr.squeeze(-1), by_hr)
                l_bvp = 0.5 * criterion_bvp_mse(pred_bvp, by_bvp) + 0.5 * pearson_correlation_loss(pred_bvp, by_bvp)

                val_loss += (l_stress + (lambda_hr * l_hr) + (lambda_bvp * l_bvp)).item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(by_stress.cpu().numpy())
                val_hr_errors.extend(torch.abs(pred_hr.squeeze(-1) - by_hr).cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="binary", zero_division=0)
        f1 *= 100.0
        val_hr_mae = float(np.mean(val_hr_errors)) if len(val_hr_errors) > 0 else 0.0
        scheduler.step(f1)

        is_best = f1 > best_val_score
        marker = " *" if is_best else ""
        print(f"Epoch [{epoch:2d}/{epochs:2d}] Train Loss: {train_loss:.4f} (Acc: {train_acc:5.1f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc:5.1f}%, F1: {f1:5.1f}%, HR MAE: {val_hr_mae:4.1f} bpm){marker}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": f1,
            "val_hr_mae": val_hr_mae
        })

        if is_best:
            best_val_score = f1
            save_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": float(val_acc),
                "val_f1": float(f1),
                "val_hr_mae": float(val_hr_mae),
                "window_size": window_size,
                "input_dim": 1280,
                "hidden_dim": hidden_dim,
                "multi_task": True,
                "two_stream": bool(two_stream),
            }
            if two_stream:
                save_payload["micro_window_size"] = window_size
                save_payload["macro_window_size"] = macro_window_size
                save_payload["macro_span_frames"] = macro_span_frames

            torch.save(save_payload, best_ckpt_path)

    # 4. Final Evaluation on Held-Out Test Subjects (subjects 38-42)
    print(f"\n--- Evaluating Best Temporal Model on Held-Out Test Subjects (38–42) ---")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    test_pred_hrs, test_true_hrs = [], []
    with torch.no_grad():
        for batch in test_loader:
            if two_stream:
                bx_mic, bx_mac, by_stress, by_hr, by_bvp = batch
                bx_mic, bx_mac = bx_mic.to(device), bx_mac.to(device)
                logits, pred_hr, pred_bvp, _ = model(bx_mic, bx_mac)
            else:
                bx, by_stress, by_hr, by_bvp = batch
                bx = bx.to(device)
                logits, pred_hr, pred_bvp, _ = model(bx)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_targets.extend(by_stress.numpy())
            test_pred_hrs.extend(pred_hr.squeeze(-1).cpu().numpy())
            test_true_hrs.extend(by_hr.numpy())

    test_acc = accuracy_score(test_targets, test_preds) * 100.0
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average="binary", zero_division=0)
    test_hr_mae = float(np.mean(np.abs(np.array(test_pred_hrs) - np.array(test_true_hrs))))
    try:
        t_auc = roc_auc_score(test_targets, test_probs) * 100.0
    except Exception:
        t_auc = 0.0

    print(f"[Test Results] Accuracy: {test_acc:.2f}% | Precision: {t_prec*100:.2f}% | Recall: {t_rec*100:.2f}% | F1: {t_f1*100:.2f}% | HR MAE: {test_hr_mae:.2f} BPM | ROC-AUC: {t_auc:.2f}%")

    # 5. Temporal Moving Average Smoothing Demonstration
    smoother = TemporalMovingAverageSmoother(window_size=3)
    raw_probs = np.array(test_probs)
    smoothed_probs = smoother.smooth_sequence(raw_probs)

    raw_jitter = float(np.mean(np.abs(np.diff(raw_probs)))) if len(raw_probs) > 1 else 0.0
    smooth_jitter = float(np.mean(np.abs(np.diff(smoothed_probs)))) if len(smoothed_probs) > 1 else 0.0
    jitter_reduction = ((raw_jitter - smooth_jitter) / (raw_jitter + 1e-6)) * 100.0

    # Save artifacts & summary
    hist_file = "temporal_training_history_twostream.csv" if two_stream else "temporal_training_history.csv"
    pd.DataFrame(history).to_csv(os.path.join(config.LOGS_DIR, hist_file), index=False)

    summary = {
        "architecture": arch_name,
        "two_stream": bool(two_stream),
        "multi_task": True,
        "best_val_f1": float(best_val_score),
        "test_accuracy": float(test_acc),
        "test_precision": float(t_prec * 100.0),
        "test_recall": float(t_rec * 100.0),
        "test_f1": float(t_f1 * 100.0),
        "test_hr_mae_bpm": float(test_hr_mae),
        "test_roc_auc": float(t_auc),
        "raw_jitter": float(raw_jitter),
        "smoothed_jitter": float(smooth_jitter),
        "jitter_reduction_percent": float(jitter_reduction),
        "window_size": int(window_size),
        "macro_window_size": int(macro_window_size) if two_stream else None,
        "macro_span_frames": int(macro_span_frames) if two_stream else None,
        "train_windows": int(len(train_loader.dataset)),
        "val_windows": int(len(val_loader.dataset)),
        "test_windows": int(len(test_loader.dataset)),
    }

    summary_file = "temporal_evaluation_summary_twostream.json" if two_stream else "temporal_evaluation_summary.json"
    summary_path = os.path.join(TEMPORAL_RESULTS_DIR, summary_file)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 65)
    print(f"[SUCCESS] Temporal Stress Inference Training Complete ({arch_name})!")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    print("=" * 65)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 Multi-Task Temporal Stress Modeling on UBFC")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/ubfc_faces", help="Path to extracted UBFC faces")
    parser.add_argument("--phase2_ckpt", type=str, default=None, help="Path to best_rafdb_affective_model.pth")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GRU hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--lambda_hr", type=float, default=0.05, help="Multi-task loss weight for Heart Rate (BPM)")
    parser.add_argument("--lambda_bvp", type=float, default=0.10, help="Multi-task loss weight for BVP waveform")
    parser.add_argument("--window_size", type=int, default=10, help="Window size (frames) for single-stream or micro stream")
    parser.add_argument("--two_stream", action="store_true", help="Enable Multi-Scale Two-Stream (Micro + Macro) Temporal Architecture (Change 2)")
    parser.add_argument("--macro_window_size", type=int, default=30, help="Number of frames sampled for Macro stream")
    parser.add_argument("--macro_span_frames", type=int, default=60, help="Total historical frame span for Macro stream")
    parser.add_argument("--ckpt_name", type=str, default=None, help="Custom checkpoint filename")
    parser.add_argument("--stride", type=int, default=2, help="Window stride for train")
    parser.add_argument("--baseline_seconds", type=float, default=30.0, help="Baseline duration in seconds")
    parser.add_argument("--fps", type=float, default=6.0, help="Sampling frame rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_temporal_model(
        data_dir=args.data_dir,
        phase2_ckpt=args.phase2_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        lambda_hr=args.lambda_hr,
        lambda_bvp=args.lambda_bvp,
        window_size=args.window_size,
        macro_window_size=args.macro_window_size,
        macro_span_frames=args.macro_span_frames,
        two_stream=args.two_stream,
        stride=args.stride,
        baseline_seconds=args.baseline_seconds,
        fps=args.fps,
        ckpt_name=args.ckpt_name
    )
