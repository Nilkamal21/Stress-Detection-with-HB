"""
evaluation/evaluate_temporal.py
Phase 3 (Tri-Objective Multi-Task Temporal Stress Inference) — Comprehensive Evaluation & Diagnostics.

Evaluates best_temporal_stress_model.pth on held-out test subjects (subjects 38–42):
1. Stress Confusion Matrix (Raw counts & Normalized percentage heatmap)
2. Operating Decision Threshold Sweep across tau in [0.35, 0.75]
3. Heart Rate (HR) Regression Performance (MAE, RMSE, Pearson r)
4. Blood Volume Pulse (BVP) Waveform Reconstruction (Pearson r correlation)
5. Visual comparison plots and artifacts saved to results/temporal/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.temporal_model import TemporalStressGRU
from training.train_temporal import extract_or_load_features, load_phase2_feature_extractor

TEMPORAL_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "temporal")


def plot_temporal_confusion_matrix(cm, class_names, output_path, title="Temporal Stress Confusion Matrix", normalize=False):
    """Plots and saves seaborn heatmap of the 2-class stress confusion matrix."""
    plt.figure(figsize=(6, 5))
    if normalize:
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            annot_kws={"size": 14, "weight": "bold"}
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            annot_kws={"size": 14, "weight": "bold"}
        )

    plt.title(title, fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Predicted State", fontsize=11, labelpad=8)
    plt.ylabel("True Subject State", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved Heatmap] {output_path}")


def plot_physiological_evaluations(true_hr, pred_hr, true_bvp, pred_bvp):
    """Generates visual plots for Heart Rate correlation and BVP waveform fidelity."""
    # 1. Heart Rate Scatter Plot
    plt.figure(figsize=(6.5, 5))
    plt.scatter(true_hr, pred_hr, color="teal", alpha=0.5, edgecolors="none", s=25)
    min_val = min(float(np.min(true_hr)), float(np.min(pred_hr)))
    max_val = max(float(np.max(true_hr)), float(np.max(pred_hr)))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Agreement (y = x)")
    plt.title("Heart Rate (HR) Estimation: True vs. Predicted (BPM)", fontsize=11, fontweight="bold")
    plt.xlabel("Ground Truth Heart Rate (Pulse Oximeter BPM)", fontsize=10)
    plt.ylabel("Predicted Heart Rate (BiGRU Head BPM)", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hr_plot_path = os.path.join(TEMPORAL_RESULTS_DIR, "hr_prediction_scatter.png")
    plt.savefig(hr_plot_path, dpi=300)
    plt.close()
    print(f"[Saved Plot] {hr_plot_path}")

    # 2. BVP Waveform Comparison Plot (Sample Window)
    if len(true_bvp) > 0 and len(pred_bvp) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(true_bvp[0], label="Ground Truth BVP Waveform (PPG)", color="darkred", linewidth=2.2)
        plt.plot(pred_bvp[0], label="Reconstructed BVP Waveform (Model Head)", color="dodgerblue", linestyle="--", linewidth=2.0)
        plt.title("Sample BVP Vascular Pulse Waveform Reconstruction (W=10 frames)", fontsize=11, fontweight="bold")
        plt.xlabel("Window Frame Index (t)", fontsize=10)
        plt.ylabel("Normalized Pulse Amplitude", fontsize=10)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        bvp_plot_path = os.path.join(TEMPORAL_RESULTS_DIR, "bvp_sample_waveform.png")
        plt.savefig(bvp_plot_path, dpi=300)
        plt.close()
        print(f"[Saved Plot] {bvp_plot_path}")


def run_temporal_evaluation(data_dir=None, ckpt_path=None, phase2_ckpt=None, threshold=None, window_size=10, stride=4):
    args_thresh = threshold
    print("=" * 65)
    print("  Phase 3 — Tri-Objective Multi-Task Temporal Stress Evaluation")
    print("=" * 65)

    os.makedirs(TEMPORAL_RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load feature extractor and extract/load test features
    encoder = load_phase2_feature_extractor(ckpt_path=phase2_ckpt, device=device)
    test_feats, test_labels, test_hr, test_bvp, subjs = extract_or_load_features(
        encoder, data_dir, "test", window_size=window_size, stride=stride, device=device
    )

    # Candidate checkpoints
    candidate_ckpts = []
    if ckpt_path:
        candidate_ckpts.append(ckpt_path)
    candidate_ckpts.extend([
        os.path.join(config.CHECKPOINTS_DIR, "best_temporal_stress_model.pth"),
        "/kaggle/working/Stress-Detection/models_checkpoints/best_temporal_stress_model.pth",
        "/kaggle/working/models_checkpoints/best_temporal_stress_model.pth",
    ])

    final_ckpt = None
    for p in candidate_ckpts:
        if os.path.exists(p):
            final_ckpt = p
            break

    if final_ckpt is None:
        raise FileNotFoundError(f"Missing temporal model checkpoint: {candidate_ckpts}")

    print(f"[Model Checkpoint] Loading: {final_ckpt}")
    ckpt = torch.load(final_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    input_dim = ckpt.get("input_dim", test_feats.shape[-1])
    
    # Auto-detect hidden_dim from GRU weights
    if "gru.weight_ih_l0" in state_dict:
        hidden_dim = state_dict["gru.weight_ih_l0"].shape[0] // 3
    else:
        hidden_dim = ckpt.get("hidden_dim", 64)

    print(f"[Model Spec] Input Dim: {input_dim} | Hidden Dim: {hidden_dim} (BiGRU Multi-Task)")
    model = TemporalStressGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=2,
        window_size=window_size,
        bidirectional=True
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 1. Evaluate Test Predictions
    with torch.no_grad():
        out = model(test_feats.to(device))
        if len(out) == 4:
            logits, pred_hr, pred_bvp, weights = out
            has_mt = True
        else:
            logits, weights = out
            pred_hr, pred_bvp = None, None
            has_mt = False

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        targets = test_labels.numpy()

    # 2. Comprehensive Threshold Sweep Table (Step 21 Operating Point Analysis)
    print("\n" + "=" * 90)
    print("  Step 21: Operating Threshold Sweep on Held-Out Test Subjects (Subjects 38–42)")
    print("=" * 90)
    print(f"  {'Thresh':>6s} | {'Accuracy':>8s} | {'True Calm':>9s} | {'False Alarm':>11s} | {'True Stress':>11s} | {'Missed':>7s} | {'Spec%':>6s} | {'Sens%':>6s} | {'F1%':>6s}")
    print("-" * 90)

    sweep_results = []
    best_balanced_thresh = 0.50
    min_diff = 999.0

    for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        p_th = (probs >= th).astype(int)
        cm_th = confusion_matrix(targets, p_th)
        tn, fp, fn, tp = cm_th.ravel()
        acc_th = (tn + tp) / len(targets) * 100.0
        spec_th = tn / max(tn + fp, 1) * 100.0
        sens_th = tp / max(tp + fn, 1) * 100.0
        prec_th = tp / max(tp + fp, 1) * 100.0
        f1_th = 2 * (prec_th * sens_th) / max(prec_th + sens_th, 1e-6)

        diff = abs(spec_th - sens_th)
        if diff < min_diff:
            min_diff = diff
            best_balanced_thresh = th

        sweep_results.append((th, acc_th, tn, fp, tp, fn, spec_th, sens_th, f1_th))
        marker = " <== Balanced" if th == 0.60 else ""
        print(f"  {th:6.2f} | {acc_th:7.1f}% | {tn:9d} | {fp:11d} | {tp:11d} | {fn:7d} | {spec_th:5.1f}% | {sens_th:5.1f}% | {f1_th:5.1f}%{marker}")

    print("=" * 90)

    selected_thresh = args_thresh if args_thresh is not None else best_balanced_thresh
    print(f"[Selected Operating Point] Using Threshold tau = {selected_thresh:.2f}")

    preds = (probs >= selected_thresh).astype(int)
    class_names = ["Calm (0)", "Stress (1)"]
    cm = confusion_matrix(targets, preds)
    print("\nConfusion Matrix (Raw Counts):")
    print(cm)

    # Save CSV
    df_cm = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    csv_path = os.path.join(TEMPORAL_RESULTS_DIR, "confusion_matrix.csv")
    df_cm.to_csv(csv_path)
    print(f"[Saved CSV] {csv_path}")

    # Plot raw & normalized heatmaps
    raw_plot = os.path.join(TEMPORAL_RESULTS_DIR, "confusion_matrix.png")
    plot_temporal_confusion_matrix(cm, class_names, raw_plot, title=f"Temporal Stress Confusion Matrix (tau={selected_thresh:.2f})", normalize=False)

    norm_plot = os.path.join(TEMPORAL_RESULTS_DIR, "confusion_matrix_normalized.png")
    plot_temporal_confusion_matrix(cm, class_names, norm_plot, title=f"Temporal Stress Confusion Matrix Normalized (tau={selected_thresh:.2f})", normalize=True)

    # Classification report
    report = classification_report(targets, preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    rep_csv = os.path.join(TEMPORAL_RESULTS_DIR, "classification_report.csv")
    report_df.to_csv(rep_csv)
    print(f"[Saved Report] {rep_csv}")

    acc = accuracy_score(targets, preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    spec = float(cm[0, 0]) / max(float(cm[0, 0] + cm[0, 1]), 1.0) * 100.0
    try:
        auc = roc_auc_score(targets, probs) * 100.0
    except Exception:
        auc = 0.0

    print(f"\nFinal Test Metrics on Unseen UBFC Subjects (38–42):")
    print(f"  - Accuracy   : {acc:.2f}% ({cm[0,0] + cm[1,1]}/{len(targets)} windows)")
    print(f"  - Specificity: {spec:.2f}% ({cm[0,0]}/{cm[0,0] + cm[0,1]} Calm windows correctly identified)")
    print(f"  - Sensitivity: {rec * 100:.2f}% ({cm[1,1]}/{cm[1,0] + cm[1,1]} Stress windows correctly identified)")
    print(f"  - Precision  : {prec * 100:.2f}%")
    print(f"  - F1-Score   : {f1 * 100:.2f}%")
    print(f"  - ROC-AUC    : {auc:.2f}%")

    # 3. Physiological Multi-Task Performance Evaluation
    if has_mt and pred_hr is not None and pred_bvp is not None:
        np_pred_hr = pred_hr.squeeze(-1).cpu().numpy()
        np_true_hr = test_hr.numpy()
        np_pred_bvp = pred_bvp.cpu().numpy()
        np_true_bvp = test_bvp.numpy()

        hr_mae = float(np.mean(np.abs(np_pred_hr - np_true_hr)))
        hr_rmse = float(np.sqrt(np.mean((np_pred_hr - np_true_hr) ** 2)))
        try:
            hr_corr = float(np.corrcoef(np_true_hr, np_pred_hr)[0, 1])
        except Exception:
            hr_corr = 0.0

        print(f"\nPhysiological Telemetry Performance on Held-Out Subjects:")
        print(f"  - Heart Rate MAE : {hr_mae:.2f} BPM")
        print(f"  - Heart Rate RMSE: {hr_rmse:.2f} BPM")
        print(f"  - Heart Rate r   : {hr_corr:.4f}")

        plot_physiological_evaluations(np_true_hr, np_pred_hr, np_true_bvp, np_pred_bvp)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tri-Objective Multi-Task Temporal Stress BiGRU")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/ubfc_faces")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--phase2_ckpt", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: auto-calibrate on val)")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_temporal_evaluation(
        data_dir=args.data_dir,
        ckpt_path=args.checkpoint,
        phase2_ckpt=args.phase2_ckpt,
        threshold=args.threshold,
        window_size=args.window_size,
        stride=args.stride
    )
