"""
models/temporal_model.py
Phase 3 (Tri-Objective Multi-Task Temporal Stress Inference) — Steps 20 & 21: Recurrent Multi-Task Model.

Models temporal relationships across video frame sequences using a Gated Recurrent Unit (GRU)
with Temporal Attention Pooling to predict three synchronized physiological/affective indicators:
    [f_1, f_2, ..., f_W] -> Bidirectional GRU -> Temporal Attention
                                                         │
                                ┌────────────────────────┼────────────────────────┐
                                ▼                        ▼                        ▼
                       [Stress Classifier]         [HR Regressor]           [BVP Regressor]
                        Logits: [B, 2]             BPM: [B, 1]              Waveform: [B, W]
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder


class TemporalAttention(nn.Module):
    """
    Computes learnable temporal attention weights across video frames in a window.
    Allows the model to focus on the exact moment a micro-expression occurs
    (e.g., rapid brow lowering or lip tightening) rather than treating all frames equally.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        """
        Args:
            rnn_outputs: Tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len, 1)
        """
        scores = self.attention(rnn_outputs)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(rnn_outputs * weights, dim=1)
        return context, weights


class TemporalStressGRU(nn.Module):
    """
    Tri-Objective Multi-Task Recurrent temporal network for affective & physiological stress inference.
    Takes sequences of frame feature vectors [B, W, input_dim] and outputs:
    1. stress_logits: [B, num_classes] (Binary Calm vs. Cognitive Stress)
    2. hr_pred: [B, 1] (Continuous Heart Rate in BPM)
    3. bvp_pred: [B, W] (Continuous Blood Volume Pulse vascular waveform)
    4. weights: [B, W, 1] (Temporal attention weights)
    """
    def __init__(
        self,
        input_dim=1280,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        window_size=10,
        dropout=0.2,
        bidirectional=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.window_size = window_size

        # Bidirectional GRU captures forward onset and backward recovery dynamics
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = TemporalAttention(rnn_out_dim)

        # Head 1: Binary Stress Classifier (Calm vs. Acute Mental Stress)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Head 2: Heart Rate (HR) Regressor in Beats Per Minute (BPM)
        self.hr_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Head 3: Blood Volume Pulse (BVP) Waveform Regressor (W-length pulse curve)
        self.bvp_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, window_size)
        )

    def forward(self, x):
        """
        Args:
            x: Feature sequence tensor of shape (batch_size, window_size, input_dim)
        Returns:
            stress_logits: (batch_size, num_classes)
            hr_pred: (batch_size, 1)
            bvp_pred: (batch_size, window_size)
            weights: (batch_size, window_size, 1)
        """
        rnn_out, _ = self.gru(x)  # shape: (batch_size, window_size, rnn_out_dim)
        context, weights = self.attention(rnn_out)  # shape: (batch_size, rnn_out_dim)
        
        stress_logits = self.classifier(context)
        hr_pred = self.hr_head(context)
        bvp_pred = self.bvp_head(context)

        return stress_logits, hr_pred, bvp_pred, weights


class TwoStreamTemporalStressGRU(nn.Module):
    """
    Two-Stream Multi-Scale Temporal Network (Change 2):
    Stream A (Micro / Fast): W_micro frames (default 10 frames = ~1.7s) to detect rapid micro-expressions.
    Stream B (Macro / Slow): W_macro frames (default 30 frames spanning 10s) to detect sustained baseline cognitive stress.
    Fused via Concatenation + Attention Pooling -> Tri-Objective Heads (Stress, HR, BVP).
    """
    def __init__(
        self,
        input_dim=1280,
        hidden_dim=64,
        num_layers=2,
        num_classes=2,
        micro_window_size=10,
        macro_window_size=30,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()
        self.micro_window_size = micro_window_size
        self.macro_window_size = macro_window_size
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Stream A: Micro (Fast)
        self.gru_micro = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn_micro = TemporalAttention(rnn_out_dim)

        # Stream B: Macro (Slow)
        self.gru_macro = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn_macro = TemporalAttention(rnn_out_dim)

        # Fused Representation
        fused_dim = rnn_out_dim * 2

        # Head 1: Binary Stress Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Head 2: Heart Rate (BPM) Regressor
        self.hr_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Head 3: BVP Waveform Regressor
        self.bvp_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, micro_window_size)
        )

    def forward(self, micro_x, macro_x=None):
        """
        Args:
            micro_x: [B, W_micro, input_dim]
            macro_x: [B, W_macro, input_dim] (optional fallback to micro_x)
        Returns:
            stress_logits: [B, num_classes]
            hr_pred: [B, 1]
            bvp_pred: [B, micro_window_size]
            weights: (weights_micro, weights_macro)
        """
        if macro_x is None:
            # Fallback interpolation if only single stream provided
            macro_x = micro_x

        rnn_micro, _ = self.gru_micro(micro_x)
        ctx_micro, w_micro = self.attn_micro(rnn_micro)

        rnn_macro, _ = self.gru_macro(macro_x)
        ctx_macro, w_macro = self.attn_macro(rnn_macro)

        ctx_fused = torch.cat([ctx_micro, ctx_macro], dim=-1)

        stress_logits = self.classifier(ctx_fused)
        hr_pred = self.hr_head(ctx_fused)
        bvp_pred = self.bvp_head(ctx_fused)

        return stress_logits, hr_pred, bvp_pred, (w_micro, w_macro)


class EndToEndTemporalStressModel(nn.Module):
    """
    Full end-to-end multi-task inference pipeline:
    Raw Video Windows [B, W, 3, 224, 224] -> CNN Backbone -> BiGRU -> (Stress, HR, BVP).
    """
    def __init__(self, visual_encoder, temporal_rnn):
        super().__init__()
        self.encoder = visual_encoder
        self.temporal_rnn = temporal_rnn

    def forward(self, video_tensor):
        """
        Args:
            video_tensor: Shape (batch_size, window_size, 3, 224, 224)
        Returns:
            stress_logits: (batch_size, num_classes)
            hr_pred: (batch_size, 1)
            bvp_pred: (batch_size, window_size)
            weights: (batch_size, window_size, 1)
        """
        b, w, c, h, d = video_tensor.shape
        flat_frames = video_tensor.view(b * w, c, h, d)
        features = self.encoder(flat_frames)  # shape: (b * w, 1280)
        seq_features = features.view(b, w, -1)  # shape: (b, w, 1280)
        return self.temporal_rnn(seq_features)


def run_temporal_model_test():
    """Standalone diagnostic verifying Step 3.2 multi-task architecture."""
    print("=" * 65)
    print("  Phase 3 — Tri-Objective Multi-Task Temporal Model Diagnostic")
    print("=" * 65)

    batch_size = 4
    window_size = 10
    feature_dim = 1280

    model = TemporalStressGRU(input_dim=feature_dim, hidden_dim=64, num_classes=2, window_size=window_size)
    dummy_features = torch.randn(batch_size, window_size, feature_dim)

    logits, hr_pred, bvp_pred, weights = model(dummy_features)
    print(f"[Input Sequence]         : {list(dummy_features.shape)} [Batch, Window, Dim]")
    print(f"[1. Stress Logits]       : {list(logits.shape)} [Batch, 2]")
    print(f"[2. Heart Rate (HR)]     : {list(hr_pred.shape)} [Batch, 1]")
    print(f"[3. BVP Pulse Waveform]  : {list(bvp_pred.shape)} [Batch, {window_size}]")
    print(f"[4. Attention Weights]   : {list(weights.shape)} [Batch, Window, 1]")

    weight_sums = weights.sum(dim=1).squeeze(-1)
    print(f"[Attention Sums (==1.0)] : {weight_sums.tolist()}")
    print("=" * 65)
    print("[SUCCESS] Tri-Objective Multi-Task Temporal Model verified!")


if __name__ == "__main__":
    run_temporal_model_test()
