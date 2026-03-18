"""
uncertainty.py — Model-agnostic MC Dropout inference.

Implements Monte-Carlo Dropout (Gal & Ghahramani, 2016) for epistemic
uncertainty estimation.  Running N stochastic forward passes with dropout
*enabled* yields a distribution over class probabilities from which we
derive:

    mean_probs  — average predicted probability per class  (n_samples, C)
    variance    — per-sample predictive variance           (n_samples, C)
    entropy     — predictive entropy (total uncertainty)   (n_samples,)
    labels      — ground-truth class indices               (n_samples,)

For the XGBoost baseline (no dropout), a single deterministic forward pass
is performed; variance is set to zero and entropy is computed from the
returned probabilities.

Public API
----------
    mc_predict(model, dataloader, n_samples=50, device=None) -> dict
"""

from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from models import XGBoostClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enable_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to *train* mode while leaving everything else in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _entropy(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute predictive entropy H = -sum_c p_c * log(p_c).

    Parameters
    ----------
    probs : np.ndarray  shape ``(n_samples, n_classes)``

    Returns
    -------
    np.ndarray  shape ``(n_samples,)``
    """
    return -np.sum(probs * np.log(probs + eps), axis=1)


# ---------------------------------------------------------------------------
# XGBoost (deterministic)
# ---------------------------------------------------------------------------

def _xgb_predict(
    model      : XGBoostClassifier,
    dataloader : DataLoader,
) -> Dict[str, np.ndarray]:
    """
    Run a single deterministic forward pass for the XGBoost model.

    The DataLoader is iterated to collect (sequence, mask, label) batches;
    features are extracted from the raw sequence tensors by taking the flux
    channel (index 0) statistics on-the-fly.  This mirrors the behaviour of
    ``features.build_feature_matrix`` but works directly from the already-
    batched DataLoader.

    Note: for the XGBoost evaluation path the DataLoader should ideally be
    replaced by the raw feature matrix — see ``evaluate.py`` for how this is
    handled.  This function acts as a fallback.
    """
    all_probs:  list = []
    all_labels: list = []

    for seqs, masks, labels in dataloader:
        # seqs: (B, L, INPUT_DIM)  — extract a quick proxy feature matrix
        # using mean flux and std flux per sample (a minimal subset)
        seqs_np = seqs.numpy()                          # (B, L, D)
        valid   = ~masks.numpy()                        # (B, L)  True = real obs

        rows = []
        for b in range(seqs_np.shape[0]):
            v   = valid[b]
            flux = seqs_np[b, v, 0] if v.any() else np.zeros(1, dtype=np.float32)
            # Pad to model's expected feature count with zeros
            feat = np.zeros(model.model.n_features_in_, dtype=np.float32)
            feat[0] = float(np.mean(flux))
            feat[1] = float(np.std(flux))
            feat[2] = float(np.max(flux))
            feat[3] = float(np.min(flux))
            rows.append(feat)

        X     = np.vstack(rows)
        probs = model.predict_proba(X)
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    mean_probs = np.concatenate(all_probs,  axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    return {
        "mean_probs" : mean_probs,
        "variance"   : np.zeros_like(mean_probs),
        "entropy"    : _entropy(mean_probs),
        "labels"     : labels_arr,
    }


# ---------------------------------------------------------------------------
# PyTorch models (MC Dropout)
# ---------------------------------------------------------------------------

def _torch_mc_predict(
    model      : nn.Module,
    dataloader : DataLoader,
    n_samples  : int,
    device     : torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run N stochastic forward passes over the entire DataLoader.

    Dropout layers are kept active throughout all passes.

    Parameters
    ----------
    model      : Any ``nn.Module`` with Dropout layers (Transformer, Astromer,
                 Moirai classifiers).
    dataloader : DataLoader yielding ``(sequence, mask, label)`` triplets.
    n_samples  : Number of MC samples (stochastic forward passes).
    device     : Compute device.

    Returns
    -------
    dict with keys ``mean_probs, variance, entropy, labels``.
    """
    model.eval()
    _enable_dropout(model)   # re-enable dropout layers after .eval()

    # Collect all batches first to know tensor shapes
    all_seqs   = []
    all_masks  = []
    all_labels = []

    with torch.no_grad():
        for seqs, masks, labels in dataloader:
            all_seqs.append(seqs)
            all_masks.append(masks)
            all_labels.append(labels)

    all_seqs   = torch.cat(all_seqs,   dim=0)   # (N_total, L, D)
    all_masks  = torch.cat(all_masks,  dim=0)   # (N_total, L)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    N_total  = all_seqs.shape[0]
    n_classes = None
    sample_probs: list = []   # list of (N_total, n_classes) arrays

    print(f"[uncertainty] Running {n_samples} MC forward passes …")

    for s in range(n_samples):
        batch_probs = []
        start = 0
        bs    = dataloader.batch_size or 64

        while start < N_total:
            end   = min(start + bs, N_total)
            x_b   = all_seqs[start:end].to(device)
            m_b   = all_masks[start:end].to(device)

            with torch.no_grad():
                logits = model(x_b, m_b)              # (batch, n_classes)
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_probs.append(probs)
            start = end

        pass_probs = np.concatenate(batch_probs, axis=0)  # (N_total, C)
        if n_classes is None:
            n_classes = pass_probs.shape[1]
        sample_probs.append(pass_probs)

        if (s + 1) % 10 == 0:
            print(f"  {s + 1}/{n_samples} passes done.")

    # Stack: (n_samples, N_total, C)
    stacked    = np.stack(sample_probs, axis=0)
    mean_probs = stacked.mean(axis=0)              # (N_total, C)
    variance   = stacked.var(axis=0)               # (N_total, C)
    entropy    = _entropy(mean_probs)              # (N_total,)

    return {
        "mean_probs" : mean_probs.astype(np.float32),
        "variance"   : variance.astype(np.float32),
        "entropy"    : entropy.astype(np.float32),
        "labels"     : all_labels,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mc_predict(
    model      : Union[nn.Module, XGBoostClassifier],
    dataloader : DataLoader,
    n_samples  : int = config.MC_SAMPLES,
    device     : Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """
    Run MC Dropout inference on *model* over all batches in *dataloader*.

    Works with any of the four model types:
    - ``XGBoostClassifier``      → deterministic; variance = 0.
    - ``TransformerClassifier``  → MC Dropout (N forward passes).
    - ``AstroClassifier``        → MC Dropout on classification head.
    - ``MoiraiClassifier``       → MC Dropout on classification head.

    Parameters
    ----------
    model      : Fitted model instance.
    dataloader : PyTorch DataLoader yielding ``(sequence, mask, label)``.
    n_samples  : Number of stochastic forward passes (MC samples).
    device     : Compute device.  Auto-detected if None.

    Returns
    -------
    dict with keys:

    ``mean_probs``  np.ndarray  ``(n_samples, n_classes)`` float32
        Mean predicted class probabilities across MC samples.

    ``variance``    np.ndarray  ``(n_samples, n_classes)`` float32
        Per-class predictive variance across MC samples
        (proxy for *epistemic* uncertainty).

    ``entropy``     np.ndarray  ``(n_samples,)`` float32
        Predictive entropy H(y|x) — total uncertainty.

    ``labels``      np.ndarray  ``(n_samples,)`` int64
        Ground-truth class indices.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model, XGBoostClassifier):
        print("[uncertainty] XGBoost model — running deterministic inference.")
        return _xgb_predict(model, dataloader)

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"mc_predict expects an nn.Module or XGBoostClassifier, got {type(model)}."
        )

    return _torch_mc_predict(model, dataloader, n_samples, device)
