"""
calibration.py — Reliability diagrams and Expected Calibration Error (ECE).

A well-calibrated classifier assigns probability p to events that actually
occur with frequency p.  The reliability diagram visualises the gap between
predicted confidence and observed accuracy across confidence bins.

Public API
----------
    compute_ece(probs, labels, n_bins=10) -> float
    plot_reliability_diagram(probs, labels, model_name, save_path=None)
    plot_all_models(results_dict, save_dir=None)
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Colab and headless)
import matplotlib.pyplot as plt
import numpy as np

import config


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def compute_ece(
    probs  : np.ndarray,
    labels : np.ndarray,
    n_bins : int = config.N_BINS,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE is defined as the weighted average of the absolute difference between
    mean confidence and observed accuracy across ``n_bins`` equally-spaced
    confidence buckets:

        ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples whose *maximum* predicted probability
    falls in the b-th confidence bin.

    Parameters
    ----------
    probs  : np.ndarray  shape ``(n_samples, n_classes)`` — predicted probs.
    labels : np.ndarray  shape ``(n_samples,)``           — true class indices.
    n_bins : int         Number of confidence bins (default: 10).

    Returns
    -------
    float  ECE in [0, 1].  Lower is better; 0 = perfectly calibrated.
    """
    confidence   = probs.max(axis=1)           # max predicted probability
    predictions  = probs.argmax(axis=1)        # predicted class
    correct      = (predictions == labels).astype(float)
    n_samples    = len(labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        n_in_bin = mask.sum()
        acc_bin  = correct[mask].mean()
        conf_bin = confidence[mask].mean()
        ece     += (n_in_bin / n_samples) * abs(acc_bin - conf_bin)

    return float(ece)


# ---------------------------------------------------------------------------
# Reliability-diagram helpers
# ---------------------------------------------------------------------------

def _calibration_curve(
    probs  : np.ndarray,
    labels : np.ndarray,
    n_bins : int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bin (mean_confidence, fraction_correct, bin_size) arrays.

    Returns three arrays of length ``n_bins``, where bins with no samples
    have NaN entries.
    """
    confidence  = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)
    n_samples   = len(labels)

    bin_edges    = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf    = np.full(n_bins, np.nan)
    frac_correct = np.full(n_bins, np.nan)
    bin_fracs    = np.zeros(n_bins)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        mean_conf[i]    = confidence[mask].mean()
        frac_correct[i] = correct[mask].mean()
        bin_fracs[i]    = mask.sum() / n_samples

    return mean_conf, frac_correct, bin_fracs


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    probs      : np.ndarray,
    labels     : np.ndarray,
    model_name : str,
    n_bins     : int = config.N_BINS,
    save_path  : Optional[str] = None,
) -> None:
    """
    Plot a reliability (calibration) diagram for *model_name*.

    The diagram shows:
    - Blue bars: fraction of correct predictions per confidence bin.
    - Red dashed line: perfect calibration (y = x).
    - Gap between bars and diagonal = calibration error.

    Parameters
    ----------
    probs      : np.ndarray  ``(n_samples, n_classes)`` predicted probabilities.
    labels     : np.ndarray  ``(n_samples,)``           true class indices.
    model_name : Display name used in the plot title and saved filename.
    n_bins     : Number of confidence bins.
    save_path  : If given, save the figure to this path; otherwise save to
                 ``config.FIG_DIR``.
    """
    ece = compute_ece(probs, labels, n_bins=n_bins)
    mean_conf, frac_correct, bin_fracs = _calibration_curve(probs, labels, n_bins)

    bin_centres = np.linspace(0.0, 1.0, n_bins + 1)[:-1] + 0.5 / n_bins
    bar_width   = 1.0 / n_bins * 0.9

    fig, ax = plt.subplots(figsize=(6, 5))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect calibration")

    # Calibration bars — only non-NaN bins
    valid = ~np.isnan(frac_correct)
    ax.bar(
        bin_centres[valid],
        frac_correct[valid],
        width   = bar_width,
        alpha   = 0.7,
        color   = "steelblue",
        label   = "Fraction correct",
        align   = "center",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence", fontsize=12)
    ax.set_ylabel("Fraction of correct predictions", fontsize=12)
    ax.set_title(f"Reliability Diagram — {model_name}\nECE = {ece:.4f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        fname     = f"reliability_{model_name.replace(' ', '_').lower()}.png"
        save_path = os.path.join(config.FIG_DIR, fname)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[calibration] Reliability diagram saved → {save_path}  (ECE={ece:.4f})")


def plot_all_models(
    results_dict : Dict[str, Dict[str, np.ndarray]],
    n_bins       : int = config.N_BINS,
    save_dir     : Optional[str] = None,
) -> None:
    """
    Plot reliability diagrams for all models on a single multi-panel figure
    and also save individual diagrams.

    Parameters
    ----------
    results_dict : Mapping  model_name → mc_predict output dict
                   (keys: ``mean_probs``, ``labels``).
    n_bins       : Number of confidence bins.
    save_dir     : Directory for saving figures.  Defaults to config.FIG_DIR.
    """
    if save_dir is None:
        save_dir = config.FIG_DIR
    os.makedirs(save_dir, exist_ok=True)

    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)

    for ax, (model_name, result) in zip(axes[0], results_dict.items()):
        probs  = result["mean_probs"]
        labels = result["labels"]
        ece    = compute_ece(probs, labels, n_bins)
        mean_conf, frac_correct, _ = _calibration_curve(probs, labels, n_bins)

        bin_centres = np.linspace(0.0, 1.0, n_bins + 1)[:-1] + 0.5 / n_bins
        bar_width   = 1.0 / n_bins * 0.9
        valid       = ~np.isnan(frac_correct)

        ax.plot([0, 1], [0, 1], "r--", linewidth=1.2)
        ax.bar(
            bin_centres[valid], frac_correct[valid],
            width=bar_width, alpha=0.7, color="steelblue",
        )
        ax.set_title(f"{model_name}\nECE={ece:.4f}", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Also save individual diagram
        plot_reliability_diagram(
            probs, labels, model_name, n_bins=n_bins,
            save_path=os.path.join(save_dir, f"reliability_{model_name.lower()}.png"),
        )

    fig.suptitle("Reliability Diagrams — All Models", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, "reliability_all_models.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[calibration] Combined reliability diagram saved → {path}")
