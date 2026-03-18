"""
prioritization.py — Follow-up budget simulation for rare transient detection.

Motivation
----------
Telescopes generate thousands of alerts per night.  Follow-up resources are
scarce, so observers must rank alerts and observe only the top-K.  A good
classifier should concentrate rare, high-value events (kilonovae, TDEs) near
the top of the ranked list.

Two ranking strategies
----------------------
(a) Confidence only:
        score_i = max_c  p(c | x_i)

(b) Uncertainty-weighted confidence:
        score_i = max_c  p(c | x_i)  /  (1 + H(y | x_i))

    where H is the predictive entropy.  Dividing by entropy penalises
    high-confidence but also high-uncertainty predictions.

For each strategy and each budget K ∈ {10, 20, …, 500}, we compute:

    Rare-class recall @ K = |{top-K} ∩ {true rare events}| / |{true rare events}|

This is the primary figure of merit for the paper.

Public API
----------
    compute_topk_recall(scores, labels, rare_label_indices, budget_range) -> np.ndarray
    run_prioritization(results_dict, label_map, budget_range, save_dir) -> dict
    plot_topk_curves(recall_dict, budget_range, save_dir)
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def compute_topk_recall(
    scores             : np.ndarray,
    labels             : np.ndarray,
    rare_label_indices : List[int],
    budget_range       : List[int] = config.BUDGET_RANGE,
) -> np.ndarray:
    """
    Compute rare-class recall for every budget K in *budget_range*.

    Parameters
    ----------
    scores             : 1-D array of ranking scores (higher → more likely rare).
    labels             : 1-D integer array of true class indices.
    rare_label_indices : List of *class index* values (0-based) for rare classes.
    budget_range       : Sequence of K values to evaluate.

    Returns
    -------
    recalls : np.ndarray  shape ``(len(budget_range),)``
        Fraction of true rare events captured in the top-K ranked alerts,
        for each K.  Returns NaN when there are no rare events in the batch.
    """
    n_total = len(labels)
    is_rare = np.isin(labels, rare_label_indices)
    n_rare  = is_rare.sum()

    if n_rare == 0:
        return np.full(len(budget_range), np.nan, dtype=np.float32)

    # Rank from highest score to lowest
    ranked_order = np.argsort(scores)[::-1]   # descending

    recalls = np.empty(len(budget_range), dtype=np.float32)
    for i, K in enumerate(budget_range):
        K_eff      = min(K, n_total)
        top_k_idx  = ranked_order[:K_eff]
        n_rare_hit = is_rare[top_k_idx].sum()
        recalls[i] = n_rare_hit / n_rare

    return recalls


# ---------------------------------------------------------------------------
# Score builders
# ---------------------------------------------------------------------------

def _confidence_scores(mean_probs: np.ndarray) -> np.ndarray:
    """Return max predicted probability per sample."""
    return mean_probs.max(axis=1)


def _uncertainty_weighted_scores(
    mean_probs : np.ndarray,
    entropy    : np.ndarray,
) -> np.ndarray:
    """
    Return confidence divided by (1 + entropy).

    Higher entropy → lower effective score, so uncertain predictions are
    deprioritised.
    """
    confidence = mean_probs.max(axis=1)
    return confidence / (1.0 + entropy)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_prioritization(
    results_dict : Dict[str, Dict[str, np.ndarray]],
    label_map    : Dict[int, int],
    budget_range : List[int]         = config.BUDGET_RANGE,
    save_dir     : Optional[str]     = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run the full prioritization simulation for all models and both strategies.

    Parameters
    ----------
    results_dict : Mapping  model_name → mc_predict output
                   (keys: ``mean_probs``, ``entropy``, ``labels``).
    label_map    : Raw PLAsTiCC target → 0-based class index
                   (from ``data_loader.build_label_map``).
    budget_range : Sequence of K values.
    save_dir     : Directory to save the Top-K recall figure.

    Returns
    -------
    recall_dict : Nested mapping
                  ``model_name → strategy → np.ndarray(len(budget_range))``
    """
    if save_dir is None:
        save_dir = config.FIG_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Map raw rare-class integers to 0-based indices
    rare_label_indices = [
        label_map[rc] for rc in config.RARE_CLASSES if rc in label_map
    ]
    if not rare_label_indices:
        raise ValueError(
            f"None of the rare classes {config.RARE_CLASSES} were found in label_map."
            " Check that the dataset contains TDE (15) and KN (64) examples."
        )

    recall_dict: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, result in results_dict.items():
        mean_probs = result["mean_probs"]
        entropy    = result["entropy"]
        labels     = result["labels"]

        scores_conf = _confidence_scores(mean_probs)
        scores_unc  = _uncertainty_weighted_scores(mean_probs, entropy)

        recalls_conf = compute_topk_recall(scores_conf, labels, rare_label_indices, budget_range)
        recalls_unc  = compute_topk_recall(scores_unc,  labels, rare_label_indices, budget_range)

        recall_dict[model_name] = {
            "confidence":             recalls_conf,
            "uncertainty_weighted":   recalls_unc,
        }

        n_rare = np.isin(labels, rare_label_indices).sum()
        print(
            f"[prioritization] {model_name:<20s}  "
            f"rare events={n_rare}  "
            f"recall@50  conf={recalls_conf[4]:.3f}  unc_wt={recalls_unc[4]:.3f}"
        )

    plot_topk_curves(recall_dict, budget_range, save_dir)
    return recall_dict


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_topk_curves(
    recall_dict  : Dict[str, Dict[str, np.ndarray]],
    budget_range : List[int]     = config.BUDGET_RANGE,
    save_dir     : Optional[str] = None,
) -> None:
    """
    Plot Top-K rare-class recall curves for all models and both strategies.

    Produces two panels:
    - Left:  confidence-only ranking.
    - Right: uncertainty-weighted ranking.

    Parameters
    ----------
    recall_dict  : Output of ``run_prioritization``.
    budget_range : K values on the x-axis.
    save_dir     : Directory for saving the figure.
    """
    if save_dir is None:
        save_dir = config.FIG_DIR
    os.makedirs(save_dir, exist_ok=True)

    budgets   = np.array(budget_range)
    strategies = ["confidence", "uncertainty_weighted"]
    titles     = ["Strategy (a): Confidence only", "Strategy (b): Uncertainty-weighted"]
    colours    = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, strategy, title in zip(axes, strategies, titles):
        for (model_name, strat_dict), colour in zip(recall_dict.items(), colours):
            recalls = strat_dict.get(strategy)
            if recalls is None:
                continue
            ax.plot(budgets, recalls, label=model_name, linewidth=2, color=colour)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Follow-up budget K", fontsize=11)
        ax.set_ylabel("Rare-class recall", fontsize=11)
        ax.set_xlim(budgets[0], budgets[-1])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    rare_names = [
        config.PLASTICC_CLASSES.get(rc, str(rc)) for rc in config.RARE_CLASSES
    ]
    fig.suptitle(
        f"Top-K Rare-Class Recall  ({', '.join(rare_names)})",
        fontsize=14,
    )
    fig.tight_layout()

    save_path = os.path.join(save_dir, "topk_recall_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[prioritization] Top-K recall figure saved → {save_path}")


def plot_accuracy_vs_truncation(
    metric_table : "pd.DataFrame",  # noqa: F821
    metric       : str   = "macro_f1",
    save_dir     : Optional[str] = None,
) -> None:
    """
    Plot model accuracy (or any metric) as a function of observation fraction.

    Parameters
    ----------
    metric_table : DataFrame with columns
                   ``[model, truncation, <metric>, …]``.
    metric       : Column name to plot on the y-axis.
    save_dir     : Save directory.
    """
    import pandas as pd

    if save_dir is None:
        save_dir = config.FIG_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, grp in metric_table.groupby("model"):
        grp = grp.sort_values("truncation")
        ax.plot(
            grp["truncation"] * 100,
            grp[metric],
            marker    = "o",
            linewidth = 2,
            label     = model_name,
        )

    ax.set_xlabel("Observation fraction (%)", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"{metric.replace('_', ' ').title()} vs. Observation Fraction", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(save_dir, f"{metric}_vs_truncation.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[prioritization] {metric} vs truncation figure saved → {save_path}")
