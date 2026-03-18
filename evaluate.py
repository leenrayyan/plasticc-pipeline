"""
evaluate.py — Full evaluation of all four models across truncation fractions.

For each model × truncation fraction the following metrics are computed:
    - Macro F1
    - PR-AUC (macro OvR)
    - Per-class recall
    - ECE (Expected Calibration Error)
    - Top-K rare-class recall at K ∈ {50, 100, 200}

All results are collected into a single pandas DataFrame and all figures
(reliability diagrams, Top-K curves, accuracy vs truncation) are generated.

Public API
----------
    evaluate_all(fractions, models_to_eval, device) -> pd.DataFrame
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import config
import calibration
import features as feat_module
import prioritization
import uncertainty
from data_loader import (
    build_label_map,
    build_sequences,
    handle_missing,
    load_raw_data,
    normalize_flux,
    truncate_observations,
    PLAsTiCCDataset,
)
from models import XGBoostClassifier, build_model
from train import train_model, save_history, plot_training_curves
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Models that use hand-crafted features (XGBoost only)
_XGB_MODELS = {"xgboost"}
# Models that use sequence DataLoaders
_DL_MODELS  = {"transformer", "astromer", "moirai"}

TOPK_EVAL   = [50, 100, 200]   # K values for per-row recall columns


# ---------------------------------------------------------------------------
# Single-model, single-fraction evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    model_name   : str,
    fraction     : float,
    label_map    : Dict[int, int],
    classes      : List[int],
    device       : torch.device,
    verbose      : bool = True,
) -> Dict:
    """
    Train (or load) *model_name* at *fraction*, run MC-Dropout inference,
    and return a dict of metrics.

    Parameters
    ----------
    model_name : One of ``"xgboost"``, ``"transformer"``, ``"astromer"``,
                 ``"moirai"``.
    fraction   : Observation fraction in {0.1, 0.3, 0.5, 1.0}.
    label_map  : Raw PLAsTiCC target → class index.
    classes    : Sorted list of raw PLAsTiCC targets.
    device     : Compute device.
    verbose    : Print progress messages.

    Returns
    -------
    dict with metric keys (scalars and small arrays).
    """
    n_classes = len(classes)
    tag       = f"{model_name}_f{fraction}"

    # ---- Load raw data (re-used across models at same fraction) -----------
    lc, meta = load_raw_data()
    lc       = handle_missing(lc)
    lc_raw   = lc.copy()               # keep un-normalised for XGBoost features
    lc       = normalize_flux(lc)
    lc       = truncate_observations(lc, fraction)
    lc_raw   = truncate_observations(lc_raw, fraction)

    sequences  = build_sequences(lc)
    labels_map = {
        int(r["object_id"]): label_map[r["target"]]
        for _, r in meta.iterrows()
        if int(r["object_id"]) in sequences and r["target"] in label_map
    }
    sequences = {oid: sequences[oid] for oid in labels_map}

    obj_ids      = list(labels_map.keys())
    strat_labels = [labels_map[o] for o in obj_ids]
    train_ids, test_ids = train_test_split(
        obj_ids, test_size=config.TEST_SIZE,
        random_state=config.SEED, stratify=strat_labels,
    )

    # ---- Build DataLoaders -------------------------------------------------
    from torch.utils.data import DataLoader
    train_ds = PLAsTiCCDataset(
        {o: sequences[o] for o in train_ids},
        {o: labels_map[o] for o in train_ids},
    )
    test_ds  = PLAsTiCCDataset(
        {o: sequences[o] for o in test_ids},
        {o: labels_map[o] for o in test_ids},
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- XGBoost branch ----------------------------------------------------
    if model_name in _XGB_MODELS:
        if verbose:
            print(f"[evaluate] XGBoost | fraction={fraction}")

        X, y, obj_arr = feat_module.build_feature_matrix(lc_raw, meta, label_map)
        # Align train / test splits
        obj_set_train = set(train_ids)
        obj_set_test  = set(test_ids)
        obj_arr_int   = obj_arr.astype(int)
        train_mask    = np.isin(obj_arr_int, list(obj_set_train))
        test_mask     = np.isin(obj_arr_int, list(obj_set_test))

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        model = XGBoostClassifier(n_classes=n_classes)
        if verbose:
            print(f"  Fitting XGBoost on {len(X_tr)} samples …")
        model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)

        probs      = model.predict_proba(X_te)
        labels_arr = y_te
        variance   = np.zeros_like(probs)
        ent        = uncertainty._entropy(probs)

    # ---- Deep-learning branch ----------------------------------------------
    else:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{tag}.pt")
        model     = build_model(model_name, n_classes, device)

        if os.path.exists(ckpt_path):
            if verbose:
                print(f"[evaluate] Loading checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            if verbose:
                print(f"[evaluate] Training {model_name} | fraction={fraction}")
            # Use test_loader as validation during training
            model, history = train_model(
                model, train_loader, test_loader,
                model_name=tag, n_classes=n_classes, device=device,
            )
            save_history(history, tag)
            plot_training_curves(history, tag)

        # MC Dropout inference
        if verbose:
            print(f"[evaluate] MC Dropout inference ({config.MC_SAMPLES} passes) …")
        result     = uncertainty.mc_predict(model, test_loader, device=device)
        probs      = result["mean_probs"]
        variance   = result["variance"]
        ent        = result["entropy"]
        labels_arr = result["labels"]

    # ---- Metrics -----------------------------------------------------------
    preds = probs.argmax(axis=1)

    macro_f1 = float(f1_score(labels_arr, preds, average="macro", zero_division=0))

    # PR-AUC (macro one-vs-rest)
    classes_present = sorted(np.unique(labels_arr).tolist())
    y_bin = label_binarize(labels_arr, classes=list(range(n_classes)))
    try:
        prauc = float(average_precision_score(y_bin, probs, average="macro"))
    except Exception:
        prauc = float("nan")

    per_class_recall = recall_score(
        labels_arr, preds, average=None, labels=list(range(n_classes)), zero_division=0
    ).tolist()

    ece = calibration.compute_ece(probs, labels_arr)

    # Top-K rare-class recall
    rare_idx = [label_map[rc] for rc in config.RARE_CLASSES if rc in label_map]
    conf_scores = probs.max(axis=1)
    unc_scores  = conf_scores / (1.0 + ent)
    topk_recalls_conf = {}
    topk_recalls_unc  = {}
    for K in TOPK_EVAL:
        topk_recalls_conf[K] = float(
            prioritization.compute_topk_recall(conf_scores, labels_arr, rare_idx, [K])[0]
        )
        topk_recalls_unc[K]  = float(
            prioritization.compute_topk_recall(unc_scores,  labels_arr, rare_idx, [K])[0]
        )

    if verbose:
        print(
            f"  ✓ {model_name:<15s} f={fraction}  "
            f"F1={macro_f1:.4f}  PR-AUC={prauc:.4f}  ECE={ece:.4f}  "
            f"recall@50(conf)={topk_recalls_conf[50]:.3f}"
        )

    row = {
        "model":           model_name,
        "truncation":      fraction,
        "macro_f1":        macro_f1,
        "pr_auc":          prauc,
        "ece":             ece,
        "per_class_recall": per_class_recall,
    }
    for K in TOPK_EVAL:
        row[f"recall_conf@{K}"] = topk_recalls_conf[K]
        row[f"recall_unc@{K}"]  = topk_recalls_unc[K]

    # Store for calibration figure
    row["_probs"]  = probs
    row["_labels"] = labels_arr
    row["_entropy"]= ent

    return row


# ---------------------------------------------------------------------------
# Full evaluation orchestrator
# ---------------------------------------------------------------------------

def evaluate_all(
    fractions      : List[float]         = config.TRUNCATION_FRACTIONS,
    models_to_eval : List[str]           = None,
    device         : Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Run full evaluation for all models across all truncation fractions.

    Parameters
    ----------
    fractions      : List of early-truncation fractions to evaluate.
    models_to_eval : List of model names.  Defaults to all four.
    device         : Compute device.  Auto-detected if None.

    Returns
    -------
    pd.DataFrame with one row per (model, fraction) combination and
    columns for every metric.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if models_to_eval is None:
        models_to_eval = ["xgboost", "transformer", "astromer", "moirai"]

    # Load label map once (from full data)
    lc_full, meta = load_raw_data()
    lc_full       = handle_missing(lc_full)
    label_map, classes = build_label_map(meta)
    n_classes = len(classes)
    print(f"[evaluate] {n_classes} classes, {len(fractions)} fractions, "
          f"{len(models_to_eval)} models → {n_classes * len(fractions) * len(models_to_eval)} runs")

    rows     : List[Dict] = []
    all_mc   : Dict[str, Dict] = {}   # for calibration plots

    for fraction in fractions:
        for model_name in models_to_eval:
            try:
                row = _evaluate_one(
                    model_name, fraction, label_map, classes, device, verbose=True
                )
                # Collect MC results for calibration / prioritization plots
                # at the full-data fraction (most informative)
                if fraction == 1.0:
                    all_mc[f"{model_name}"] = {
                        "mean_probs": row.pop("_probs"),
                        "labels":     row.pop("_labels"),
                        "entropy":    row.pop("_entropy"),
                        "variance":   np.zeros(1),   # placeholder
                    }
                else:
                    row.pop("_probs",   None)
                    row.pop("_labels",  None)
                    row.pop("_entropy", None)
                rows.append(row)

            except Exception as exc:
                print(f"[evaluate] WARNING: {model_name} @ f={fraction} failed: {exc}")

    results_df = pd.DataFrame(rows)

    # ---- Save results table ------------------------------------------------
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(config.RESULTS_DIR, "results_table.csv")
    results_df.drop(columns=["per_class_recall"], errors="ignore").to_csv(
        table_path, index=False
    )
    print(f"\n[evaluate] Results table saved → {table_path}")

    # ---- Generate figures (using fraction=1.0 results) --------------------
    if all_mc:
        print("\n[evaluate] Generating calibration diagrams …")
        calibration.plot_all_models(all_mc)

        print("[evaluate] Generating Top-K recall curves …")
        prioritization.run_prioritization(all_mc, label_map)

    print("[evaluate] Generating accuracy vs truncation plots …")
    for metric in ["macro_f1", "pr_auc", "ece"]:
        if metric in results_df.columns:
            prioritization.plot_accuracy_vs_truncation(results_df, metric=metric)

    print("\n[evaluate] All evaluations complete.")
    return results_df
