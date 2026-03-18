"""
main.py — End-to-end orchestration of the PLAsTiCC classification pipeline.

Execution order
---------------
1.  Load & preprocess data.
2.  Train all four models (XGBoost, Transformer, Astromer, Moirai).
3.  Run MC Dropout inference on the test set.
4.  Compute calibration metrics and generate reliability diagrams.
5.  Run the follow-up budget / prioritization simulation.
6.  Generate all result tables and figures.

Usage
-----
Run all truncation fractions:
    python main.py

Run a single truncation fraction (faster for debugging):
    python main.py --truncation 0.3

Run only specific models:
    python main.py --models transformer xgboost

Skip training (load saved checkpoints):
    python main.py --skip-training

Results are written to:
    results/results_table.csv
    results/figures/
    checkpoints/
"""

import argparse
import os
import random
import sys
import time
from typing import List, Optional

import numpy as np
import torch

import config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = config.SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PLAsTiCC Uncertainty-Aware Transient Classification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--truncation",
        type    = float,
        default = None,
        metavar = "F",
        help    = "Run only this observation fraction (e.g. 0.3). "
                  "If omitted, all fractions in config.TRUNCATION_FRACTIONS are used.",
    )
    parser.add_argument(
        "--models",
        nargs   = "+",
        default = None,
        choices = ["xgboost", "transformer", "astromer", "moirai"],
        metavar = "MODEL",
        help    = "Space-separated list of models to evaluate. "
                  "Defaults to all four.",
    )
    parser.add_argument(
        "--skip-training",
        action  = "store_true",
        help    = "Skip training and load existing checkpoints (evaluate only).",
    )
    parser.add_argument(
        "--mc-samples",
        type    = int,
        default = config.MC_SAMPLES,
        help    = "Number of MC Dropout forward passes.",
    )
    parser.add_argument(
        "--batch-size",
        type    = int,
        default = config.BATCH_SIZE,
        help    = "Mini-batch size.",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = config.SEED,
        help    = "Global random seed.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[main] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[main] No GPU detected — running on CPU.")
    return dev


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full classification and evaluation pipeline."""

    # ---- Setup -------------------------------------------------------------
    set_seeds(args.seed)

    # Propagate any CLI overrides to config module at runtime
    config.MC_SAMPLES  = args.mc_samples
    config.BATCH_SIZE  = args.batch_size

    device = get_device()

    fractions = [args.truncation] if args.truncation is not None else config.TRUNCATION_FRACTIONS
    models    = args.models if args.models is not None else ["xgboost", "transformer", "astromer", "moirai"]

    print("\n" + "=" * 65)
    print("  PLAsTiCC Uncertainty-Aware Early Transient Classification")
    print("=" * 65)
    print(f"  Fractions  : {fractions}")
    print(f"  Models     : {models}")
    print(f"  MC samples : {config.MC_SAMPLES}")
    print(f"  Seed       : {args.seed}")
    print(f"  Device     : {device}")
    print("=" * 65 + "\n")

    t_start = time.time()

    # ---- Step 1: Verify data files exist -----------------------------------
    print("[main] Step 1 — Verifying data files …")
    from data_loader import _verify_paths
    _verify_paths()
    print("  Data files found.\n")

    # ---- Steps 2–6: Train + Evaluate ---------------------------------------
    # evaluate_all handles training internally (or loads checkpoints if present)
    print("[main] Steps 2–6 — Train / Evaluate / Calibrate / Prioritize …\n")
    from evaluate import evaluate_all
    results_df = evaluate_all(
        fractions      = fractions,
        models_to_eval = models,
        device         = device,
    )

    # ---- Summary table -----------------------------------------------------
    display_cols = [c for c in results_df.columns if not c.startswith("per_class")]
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(results_df[display_cols].to_string(index=False))

    elapsed = time.time() - t_start
    print(f"\n[main] Pipeline finished in {elapsed / 60:.1f} minutes.")
    print(f"[main] All artefacts in: {config.RESULTS_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Ensure output dirs exist
    for d in [config.RESULTS_DIR, config.FIG_DIR, config.CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[main] Fatal error: {exc}")
        raise
