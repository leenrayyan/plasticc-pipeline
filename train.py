"""
train.py — Training loop for PyTorch models (Transformer, Astromer, Moirai).

Features
--------
- Cross-entropy loss with class weights to handle PLAsTiCC class imbalance.
- Adam optimiser with cosine annealing LR schedule (linear warm-up).
- Early stopping on validation loss (patience = config.EARLY_STOPPING_PATIENCE).
- Saves the best checkpoint per model to config.CHECKPOINT_DIR.
- Logs: train loss, val loss, val macro-F1 per epoch.

The XGBoost baseline (Model 1) is trained separately via its own ``fit``
method — see ``evaluate.py``.

Public API
----------
    train_model(model, train_loader, val_loader, model_name, n_classes, device)
        -> (trained_model, history_dict)
"""

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_class_weights(loader: DataLoader, n_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a DataLoader.

    Weight_c = n_samples / (n_classes * n_samples_c)

    Returns a float Tensor of shape ``(n_classes,)``.
    """
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for _, _, labels in loader:
        for c in range(n_classes):
            counts[c] += (labels == c).sum().item()

    n_total = counts.sum()
    weights = n_total / (n_classes * counts.clamp(min=1))
    weights = weights / weights.mean()   # normalise so mean weight ≈ 1
    return weights.float()


def _make_lr_lambda(
    warmup_epochs : int,
    total_epochs  : int,
) -> callable:
    """
    Return a learning-rate schedule function for ``LambdaLR``.

    Linear warm-up for ``warmup_epochs`` epochs, then cosine annealing.
    """
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _run_epoch(
    model      : nn.Module,
    loader     : DataLoader,
    criterion  : nn.Module,
    optimiser  : Optional[torch.optim.Optimizer],
    device     : torch.device,
    n_classes  : int,
    is_train   : bool,
) -> Tuple[float, float]:
    """
    Run one training or validation epoch.

    Returns
    -------
    (mean_loss, macro_f1)
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches  = 0
    all_preds  = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for seqs, masks, labels in loader:
            seqs   = seqs.to(device)
            masks  = masks.to(device)
            labels = labels.to(device)

            logits = model(seqs, masks)                  # (B, n_classes)
            loss   = criterion(logits, labels)

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            total_loss += loss.item()
            n_batches  += 1
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    mean_loss = total_loss / max(n_batches, 1)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return float(mean_loss), float(macro_f1)


# ---------------------------------------------------------------------------
# Public training function
# ---------------------------------------------------------------------------

def train_model(
    model        : nn.Module,
    train_loader : DataLoader,
    val_loader   : DataLoader,
    model_name   : str,
    n_classes    : int,
    device       : Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Train a PyTorch model with early stopping and cosine LR scheduling.

    Parameters
    ----------
    model        : Instantiated (untrained) PyTorch model.
    train_loader : DataLoader for training data.
    val_loader   : DataLoader for validation data.
    model_name   : Name used for checkpoint filename and logging.
    n_classes    : Number of output classes.
    device       : Compute device.  Auto-detected if None.

    Returns
    -------
    model   : The best checkpoint model (loaded from disk).
    history : dict with keys ``train_loss``, ``val_loss``, ``val_f1``, ``lr``.
              Each value is a list of per-epoch scalars.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # ---- Class-weighted loss -----------------------------------------------
    print(f"[train] Computing class weights for {model_name} …")
    class_weights = _compute_class_weights(train_loader, n_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Optimiser + scheduler ---------------------------------------------
    optimiser = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config.LEARNING_RATE,
    )
    scheduler = LambdaLR(
        optimiser,
        lr_lambda = _make_lr_lambda(config.LR_WARMUP_EPOCHS, config.MAX_EPOCHS),
    )

    # ---- Checkpoint path ---------------------------------------------------
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}.pt")

    # ---- Training loop -----------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}
    best_val_loss  = float("inf")
    patience_count = 0

    print(f"\n[train] Starting training: {model_name} | device={device}")
    print(f"  Max epochs={config.MAX_EPOCHS}, patience={config.EARLY_STOPPING_PATIENCE}")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_f1 = _run_epoch(
            model, train_loader, criterion, optimiser, device, n_classes, is_train=True
        )
        val_loss, val_f1 = _run_epoch(
            model, val_loader, criterion, None, device, n_classes, is_train=False
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:03d}/{config.MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_F1={val_f1:.4f}  lr={current_lr:.2e}  "
            f"({elapsed:.1f}s)"
        )

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ✓ Checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"[train] Early stopping at epoch {epoch} "
                    f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)."
                )
                break

    # ---- Reload best checkpoint -------------------------------------------
    print(f"[train] Loading best checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[train] Done — {model_name}  best val_loss={best_val_loss:.4f}\n")

    return model, history


def save_history(history: Dict, model_name: str) -> None:
    """
    Persist training history as a NumPy .npz file.

    Parameters
    ----------
    history    : Output of ``train_model``.
    model_name : Used for the filename.
    """
    path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_history.npz")
    np.savez(path, **{k: np.array(v) for k, v in history.items()})
    print(f"[train] History saved → {path}")


def plot_training_curves(history: Dict, model_name: str) -> None:
    """
    Plot and save train/val loss and val F1 curves.

    Parameters
    ----------
    history    : Output of ``train_model``.
    model_name : Used for the figure title and filename.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_f1"], color="green", label="Val macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title(f"{model_name} — Val Macro-F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(config.FIG_DIR, f"training_{model_name.lower()}.png")
    os.makedirs(config.FIG_DIR, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Training curves saved → {path}")
