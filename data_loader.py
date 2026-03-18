"""
data_loader.py — PLAsTiCC data loading, preprocessing, and DataLoader creation.

Pipeline
--------
1. Load training_set.csv + training_set_metadata.csv and merge on object_id.
2. Handle missing values (fill flux/flux_err with 0; drop rows with NaN mjd).
3. Normalise flux per object with z-score (zero-mean, unit-variance).
4. Early truncation: keep only the first *fraction* of each object's
   observations when sorted by MJD (simulates early-alert classification).
5. Build per-object sequences of shape (n_obs, INPUT_DIM):
       [flux_norm, flux_err, time_delta_days, passband_norm]
6. Pad / truncate every sequence to MAX_SEQ_LEN for batching.
7. Object-level stratified train/test split — no data leakage.
8. Return PyTorch DataLoaders that yield (sequence, mask, label) triplets:
       sequence : Tensor[MAX_SEQ_LEN, INPUT_DIM]   float32
       mask     : Tensor[MAX_SEQ_LEN]               bool   (True = padded)
       label    : Tensor[]                           int64
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import config


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def _verify_paths() -> None:
    """Raise FileNotFoundError with an informative message if data files are absent."""
    for path, name in [
        (config.TRAINING_SET_PATH, "training_set.csv"),
        (config.METADATA_PATH,     "training_set_metadata.csv"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nData file not found: {path}\n"
                f"Expected '{name}' inside {config.DATA_DIR}.\n"
                "Run colab_setup.py (Colab) or download the PLAsTiCC dataset "
                "from Kaggle and place the files in the data/ directory.\n"
                "See README.md for full instructions."
            )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two PLAsTiCC CSVs and return ``(light_curves, metadata)``."""
    _verify_paths()
    print("[data_loader] Loading light curves …")
    lc = pd.read_csv(config.TRAINING_SET_PATH)
    print(f"  Rows: {len(lc):,}   Objects: {lc['object_id'].nunique():,}")

    print("[data_loader] Loading metadata …")
    meta = pd.read_csv(config.METADATA_PATH)
    print(f"  Objects: {len(meta):,}")
    return lc, meta


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def handle_missing(lc: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values in the light-curve table.

    Strategy
    --------
    - Rows where ``mjd`` is NaN are unrecoverable and are dropped.
    - ``flux`` and ``flux_err`` NaNs are filled with 0.
    - ``detected`` NaNs are filled with 0 (assume not detected).
    """
    before = len(lc)
    lc = lc.dropna(subset=["mjd"]).copy()
    dropped = before - len(lc)
    if dropped:
        print(f"  [handle_missing] Dropped {dropped} rows with NaN mjd.")

    lc["flux"]     = lc["flux"].fillna(0.0)
    lc["flux_err"] = lc["flux_err"].fillna(0.0)
    detected_col = "detected_bool" if "detected_bool" in lc.columns else "detected"
    lc[detected_col] = lc[detected_col].fillna(0).astype(int)
    return lc


def normalize_flux(lc: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalise flux per object (mean=0, std=1).

    Objects with std=0 (e.g. a single observation or constant flux) have
    their flux set to 0 to avoid division by zero.
    """
    lc = lc.copy()

    def _zscore(grp: pd.DataFrame) -> pd.DataFrame:
        mu    = grp["flux"].mean()
        sigma = grp["flux"].std()
        if sigma == 0.0 or np.isnan(sigma):
            grp = grp.copy()
            grp["flux"] = 0.0
        else:
            grp = grp.copy()
            grp["flux"] = (grp["flux"] - mu) / sigma
        return grp

    lc = lc.groupby("object_id", group_keys=False).apply(_zscore)
    return lc


def truncate_observations(lc: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """
    Keep only the first ``fraction`` of each object's observations (by MJD).

    Parameters
    ----------
    lc       : Light-curve DataFrame with columns [object_id, mjd, …].
    fraction : Float in (0, 1].  1.0 → keep all observations.

    Returns
    -------
    pd.DataFrame with the same columns, fewer rows.
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    if fraction == 1.0:
        return lc

    def _keep_first(grp: pd.DataFrame) -> pd.DataFrame:
        grp   = grp.sort_values("mjd")
        n_keep = max(1, int(np.ceil(len(grp) * fraction)))
        return grp.iloc[:n_keep]

    return lc.groupby("object_id", group_keys=False).apply(_keep_first)


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_sequences(lc: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Convert light curves into fixed-format per-object sequences.

    Each time step is represented as a feature vector::

        [flux_norm,  flux_err,  time_delta_days,  passband_norm]

    where ``time_delta_days = mjd - first_mjd_of_object`` and
    ``passband_norm = passband / (N_PASSBANDS - 1)`` (maps 0–5 → 0–1).

    Returns
    -------
    dict : object_id → np.ndarray of shape (n_obs, INPUT_DIM), dtype float32
    """
    sequences: Dict[int, np.ndarray] = {}

    for obj_id, grp in lc.groupby("object_id"):
        grp        = grp.sort_values("mjd").reset_index(drop=True)
        t0         = grp["mjd"].iloc[0]
        time_delta = (grp["mjd"] - t0).values.astype(np.float32)
        flux       = grp["flux"].values.astype(np.float32)
        flux_err   = grp["flux_err"].values.astype(np.float32)
        passband   = (grp["passband"].values / (config.N_PASSBANDS - 1)).astype(np.float32)

        seq = np.stack([flux, flux_err, time_delta, passband], axis=1)  # (n_obs, 4)
        sequences[obj_id] = seq

    return sequences


def pad_sequence(
    seq: np.ndarray,
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad (or truncate) a sequence to ``max_len``.

    Returns
    -------
    padded : np.ndarray of shape (max_len, INPUT_DIM)  — zeros for padded positions
    mask   : np.ndarray of shape (max_len,)  bool       — True where padded
    """
    n      = len(seq)
    n_keep = min(n, max_len)
    padded = np.zeros((max_len, seq.shape[1]), dtype=np.float32)
    mask   = np.ones(max_len, dtype=bool)   # True = padding (ignored in attention)

    padded[:n_keep] = seq[:n_keep]
    mask[:n_keep]   = False                 # real observations are not masked
    return padded, mask


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def build_label_map(meta: pd.DataFrame) -> Tuple[Dict[int, int], List[int]]:
    """
    Map raw PLAsTiCC integer targets to contiguous 0-based class indices.

    Returns
    -------
    label_map : dict  raw_target → class_idx
    classes   : list  sorted raw PLAsTiCC target integers
    """
    classes   = sorted(meta["target"].unique().tolist())
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    return label_map, classes


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PLAsTiCCDataset(Dataset):
    """
    PyTorch Dataset for PLAsTiCC transient classification.

    Yields ``(sequence, mask, label)`` triplets::

        sequence : Tensor[MAX_SEQ_LEN, INPUT_DIM]  float32
        mask     : Tensor[MAX_SEQ_LEN]              bool   (True = padded)
        label    : Tensor[]                         int64
    """

    def __init__(
        self,
        sequences : Dict[int, np.ndarray],
        labels    : Dict[int, int],
        max_len   : int = config.MAX_SEQ_LEN,
    ) -> None:
        self.object_ids = list(sequences.keys())
        self.sequences  = sequences
        self.labels     = labels
        self.max_len    = max_len

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obj_id  = self.object_ids[idx]
        seq     = self.sequences[obj_id]
        label   = self.labels[obj_id]
        padded, mask = pad_sequence(seq, self.max_len)

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_dataloaders(
    fraction   : float = 1.0,
    batch_size : int   = config.BATCH_SIZE,
    seed       : int   = config.SEED,
) -> Tuple[DataLoader, DataLoader, Dict[int, int], List[int]]:
    """
    Full preprocessing pipeline → PyTorch DataLoaders.

    Parameters
    ----------
    fraction   : Fraction of each object's timeline to keep (early truncation).
    batch_size : Mini-batch size for both loaders.
    seed       : Random seed for the stratified train/test split.

    Returns
    -------
    train_loader : DataLoader  (shuffled)
    test_loader  : DataLoader  (ordered)
    label_map    : dict  raw_target → class_idx
    classes      : list  sorted raw PLAsTiCC target integers
    """
    lc, meta = load_raw_data()

    print("[data_loader] Handling missing values …")
    lc = handle_missing(lc)

    print("[data_loader] Normalising flux per object …")
    lc = normalize_flux(lc)

    print(f"[data_loader] Truncating to {fraction * 100:.0f}% of observations …")
    lc = truncate_observations(lc, fraction)

    print("[data_loader] Building per-object sequences …")
    sequences = build_sequences(lc)

    label_map, classes = build_label_map(meta)

    # Keep only objects present in both light-curve and metadata tables
    labels: Dict[int, int] = {
        int(row["object_id"]): label_map[row["target"]]
        for _, row in meta.iterrows()
        if int(row["object_id"]) in sequences and row["target"] in label_map
    }
    sequences = {oid: sequences[oid] for oid in labels}

    # Object-level stratified split — guarantees no leakage
    object_ids   = list(labels.keys())
    strat_labels = [labels[o] for o in object_ids]
    train_ids, test_ids = train_test_split(
        object_ids,
        test_size   = config.TEST_SIZE,
        random_state = seed,
        stratify    = strat_labels,
    )
    print(f"[data_loader] Split: {len(train_ids):,} train / {len(test_ids):,} test objects")

    train_dataset = PLAsTiCCDataset(
        {o: sequences[o] for o in train_ids},
        {o: labels[o]    for o in train_ids},
    )
    test_dataset = PLAsTiCCDataset(
        {o: sequences[o] for o in test_ids},
        {o: labels[o]    for o in test_ids},
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle    = True,
        num_workers = 0,
        pin_memory  = pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle    = False,
        num_workers = 0,
        pin_memory  = pin,
    )

    print(
        f"[data_loader] Ready. "
        f"{len(train_dataset):,} train samples, "
        f"{len(test_dataset):,} test samples, "
        f"{len(classes)} classes."
    )
    return train_loader, test_loader, label_map, classes
