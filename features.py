"""
features.py — Hand-crafted light-curve features for the XGBoost baseline.

Features are extracted from the *raw* (un-normalised) flux so that absolute
scale information is preserved for the tree-based model.  Deep-learning
models use the normalised sequences produced by data_loader.py instead.

Feature vector (per object, 15 values)
---------------------------------------
Global:
    mean_flux, std_flux, max_flux, min_flux, flux_range,
    slope (linear regression mjd→flux), skewness,
    n_detections, median_flux_err

Per-passband mean flux (6 bands: u, g, r, i, z, Y):
    mean_flux_pb0 … mean_flux_pb5
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress, skew

import config


# ---------------------------------------------------------------------------
# Feature names (keep in sync with extract_features)
# ---------------------------------------------------------------------------

_GLOBAL_FEATURES = [
    "mean_flux",
    "std_flux",
    "max_flux",
    "min_flux",
    "flux_range",
    "slope",
    "skewness",
    "n_detections",
    "median_flux_err",
]
_PB_FEATURES = [f"mean_flux_pb{i}" for i in range(config.N_PASSBANDS)]
FEATURE_NAMES: List[str] = _GLOBAL_FEATURES + _PB_FEATURES


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names matching ``extract_features``."""
    return FEATURE_NAMES.copy()


# ---------------------------------------------------------------------------
# Per-object feature extraction
# ---------------------------------------------------------------------------

def extract_features(grp: pd.DataFrame) -> np.ndarray:
    """
    Extract hand-crafted statistical features for a single light curve.

    Parameters
    ----------
    grp : pd.DataFrame
        Rows belonging to one object. Expected columns:
        ``[mjd, flux, flux_err, passband, detected]``.

    Returns
    -------
    np.ndarray of shape ``(len(FEATURE_NAMES),)`` with dtype float32.
    """
    grp      = grp.sort_values("mjd").reset_index(drop=True)
    flux     = grp["flux"].values.astype(float)
    flux_err = grp["flux_err"].values.astype(float)
    mjd      = grp["mjd"].values.astype(float)
    detected = grp["detected"].values.astype(int)

    # ---- Global statistics -------------------------------------------------
    mean_flux  = float(np.mean(flux))
    std_flux   = float(np.std(flux))   if len(flux) > 1 else 0.0
    max_flux   = float(np.max(flux))
    min_flux   = float(np.min(flux))
    flux_range = max_flux - min_flux

    # Linear slope: regress flux against MJD
    if len(mjd) >= 2 and np.ptp(mjd) > 0:
        slope, *_ = linregress(mjd, flux)
        slope = float(slope)
    else:
        slope = 0.0

    # Skewness (requires ≥3 observations; NaN → 0)
    if len(flux) >= 3:
        skewness = float(skew(flux))
        if np.isnan(skewness):
            skewness = 0.0
    else:
        skewness = 0.0

    n_detections   = int(detected.sum())
    median_flux_err = float(np.median(flux_err))

    # ---- Per-passband mean flux --------------------------------------------
    pb_means: List[float] = []
    for pb in range(config.N_PASSBANDS):
        mask = grp["passband"].values == pb
        pb_means.append(float(np.mean(flux[mask])) if mask.any() else 0.0)

    features = np.array(
        [
            mean_flux, std_flux, max_flux, min_flux, flux_range,
            slope, skewness, n_detections, median_flux_err,
        ] + pb_means,
        dtype=np.float32,
    )
    return features


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    lc        : pd.DataFrame,
    meta      : pd.DataFrame,
    label_map : Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (X, y, object_ids) arrays for all objects present in *lc*.

    Parameters
    ----------
    lc        : Raw (un-normalised) light-curve DataFrame.  May be truncated.
    meta      : Metadata DataFrame with columns ``[object_id, target, …]``.
    label_map : Mapping raw PLAsTiCC target integer → contiguous class index.

    Returns
    -------
    X          : np.ndarray  shape ``(n_objects, n_features)``  float32
    y          : np.ndarray  shape ``(n_objects,)``             int64
    object_ids : np.ndarray  shape ``(n_objects,)``             int64
    """
    meta_idx = meta.set_index("object_id")

    rows_X: List[np.ndarray] = []
    rows_y: List[int]        = []
    obj_ids: List[int]       = []

    for obj_id, grp in lc.groupby("object_id"):
        if obj_id not in meta_idx.index:
            continue
        target = meta_idx.loc[obj_id, "target"]
        if target not in label_map:
            continue

        feat = extract_features(grp)
        rows_X.append(feat)
        rows_y.append(label_map[target])
        obj_ids.append(int(obj_id))

    if not rows_X:
        raise ValueError(
            "build_feature_matrix produced an empty matrix. "
            "Check that lc and meta share object_ids and that label_map is correct."
        )

    X          = np.vstack(rows_X).astype(np.float32)
    y          = np.array(rows_y, dtype=np.int64)
    object_ids = np.array(obj_ids, dtype=np.int64)

    print(
        f"[features] Feature matrix: {X.shape[0]:,} objects × "
        f"{X.shape[1]} features, {len(np.unique(y))} classes."
    )
    return X, y, object_ids
