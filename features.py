"""
features.py — Physics-informed light-curve features for the XGBoost baseline.

Features are extracted from the *raw* (un-normalised) flux so that absolute
scale information is preserved for the tree-based model.

Feature vector (per object, ~50 values)
-----------------------------------------
Global statistics (9):
    mean_flux, std_flux, max_flux, min_flux, flux_range,
    slope, skewness, n_detections, median_flux_err

Per-passband mean flux (6):
    mean_flux_pb0 … mean_flux_pb5

Per-passband peak flux (6):
    peak_flux_pb0 … peak_flux_pb5

Color indices at peak — flux ratios between bands (5):
    color_g_r, color_r_i, color_i_z, color_g_i, color_r_z
    (These are the most discriminating features for rare classes:
     kilonovae are red, SNIa are blue at peak)

Temporal features (9):
    time_to_peak       — MJD from first detection to peak flux
    rise_time          — days from first detected to peak (detected obs only)
    fade_time          — days from peak to half-peak flux (after peak)
    peak_mjd           — absolute MJD of peak (normalized by survey start)
    duration           — total observed duration (last - first MJD)
    n_obs_before_peak  — number of observations before peak
    n_obs_after_peak   — number of observations after peak
    peak_asymmetry     — (fade_time - rise_time) / (fade_time + rise_time + 1)
    flux_at_start      — mean flux in first 10% of observations

Metadata features (4):
    hostgal_photoz     — photometric redshift of host galaxy
    hostgal_photoz_err — uncertainty on photometric redshift
    distmod            — distance modulus
    mwebv              — Milky Way dust extinction

Per-passband std flux (6):
    std_flux_pb0 … std_flux_pb5
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress, skew

import config


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

_GLOBAL_FEATURES = [
    "mean_flux", "std_flux", "max_flux", "min_flux", "flux_range",
    "slope", "skewness", "n_detections", "median_flux_err",
]
_PB_MEAN_FEATURES  = [f"mean_flux_pb{i}" for i in range(config.N_PASSBANDS)]
_PB_PEAK_FEATURES  = [f"peak_flux_pb{i}" for i in range(config.N_PASSBANDS)]
_PB_STD_FEATURES   = [f"std_flux_pb{i}"  for i in range(config.N_PASSBANDS)]
_COLOR_FEATURES    = ["color_g_r", "color_r_i", "color_i_z", "color_g_i", "color_r_z"]
_TEMPORAL_FEATURES = [
    "time_to_peak", "rise_time", "fade_time", "peak_mjd", "duration",
    "n_obs_before_peak", "n_obs_after_peak", "peak_asymmetry", "flux_at_start",
]
_META_FEATURES = ["hostgal_photoz", "hostgal_photoz_err", "distmod", "mwebv"]

FEATURE_NAMES: List[str] = (
    _GLOBAL_FEATURES
    + _PB_MEAN_FEATURES
    + _PB_PEAK_FEATURES
    + _PB_STD_FEATURES
    + _COLOR_FEATURES
    + _TEMPORAL_FEATURES
    + _META_FEATURES
)


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names matching ``extract_features``."""
    return FEATURE_NAMES.copy()


# ---------------------------------------------------------------------------
# Per-object feature extraction
# ---------------------------------------------------------------------------

def extract_features(grp: pd.DataFrame, meta_row: pd.Series = None) -> np.ndarray:
    """
    Extract physics-informed features for a single light curve.

    Parameters
    ----------
    grp      : Rows for one object. Expected columns:
               [mjd, flux, flux_err, passband, detected / detected_bool].
    meta_row : Optional Series with metadata columns
               [hostgal_photoz, hostgal_photoz_err, distmod, mwebv].

    Returns
    -------
    np.ndarray of shape (len(FEATURE_NAMES),) dtype float32.
    """
    grp      = grp.sort_values("mjd").reset_index(drop=True)
    flux     = grp["flux"].values.astype(float)
    flux_err = grp["flux_err"].values.astype(float)
    mjd      = grp["mjd"].values.astype(float)
    passband = grp["passband"].values.astype(int)

    detected_col = "detected_bool" if "detected_bool" in grp.columns else "detected"
    detected = grp[detected_col].values.astype(int)

    # ---- Global statistics -------------------------------------------------
    mean_flux   = float(np.mean(flux))
    std_flux    = float(np.std(flux))  if len(flux) > 1 else 0.0
    max_flux    = float(np.max(flux))
    min_flux    = float(np.min(flux))
    flux_range  = max_flux - min_flux

    if len(mjd) >= 2 and np.ptp(mjd) > 0:
        slope, *_ = linregress(mjd, flux)
        slope = float(slope)
    else:
        slope = 0.0

    skewness = float(skew(flux)) if len(flux) >= 3 else 0.0
    if np.isnan(skewness):
        skewness = 0.0

    n_detections    = int(detected.sum())
    median_flux_err = float(np.median(flux_err))

    # ---- Per-passband features ---------------------------------------------
    pb_means = []
    pb_peaks = []
    pb_stds  = []
    for pb in range(config.N_PASSBANDS):
        mask = passband == pb
        if mask.any():
            pb_flux = flux[mask]
            pb_means.append(float(np.mean(pb_flux)))
            pb_peaks.append(float(np.max(pb_flux)))
            pb_stds.append(float(np.std(pb_flux)) if len(pb_flux) > 1 else 0.0)
        else:
            pb_means.append(0.0)
            pb_peaks.append(0.0)
            pb_stds.append(0.0)

    # ---- Color indices (flux ratios between bands at peak) -----------------
    # Bands: u=0, g=1, r=2, i=3, z=4, Y=5
    # Ratio = peak_flux_band_A / (peak_flux_band_B + 1e-10) to avoid /0
    # High r/g ratio = redder object (kilonova signature)
    def safe_ratio(a, b):
        return float(a / (b + 1e-10)) if (a != 0 or b != 0) else 0.0

    color_g_r = safe_ratio(pb_peaks[1], pb_peaks[2])  # g/r
    color_r_i = safe_ratio(pb_peaks[2], pb_peaks[3])  # r/i
    color_i_z = safe_ratio(pb_peaks[3], pb_peaks[4])  # i/z
    color_g_i = safe_ratio(pb_peaks[1], pb_peaks[3])  # g/i
    color_r_z = safe_ratio(pb_peaks[2], pb_peaks[4])  # r/z

    # ---- Temporal features -------------------------------------------------
    peak_idx = int(np.argmax(flux))
    peak_mjd_val = float(mjd[peak_idx])
    survey_start = float(mjd[0])
    duration = float(mjd[-1] - mjd[0]) if len(mjd) > 1 else 0.0

    # Time from first observation to peak
    time_to_peak = peak_mjd_val - survey_start

    # Rise time: first detected observation to peak
    det_indices = np.where(detected == 1)[0]
    if len(det_indices) > 0:
        first_det_mjd = float(mjd[det_indices[0]])
        rise_time = max(0.0, peak_mjd_val - first_det_mjd)
    else:
        rise_time = time_to_peak

    # Fade time: peak to half-peak flux (after peak)
    half_peak = max_flux / 2.0
    after_peak = np.where((mjd > peak_mjd_val) & (flux <= half_peak))[0]
    if len(after_peak) > 0:
        fade_time = float(mjd[after_peak[0]] - peak_mjd_val)
    else:
        # If never fades to half-peak, use time from peak to end
        fade_time = float(mjd[-1] - peak_mjd_val) if mjd[-1] > peak_mjd_val else 0.0

    # Observations before/after peak
    n_obs_before_peak = int(np.sum(mjd < peak_mjd_val))
    n_obs_after_peak  = int(np.sum(mjd > peak_mjd_val))

    # Peak asymmetry: negative = fast rise slow fade (SNIa), positive = slow rise fast fade
    denom = fade_time + rise_time + 1.0
    peak_asymmetry = float((fade_time - rise_time) / denom)

    # Mean flux in first 10% of observations (pre-explosion baseline)
    n_early = max(1, int(len(flux) * 0.1))
    flux_at_start = float(np.mean(flux[:n_early]))

    # Normalized peak MJD (relative to survey start)
    peak_mjd_norm = time_to_peak

    # ---- Metadata features -------------------------------------------------
    if meta_row is not None:
        hostgal_photoz     = float(meta_row.get("hostgal_photoz", 0.0)     or 0.0)
        hostgal_photoz_err = float(meta_row.get("hostgal_photoz_err", 0.0) or 0.0)
        distmod            = float(meta_row.get("distmod", 0.0)            or 0.0)
        mwebv              = float(meta_row.get("mwebv", 0.0)              or 0.0)
    else:
        hostgal_photoz     = 0.0
        hostgal_photoz_err = 0.0
        distmod            = 0.0
        mwebv              = 0.0

    # ---- Assemble feature vector -------------------------------------------
    features = np.array(
        [
            # Global (9)
            mean_flux, std_flux, max_flux, min_flux, flux_range,
            slope, skewness, n_detections, median_flux_err,
        ]
        + pb_means   # per-band mean (6)
        + pb_peaks   # per-band peak (6)
        + pb_stds    # per-band std  (6)
        + [
            # Color indices (5)
            color_g_r, color_r_i, color_i_z, color_g_i, color_r_z,
            # Temporal (9)
            time_to_peak, rise_time, fade_time, peak_mjd_norm, duration,
            float(n_obs_before_peak), float(n_obs_after_peak),
            peak_asymmetry, flux_at_start,
            # Metadata (4)
            hostgal_photoz, hostgal_photoz_err, distmod, mwebv,
        ],
        dtype=np.float32,
    )

    # Replace any NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
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
    lc        : Raw light-curve DataFrame (may be truncated).
    meta      : Metadata DataFrame with columns [object_id, target, …].
    label_map : Mapping raw PLAsTiCC target integer → contiguous class index.

    Returns
    -------
    X          : np.ndarray shape (n_objects, n_features) float32
    y          : np.ndarray shape (n_objects,)            int64
    object_ids : np.ndarray shape (n_objects,)            int64
    """
    meta_idx = meta.set_index("object_id")

    rows_X : List[np.ndarray] = []
    rows_y : List[int]        = []
    obj_ids: List[int]        = []

    for obj_id, grp in lc.groupby("object_id"):
        if obj_id not in meta_idx.index:
            continue
        target = meta_idx.loc[obj_id, "target"]
        if target not in label_map:
            continue

        # Pass metadata row to extract_features for redshift/distance features
        meta_row = meta_idx.loc[obj_id]
        feat = extract_features(grp, meta_row=meta_row)

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
