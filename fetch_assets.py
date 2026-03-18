"""
fetch_assets.py — Download all external assets needed by the pipeline.

Run once before main.py:
    python fetch_assets.py

What it fetches
---------------
1. PLAsTiCC training data from Kaggle  (uses ~/.kaggle/kaggle.json automatically)
2. Astromer weights                    (pip install astromer)
3. Transformers / HuggingFace models:
   - amazon/chronos-t5-small           (Moirai fallback)
   - Salesforce/moirai-1.0-R-small     (attempted; skipped if unavailable)
"""

import argparse
import os
import shutil
import subprocess
import sys

# Windows: use UTF-8 output so symbols don't crash cp1252 terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, check: bool = True) -> int:
    """Run a shell command using the same Python interpreter that launched this script."""
    # Always use the current interpreter's pip to avoid wrong-environment installs
    cmd = cmd.replace("pip install", f'"{sys.executable}" -m pip install', 1)
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")
    return r.returncode


# ---------------------------------------------------------------------------
# 1. Kaggle dataset
# ---------------------------------------------------------------------------

def _find_kaggle_json(explicit_path=None) -> str:
    """Return path to a usable kaggle.json, or raise FileNotFoundError."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates += [
        os.path.expanduser("~/.kaggle/kaggle.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "kaggle.json not found.\n"
        "Run:  python make_kaggle_json.py\n"
        "to create it from your Kaggle username + API token."
    )


def download_plasticc(data_dir: str, kaggle_json: str = None) -> None:
    """Download training_set.csv and training_set_metadata.csv from Kaggle."""
    print("\n[fetch] Downloading PLAsTiCC dataset from Kaggle ...")

    src = _find_kaggle_json(kaggle_json)
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    dst = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
        os.chmod(dst, 0o600)
    print(f"  Using credentials: {dst}")

    os.makedirs(data_dir, exist_ok=True)

    needed  = ["training_set.csv", "training_set_metadata.csv"]
    missing = [f for f in needed if not os.path.exists(os.path.join(data_dir, f))]

    if not missing:
        print("  Dataset already present -- skipping download.")
        return

    for fname in missing:
        print(f"  Downloading {fname} ...")
        rc = subprocess.run(
            f'kaggle competitions download -c PLAsTiCC-2018 --path "{data_dir}" -f {fname}',
            shell=True,
        ).returncode

        if rc != 0:
            print(
                f"\n  ERROR: Could not download {fname}.\n"
                "  If you see a 403 error, you need to accept the competition rules:\n"
                "  -> https://www.kaggle.com/c/PLAsTiCC-2018/rules\n"
                "     Click 'I Understand and Accept', then re-run this script.\n"
            )
            continue

        # Unzip if Kaggle returned a .zip
        zip_path = os.path.join(data_dir, fname + ".zip")
        if os.path.exists(zip_path):
            print(f"  Unzipping {zip_path} ...")
            subprocess.run(
                f'"{sys.executable}" -m zipfile -e "{zip_path}" "{data_dir}"',
                shell=True,
            )
            os.remove(zip_path)

    # Verify
    for f in needed:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  OK  {f}  ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {f} -- download may have failed")


# ---------------------------------------------------------------------------
# 2. Astromer
# ---------------------------------------------------------------------------

def install_astromer() -> None:
    """Install the astromer package and verify the import."""
    print("\n[fetch] Installing Astromer ...")
    _run("pip install -q astromer", check=False)

    # Reload so the freshly installed package is visible
    import importlib, sys as _sys
    for mod in list(_sys.modules.keys()):
        if "astromer" in mod:
            del _sys.modules[mod]

    try:
        from astromer import SingleBandEncoder  # noqa: F401
        print("  OK  Astromer installed.")
    except Exception as e:
        print(f"  WARNING: Astromer post-install check failed: {e}")
        print("    This is non-fatal -- the pipeline will still work.")


# ---------------------------------------------------------------------------
# 3. HuggingFace models
# ---------------------------------------------------------------------------

def cache_hf_model(model_id: str, model_type: str = "auto") -> None:
    """Pre-download a HuggingFace model to the local cache."""
    print(f"\n[fetch] Caching HuggingFace model: {model_id} ...")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_id)
        if model_type == "t5encoder":
            from transformers import T5EncoderModel
            T5EncoderModel.from_pretrained(model_id)
        print(f"  OK  {model_id} cached.")
    except Exception as e:
        print(f"  WARNING: Could not cache {model_id}: {e}")
        print("    The pipeline will attempt to download it at runtime.")


def cache_moirai() -> None:
    """Attempt to cache Moirai-small via uni2ts; skip gracefully if unavailable."""
    print("\n[fetch] Attempting to cache Moirai-small ...")
    try:
        from uni2ts.model.moirai import MoiraiModule
        MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")
        print("  OK  Moirai-small cached.")
    except ImportError:
        print(
            "  INFO: uni2ts not installed -- Moirai will be skipped.\n"
            "        Chronos will be used as Model 4 fallback (already cached).\n"
            "        To install Moirai:\n"
            "          pip install git+https://github.com/SalesforceAIResearch/uni2ts.git"
        )
    except Exception as e:
        print(f"  WARNING: Moirai cache failed: {e} -- Chronos fallback will be used.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Fetch all external assets for the PLAsTiCC pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kaggle-json",
        default=None,
        metavar="PATH",
        help="Path to kaggle.json. Auto-detected from ~/.kaggle/ if omitted.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(here, "data"),
        metavar="DIR",
        help="Directory where PLAsTiCC CSVs will be saved.",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle download.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip HuggingFace / Astromer model caching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  PLAsTiCC Pipeline -- Asset Fetcher")
    print("=" * 60)

    # ---- PLAsTiCC data -----------------------------------------------------
    if not args.skip_kaggle:
        try:
            download_plasticc(args.data_dir, args.kaggle_json)
        except FileNotFoundError as e:
            print(f"\n[fetch] {e}")

    # ---- Models ------------------------------------------------------------
    if not args.skip_models:
        install_astromer()
        _run("pip install -q transformers>=4.40 huggingface_hub", check=False)
        cache_hf_model("amazon/chronos-t5-small", model_type="t5encoder")
        cache_moirai()

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Asset fetch complete.")
    print(f"  Data dir : {args.data_dir}")

    csvs = ["training_set.csv", "training_set_metadata.csv"]
    for f in csvs:
        path   = os.path.join(args.data_dir, f)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  {status:<8} {f}")

    print("\n  When both CSVs are present, run:")
    print("    python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
