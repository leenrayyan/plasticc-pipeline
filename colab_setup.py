"""
colab_setup.py — One-shot environment bootstrap for Google Colab (T4 GPU).

Run this script in a Colab cell BEFORE running main.py:

    !python colab_setup.py

What it does
------------
1.  Mounts Google Drive at /content/drive.
2.  Clones the project GitHub repository into /content/plasticc_pipeline.
3.  Installs all dependencies from requirements.txt.
4.  Locates kaggle.json (uploaded by the user) and configures Kaggle CLI.
5.  Downloads the PLAsTiCC dataset from Kaggle into the data/ directory.
6.  Patches config.py paths so they point to the correct Colab directories.

Prerequisites
-------------
- Upload kaggle.json to /content/ (or Google Drive root) before running.
- Set GITHUB_REPO_URL below to your repository's HTTPS clone URL.
"""

import os
import sys
import subprocess
import shutil

# ===========================================================================
# CONFIGURE THESE BEFORE RUNNING
# ===========================================================================
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/plasticc_pipeline.git"
REPO_NAME       = "plasticc_pipeline"
# ===========================================================================

COLAB_CONTENT   = "/content"
REPO_DIR        = os.path.join(COLAB_CONTENT, REPO_NAME)
DATA_DIR        = os.path.join(REPO_DIR, "data")
DRIVE_ROOT      = "/content/drive/MyDrive"
KAGGLE_JSON_SRC = os.path.join(COLAB_CONTENT, "kaggle.json")
KAGGLE_DIR      = os.path.expanduser("~/.kaggle")
KAGGLE_JSON_DST = os.path.join(KAGGLE_DIR, "kaggle.json")


def _run(cmd: str, check: bool = True) -> int:
    """Run a shell command and print it; raise on non-zero exit if check=True."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result.returncode


def _is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Step 1 — Mount Google Drive
# ---------------------------------------------------------------------------

def mount_drive() -> None:
    """Mount Google Drive to /content/drive."""
    print("\n[setup] Step 1 — Mounting Google Drive …")
    if not _is_colab():
        print("  Not running in Colab — skipping Drive mount.")
        return
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("  Drive mounted at /content/drive")


# ---------------------------------------------------------------------------
# Step 2 — Clone repository
# ---------------------------------------------------------------------------

def clone_repo() -> None:
    """Clone the GitHub repository into /content/."""
    print(f"\n[setup] Step 2 — Cloning {GITHUB_REPO_URL} …")
    if os.path.isdir(REPO_DIR):
        print(f"  Repository already exists at {REPO_DIR} — pulling latest …")
        _run(f"git -C {REPO_DIR} pull --ff-only", check=False)
    else:
        _run(f"git clone {GITHUB_REPO_URL} {REPO_DIR}")
    # Add repo to sys.path so modules can be imported directly
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    os.chdir(REPO_DIR)
    print(f"  Working directory set to: {REPO_DIR}")


# ---------------------------------------------------------------------------
# Step 3 — Install dependencies
# ---------------------------------------------------------------------------

def install_dependencies() -> None:
    """Install Python packages from requirements.txt."""
    print("\n[setup] Step 3 — Installing dependencies …")
    req_path = os.path.join(REPO_DIR, "requirements.txt")
    if not os.path.exists(req_path):
        raise FileNotFoundError(f"requirements.txt not found at {req_path}")
    _run(f"pip install -q -r {req_path}")
    print("  Dependencies installed.")


# ---------------------------------------------------------------------------
# Step 4 — Configure Kaggle API
# ---------------------------------------------------------------------------

def setup_kaggle() -> None:
    """Locate kaggle.json and configure the Kaggle CLI."""
    print("\n[setup] Step 4 — Configuring Kaggle API …")

    # Search for kaggle.json in common locations
    search_paths = [
        KAGGLE_JSON_SRC,
        os.path.join(DRIVE_ROOT, "kaggle.json"),
        os.path.join(DRIVE_ROOT, "Colab Notebooks", "kaggle.json"),
    ]
    found = None
    for p in search_paths:
        if os.path.exists(p):
            found = p
            break

    if found is None:
        raise FileNotFoundError(
            "kaggle.json not found.  Please upload it to /content/ "
            "(or place it in the root of your Google Drive) and re-run this script.\n"
            "Get your API token from: https://www.kaggle.com/settings → API → "
            "'Create New Token'."
        )

    os.makedirs(KAGGLE_DIR, exist_ok=True)
    shutil.copy2(found, KAGGLE_JSON_DST)
    os.chmod(KAGGLE_JSON_DST, 0o600)
    print(f"  kaggle.json configured from {found}")


# ---------------------------------------------------------------------------
# Step 5 — Download PLAsTiCC dataset
# ---------------------------------------------------------------------------

def download_plasticc() -> None:
    """Download the PLAsTiCC training files from Kaggle."""
    print("\n[setup] Step 5 — Downloading PLAsTiCC dataset …")
    os.makedirs(DATA_DIR, exist_ok=True)

    target_files = ["training_set.csv", "training_set_metadata.csv"]
    missing      = [f for f in target_files if not os.path.exists(os.path.join(DATA_DIR, f))]

    if not missing:
        print(f"  Dataset already present in {DATA_DIR} — skipping download.")
        return

    # Download and unzip the competition files
    _run(
        f"kaggle competitions download -c PLAsTiCC-2018 "
        f"--path {DATA_DIR} -f training_set.csv",
        check=False,
    )
    _run(
        f"kaggle competitions download -c PLAsTiCC-2018 "
        f"--path {DATA_DIR} -f training_set_metadata.csv",
        check=False,
    )

    # Unzip any .zip archives that were downloaded
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".zip"):
            _run(f"unzip -o {os.path.join(DATA_DIR, fname)} -d {DATA_DIR}", check=False)
            os.remove(os.path.join(DATA_DIR, fname))

    # Verify
    for f in target_files:
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file not found after download: {path}\n"
                "Check that you have accepted the PLAsTiCC competition rules on Kaggle."
            )
    print(f"  PLAsTiCC dataset ready in {DATA_DIR}")


# ---------------------------------------------------------------------------
# Step 6 — Patch config.py paths
# ---------------------------------------------------------------------------

def patch_config() -> None:
    """
    Dynamically update config module so all paths point to Colab directories.

    This is non-destructive (modifies the in-memory module only; the file on
    disk is unchanged so the repo stays clean).
    """
    print("\n[setup] Step 6 — Patching config paths for Colab …")
    import importlib
    # Ensure the repo is on the path before importing
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    import config as cfg

    cfg.BASE_DIR        = REPO_DIR
    cfg.DATA_DIR        = DATA_DIR
    cfg.RESULTS_DIR     = os.path.join(REPO_DIR, "results")
    cfg.FIG_DIR         = os.path.join(cfg.RESULTS_DIR, "figures")
    cfg.CHECKPOINT_DIR  = os.path.join(REPO_DIR, "checkpoints")
    cfg.TRAINING_SET_PATH = os.path.join(DATA_DIR, "training_set.csv")
    cfg.METADATA_PATH     = os.path.join(DATA_DIR, "training_set_metadata.csv")

    for d in [cfg.RESULTS_DIR, cfg.FIG_DIR, cfg.CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"  config.DATA_DIR       = {cfg.DATA_DIR}")
    print(f"  config.RESULTS_DIR    = {cfg.RESULTS_DIR}")
    print(f"  config.CHECKPOINT_DIR = {cfg.CHECKPOINT_DIR}")
    print("  Config patched successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all setup steps in order."""
    print("=" * 60)
    print("  PLAsTiCC Pipeline — Colab Environment Setup")
    print("=" * 60)

    if GITHUB_REPO_URL == "https://github.com/YOUR_USERNAME/plasticc_pipeline.git":
        raise ValueError(
            "Please set GITHUB_REPO_URL at the top of colab_setup.py "
            "to your actual repository URL before running."
        )

    mount_drive()
    clone_repo()
    install_dependencies()
    setup_kaggle()
    download_plasticc()
    patch_config()

    print("\n" + "=" * 60)
    print("  Setup complete!  You can now run:")
    print("    python main.py")
    print("  or with a specific truncation fraction:")
    print("    python main.py --truncation 0.3")
    print("=" * 60)


if __name__ == "__main__":
    main()
