"""
make_kaggle_json.py — Interactively create ~/.kaggle/kaggle.json from a token.

Run in your terminal:
    python make_kaggle_json.py
"""

import getpass
import json
import os
import stat

KAGGLE_DIR  = os.path.expanduser("~/.kaggle")
KAGGLE_JSON = os.path.join(KAGGLE_DIR, "kaggle.json")

print("=" * 50)
print("  Kaggle API token setup")
print("=" * 50)
print("(Nothing you type here is sent anywhere — it is only saved locally.)\n")

username = input("Enter your Kaggle username: ").strip()
token    = getpass.getpass("Enter your Kaggle API token (hidden): ").strip()

if not username or not token:
    print("ERROR: username and token cannot be empty.")
    raise SystemExit(1)

os.makedirs(KAGGLE_DIR, exist_ok=True)
with open(KAGGLE_JSON, "w") as f:
    json.dump({"username": username, "key": token}, f)

# Restrict permissions (required by Kaggle CLI)
os.chmod(KAGGLE_JSON, stat.S_IRUSR | stat.S_IWUSR)

print(f"\nSaved to {KAGGLE_JSON}")
print("Now run:  python fetch_assets.py")
