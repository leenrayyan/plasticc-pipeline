# Running the Pipeline in Google Colab (T4 GPU)

Follow these 5 steps to go from zero to results in Colab.

---

## Step 1 — Open a new Colab notebook and enable GPU

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a **New notebook**.
2. Click **Runtime → Change runtime type**.
3. Set **Hardware accelerator** to **T4 GPU** and click **Save**.

---

## Step 2 — Mount Google Drive and clone the repository

Paste the following into the first cell and run it:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/plasticc_pipeline.git /content/plasticc_pipeline
%cd /content/plasticc_pipeline
```

> Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Step 3 — Upload your Kaggle API token (`kaggle.json`)

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New API Token**.
   This downloads a file called `kaggle.json`.
2. In Colab, click the **folder icon** in the left sidebar, then **Upload**.
3. Upload `kaggle.json` to `/content/` (the default Colab directory).

You can verify it arrived:
```python
import os
print(os.path.exists('/content/kaggle.json'))   # should print True
```

---

## Step 4 — Run `colab_setup.py`

This single script handles everything: installing dependencies, configuring
Kaggle, downloading the PLAsTiCC dataset, and patching paths.

```python
# Edit GITHUB_REPO_URL inside colab_setup.py first, then:
!python colab_setup.py
```

Expected output (abbreviated):
```
[setup] Step 1 — Mounting Google Drive …  ✓
[setup] Step 2 — Cloning repo …           ✓
[setup] Step 3 — Installing dependencies … ✓
[setup] Step 4 — Configuring Kaggle API … ✓
[setup] Step 5 — Downloading PLAsTiCC … ✓
[setup] Step 6 — Patching config paths … ✓
Setup complete!
```

> The download of `training_set.csv` (~500 MB) may take a few minutes.

---

## Step 5 — Run the pipeline

```python
# Full pipeline — all 4 models, all 4 truncation fractions (~2–4 hours on T4)
!python main.py

# Quick smoke test — one model, one fraction, fewer MC samples
!python main.py --truncation 0.3 --models transformer xgboost --mc-samples 10
```

Results are saved to `/content/plasticc_pipeline/results/`.

To view figures inline:
```python
from IPython.display import Image
Image('/content/plasticc_pipeline/results/figures/topk_recall_curves.png')
```

To download all results to your Drive:
```python
import shutil
shutil.copytree(
    '/content/plasticc_pipeline/results',
    '/content/drive/MyDrive/plasticc_results',
    dirs_exist_ok=True,
)
print("Results copied to Google Drive.")
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: kaggle.json` | Re-upload `kaggle.json` to `/content/` and re-run `colab_setup.py` |
| `404 – competition not found` | Accept the PLAsTiCC competition rules at kaggle.com/c/PLAsTiCC-2018 |
| `CUDA out of memory` | Reduce `--batch-size 32` or lower `MAX_SEQ_LEN` in `config.py` |
| `ImportError: astromer` | Run `!pip install astromer` manually, then re-run `main.py` |
| `ImportError: uni2ts` (Moirai) | Moirai falls back to Chronos automatically — no action needed |
| Runtime disconnects mid-training | Checkpoints are saved each epoch; re-run `main.py` to resume |
