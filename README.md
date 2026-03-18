# Uncertainty-Aware Early Transient Classification Framework

A complete ML pipeline for classifying astronomical transients from the
[PLAsTiCC](https://www.kaggle.com/c/PLAsTiCC-2018) dataset, featuring:

- **Four classifiers**: XGBoost baseline, scratch Transformer, Astromer (pretrained), Moirai/Chronos (foundation model).
- **Early truncation**: simulate real-world early-alert classification at 10%, 30%, 50%, and 100% of each object's light curve.
- **Uncertainty quantification**: MC Dropout with 50 stochastic forward passes.
- **Calibration evaluation**: reliability diagrams + Expected Calibration Error (ECE).
- **Prioritization simulation**: Top-K rare-class (kilonova, TDE) recall curves as the primary paper figure.

---

## Repository structure

```
plasticc_pipeline/
├── config.py            All paths and hyperparameters (single source of truth)
├── data_loader.py       Data loading, normalisation, truncation, DataLoaders
├── features.py          Hand-crafted features for XGBoost
├── models.py            All four model definitions
├── uncertainty.py       MC Dropout inference (model-agnostic)
├── calibration.py       Reliability diagrams and ECE
├── prioritization.py    Follow-up budget / Top-K recall simulation
├── train.py             Training loop (PyTorch models)
├── evaluate.py          Full evaluation across models × truncation fractions
├── main.py              Pipeline orchestrator with CLI
├── colab_setup.py       One-shot Colab environment bootstrap
├── requirements.txt
├── README.md
└── RUN_IN_COLAB.md
```

---

## Quick start (local)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/plasticc_pipeline.git
cd plasticc_pipeline
pip install -r requirements.txt
```

### 2. Download data

Download the PLAsTiCC dataset from Kaggle:
```bash
kaggle competitions download -c PLAsTiCC-2018
unzip PLAsTiCC-2018.zip -d data/
```

Required files:
- `data/training_set.csv`
- `data/training_set_metadata.csv`

### 3. Run the full pipeline

```bash
# All four models, all four truncation fractions
python main.py

# Single fraction (faster, for debugging)
python main.py --truncation 0.3

# Specific models only
python main.py --models transformer xgboost

# Fewer MC samples for quick testing
python main.py --mc-samples 10 --truncation 0.5
```

### 4. Find results

| Path | Contents |
|------|----------|
| `results/results_table.csv` | Metrics for all models × fractions |
| `results/figures/reliability_*.png` | Calibration diagrams per model |
| `results/figures/reliability_all_models.png` | Combined calibration panel |
| `results/figures/topk_recall_curves.png` | **Main paper figure** |
| `results/figures/macro_f1_vs_truncation.png` | F1 vs observation fraction |
| `results/figures/training_*.png` | Loss/F1 curves per DL model |
| `checkpoints/*.pt` | Best PyTorch checkpoints |

---

## Models

| # | Name | Architecture | Notes |
|---|------|-------------|-------|
| 1 | `xgboost` | XGBoost on 15 hand-crafted features | Baseline |
| 2 | `transformer` | 2-layer Transformer, d=64, 4 heads, sinusoidal time encoding | Trained from scratch |
| 3 | `astromer` | Frozen Astromer1/2 encoder + 2-layer head | Falls back to `pip install astromer` |
| 4 | `moirai` | Frozen Moirai-small or Chronos-small encoder + head | Falls back to Chronos T5 |

All deep models use **MC Dropout** (p=0.1, 50 forward passes) for uncertainty.

---

## Hyperparameters

All hyperparameters are centralised in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_SEQ_LEN` | 256 | Fixed sequence length (pad/truncate) |
| `D_MODEL` | 64 | Transformer hidden dimension |
| `N_HEADS` | 4 | Attention heads |
| `N_ENCODER_LAYERS` | 2 | Transformer encoder layers |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `MAX_EPOCHS` | 50 | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | 5 | Early stopping patience |
| `MC_SAMPLES` | 50 | MC Dropout forward passes |
| `SEED` | 42 | Global random seed |

---

## Reproducing paper figures

The primary result figure (`topk_recall_curves.png`) is generated automatically
by `evaluate.py` / `main.py`.  To regenerate it from saved MC inference results:

```python
from prioritization import run_prioritization, plot_topk_curves
import pickle, config

with open("results/mc_results.pkl", "rb") as f:
    mc = pickle.load(f)

label_map = ...   # from data_loader.build_label_map
run_prioritization(mc, label_map)
```

---

## Colab

See [RUN_IN_COLAB.md](RUN_IN_COLAB.md) for step-by-step Colab instructions.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{plasticc_pipeline,
  title  = {Uncertainty-Aware Early Transient Classification Framework},
  year   = {2025},
  url    = {https://github.com/YOUR_USERNAME/plasticc_pipeline}
}
```
