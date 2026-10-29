# Option Pricing with KAN and KAT (Kolmogorov-Arnold Transformer)

Data-driven option pricing on Indian index options (NIFTY and BANKNIFTY) using supervised learning. The project starts by replicating a published single- and multi-asset pricing study, then moves to recent market data (2021-2026) and tests newer deep learning models: **KAN** (Kolmogorov-Arnold Networks) and **KAT** (Kolmogorov-Arnold Transformer).

## Background

The baseline is Goswami, Rajani and Tanksale (2021), *Data-driven option pricing using single and multi-asset supervised learning*. The workflow here:

1. Read the paper and replicate its results (ANN, DINN and XGBoost baselines).
2. Move the same pipeline to current NSE data (2021-2026).
3. Apply KAN with the same feature set.
4. Introduce KAT, a KAN + Transformer model, to fix KAN's scaling problems.

## Why KAT

The original KAN does not scale well because of two issues:

- **Parameter and compute inefficiency.** A separate learnable base function per input-output channel pair makes parameters and computation grow quickly with width and depth.
- **Poor weight initialization.** Variance-preserving initialization is needed so that signal variance stays constant across layers.

KAT addresses these by combining a Transformer backbone (attention blocks) with rational-activation KAN layers.

## Experiments

- **Single-asset, cross-asset and multi-asset** pricing setups, compared against the baselines and Black-Scholes.
- **Periods:** 2014-2018 including the COVID crash, and 2021-2026.
- **Small-data test:** trained and tested on H1 2026 only, to check behaviour under data scarcity.
- **Feature engineering:** reducing four log-return features to one in Approaches 2 and 3 improved KAN results (the originals were almost perfectly multicollinear). PCA at 95% variance hurt, as it removed important features such as covariance matrix entries.
- **Metrics:** EM and rho, reported per approach.

Known issue: on BANKNIFTY 2021-2026, the models currently do worse than Black-Scholes.

## Repository Structure

```
Option-pricing/
├── katransformer.py                          # KAT model (KAN layer and Transformer Block)
├── KAT_multi_asset_experiments.ipynb         # KAT multi-asset experiments
├── KAT_multi_asset_experiment_covid.ipynb    # KAT multi-asset experiments incl. COVID period
├── notebooks/
│   ├── ANN_XGB.ipynb                         # ANN and XGBoost baselines
│   ├── ANN_XGB_full_experiments.ipynb        # Full baseline experiments
│   ├── DINN_                                 # DINN baseline (TensorFlow/Keras)
│   ├── KAN Paper Data                        # KAN on the paper's data
│   ├── KAN(2021-2026)                        # KAN on 2021-2026 data
│   ├── KAN_multi_asset_experiments.ipynb     # KAN multi-asset experiments
│   ├── KAT                                   # KAT experiments
│   └── KAT_multi_asset_experiments.ipynb
├── kat/                                      # KAT code (git submodule)
└── rational_kat_cu/                          # CUDA rational activation (git submodule)
```

## Getting Started

```bash
git clone --recurse-submodules https://github.com/rahul820913/Option-pricing.git
cd Option-pricing
pip install -r requirements.txt
pip install -e rational_kat_cu     # needs a CUDA-enabled PyTorch and GPU toolchain
jupyter notebook
```

## Requirements
```
- Python 3.9+
- PyTorch with CUDA (for KAT and `rational_kat_cu`)
- TensorFlow and XGBoost (baseline models)
- PyKAN (KAN models)
- Full list in `requirements.txt`
```

Notes:

- The KAT notebooks import `katransformer.py`, which depends on `rational_kat_cu` (module `kat_rational`), so they need a CUDA GPU.
- The KAN notebooks use [PyKAN](https://github.com/KindXiaoming/pykan); the ANN, DINN and XGBoost baselines use TensorFlow/Keras and XGBoost.
- Many notebooks were written for Google Colab and mount Google Drive for data (`from google.colab import drive`). Update the data paths if you run them locally.

## Data

Option and index data for NIFTY and BANKNIFTY comes from the NSE website. Data and result files (`*.csv`, `*.xlsx`, `data_files/`, `result/`) are git-ignored and not included in the repository.

## References

1. A. Goswami, S. Rajani, A. Tanksale, "Data-driven option pricing using single and multi-asset supervised learning", *International Journal of Financial Engineering*, Vol. 8, No. 2 (2021), 2141001.
2. Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic, T. Y. Hou, M. Tegmark, "KAN: Kolmogorov-Arnold Networks".
3. X. Yang, X. Wang, "Kolmogorov-Arnold Transformer".

## Author

Rahul Singh Meena, B.Tech Mathematics and Computing, IIT Patna
[LinkedIn](https://www.linkedin.com/in/rahul-singh-meena-95ab16279) | [GitHub](https://github.com/rahul820913)
