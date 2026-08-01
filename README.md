<div align="center">

<img src="assets/banner.png">

<br><br>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-009639?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Estimating external load (and an illustrative joint torque extension) from surface Electromyography (sEMG) signals using 60 engineered features, 10 classifiers, 11 regressors, an interpretable equation, a 1D CNN on raw signals, and subject-grouped / Leave-One-Subject-Out validation.**

[Overview](#overview) · [Dataset](#dataset) · [Methodology](#methodology) · [Results](#results) · [Installation](#installation) · [Usage](#usage) · [Repository Structure](#repository-structure) · [Honest Limitations](#honest-limitations) · [Future Work](#future-work)

</div>

---

## Overview

This repository implements a complete, honestly-validated **machine learning pipeline** for estimating **external load** from **surface Electromyography (sEMG)** signals recorded at the elbow, with an additional illustrative section that converts predicted load into an assumption-based joint torque estimate.

The pipeline covers the full workflow end to end:

| Stage | Description |
|-------|-------------|
| **Data Loading** | Automatic download & extraction of the open EMG Elbow Dataset from Zenodo (10 subjects, 130 files) |
| **Feature Engineering** | 60 statistical, shape, and frequency-domain features from 5 sEMG channels (12 features/channel) |
| **EDA** | Raw waveform plots, histograms, correlation heatmaps, per-subject response curves |
| **Frequency Analysis** | Welch PSD, spectrograms, median/mean frequency vs. load |
| **Subject-Wise Split** | Train/test split by subject (never by row) to avoid identity leakage |
| **Supervised Classification** | 10 classifiers for discrete load-level prediction |
| **Supervised Regression** | 11 regressors for continuous load estimation (grams) |
| **Interpretable Model** | Explicit polynomial equation linking features to load, full equation exported to CSV |
| **Torque Illustration** | Assumption-based load-to-torque conversion, clearly labeled as an estimate, not a measurement |
| **Feature Importance** | Random Forest, XGBoost, Permutation, and SHAP importances compared |
| **Unsupervised Clustering** | 8 clustering algorithms evaluated with ARI / Silhouette / Homogeneity |
| **Learning Curves** | Subject-grouped (`GroupKFold`) overfitting/underfitting diagnostics |
| **Cross-Validation** | Subject-grouped cross-validation for 6 regressors |
| **Leave-One-Subject-Out (LOSO)** | Every one of the 10 subjects used as the held-out test subject exactly once |
| **Deep Learning (MLP)** | Architecture search over 5 MLP configurations for classification & regression |
| **Deep Sequence Model (1D CNN)** | CNN trained directly on raw 5-channel sEMG windows, with subject-disjoint fit/validation/test sets |

> **Author:** Waqar Ali

---

## Dataset

We use the **[EMG Elbow Dataset](https://zenodo.org/record/7946782/files/EMG%20elbow%20dataset.zip)** hosted on Zenodo (Record `7946782`).

| Property | Value |
|----------|-------|
| **Subjects** | 10 (real subjects only — no synthetic or duplicated data) |
| **EMG Channels** | 5 per recording |
| **Sampling Rate** | 2000 Hz |
| **Load Conditions** | 0 g · 1360 g · 2270 g |
| **Exercises** | Flexion–Extension (`flex`) · Pronation–Supination (`pronsup`) |
| **Total Recordings** | 130 raw text files across all subjects/conditions |
| **File Format** | `{subj}_{exercise}_{set_type}_{load}.txt` |
| **Target Variable** | Load (grams) — the only ground-truth label the dataset provides |

The dataset is **automatically downloaded and extracted** at the start of the notebook — no manual setup required.

> Every split, score, and plot in this project is computed from these same 10 real subjects. This is a property of the public source data, not a design choice — using more subjects would require a larger dataset (see [Future Work](#future-work)).

---

## Methodology

### 1. Feature Engineering (60 Features)

**12 features per channel × 5 channels = 60 total engineered features:**

| Feature | Description |
|---------|-------------|
| `mean_i` | Average amplitude |
| `std_i` | Signal variability |
| `max_i` / `min_i` | Signal extrema |
| `rms_i` | Root mean square amplitude |
| `p2p_i` | Peak-to-peak amplitude (max − min) |
| `energy_i` | Sum of squared samples |
| `zcr_i` | Zero-crossing rate |
| `skew_i` | Distribution skewness |
| `kurt_i` | Distribution kurtosis |
| `meanfreq_i` | Power-weighted mean frequency (Welch PSD) |
| `peakfreq_i` | Dominant frequency (Welch PSD) |

Two encoded metadata columns (`exercise`, `set_type`) are appended, giving **62 total model input columns**.

### 2. Preprocessing & Splitting

- `StandardScaler` normalization fit on train only, applied to test
- `LabelEncoder` for categorical variables (`exercise`, `set_type`, load classes)
- **Subject-wise train/test split** (75% / 25% of subjects, `random_state=42`) — a subject's recordings never appear on both sides
- All cross-validation uses `GroupKFold` / `LeaveOneGroupOut`, grouped by subject, so no fold ever mixes one subject's data between train and validation

### 3. Models Evaluated

#### Supervised Classification (10 Models)
Logistic Regression · KNN (k=5) · KNN (k=7) · SVM (RBF) · SVM (Linear) · Decision Tree · Random Forest · Gradient Boosting · AdaBoost · XGBoost

#### Supervised Regression (11 Models)
Linear · Ridge · Lasso · ElasticNet · Bayesian Ridge · SVR (RBF) · SVR (Linear) · Decision Tree · Random Forest · Gradient Boosting · XGBoost

#### Interpretable Model
Polynomial `LinearRegression` (degree 1–3 searched), full coefficient table exported so the load equation can be read term-by-term rather than hidden in a black box.

#### Unsupervised Clustering (8 Models)
KMeans (k=3) · KMeans (k=4) · Agglomerative · Gaussian Mixture · DBSCAN · OPTICS · Birch · MeanShift

#### Deep Learning
- **MLP architecture search** — 5 hidden-layer configurations, for both classification and regression, on the 62 engineered features
- **1D CNN** — trained directly on raw, per-channel-normalized 4000-sample (2s) windows of all 5 sEMG channels, with early stopping on a subject-disjoint validation set and a fully held-out subject-disjoint test set

### 4. Validation Strategy

- **Single subject-wise split** — quick, fair estimate but sensitive to which 3 subjects land in test
- **Subject-grouped cross-validation** (`GroupKFold`) — used for learning curves and 6-model CV comparison
- **Leave-One-Subject-Out (LOSO)** — the most robust estimate: every one of the 10 subjects is held out and tested exactly once, then results are averaged

---

## Results

All figures are generated at 600 DPI and organized in [`results/`](results/) by category. Numbers below come directly from the notebook run bundled with this repository (no numbers are invented).

### Best Model Performance (Single Subject-Wise Split)

| Task | Best Model | Score |
|------|-----------|-------|
| **Classification** | AdaBoost | **100.0% Accuracy** |
| **Regression** | Random Forest | **R² = 0.959**, RMSE = 187.9 g |
| **Interpretable Equation** | Linear (degree 1) | R² = 0.870 |
| **Approximate Torque (illustrative)** | Linear | R² = 0.870, RMSE = 0.988 Nm |
| **Unsupervised (best ARI)** | Gaussian Mixture | ARI = 0.114 |
| **MLP (Deep Learning, features)** | (256,128,64) clf / (128,64,32) reg | Acc = 0.750 / R² = 0.893 |
| **1D CNN (raw signal, unseen subjects)** | — | Test Accuracy = 0.611 |

### Leave-One-Subject-Out (LOSO) — Most Trustworthy Estimate

| Task | Mean ± Std (across all 10 subjects) |
|------|--------------------------------------|
| **Classification Accuracy** | 0.867 ± 0.140 |
| **Regression R²** | 0.886 ± 0.121 |
| **Regression RMSE** | 276.1 g |

> The single-split accuracy (100%) is noticeably higher than the LOSO average (86.7%), which means the single split was optimistic. **LOSO is the number that should be quoted for real-world generalisation**, since it uses every subject as the test subject exactly once instead of depending on one lucky/unlucky split of 3 subjects.

### Key Findings

- **EMG amplitude and mean frequency both increase with load** — confirming sEMG amplitude and spectral content both encode applied force.
- **Tree-based ensembles win** — AdaBoost, Random Forest, and XGBoost outperform linear models and simple neural networks on the 62 engineered features.
- **Subject-wise splitting matters** — subject-grouped CV and LOSO both give more conservative, trustworthy scores than plain row-wise splitting.
- **Clustering fails without labels** — best unsupervised ARI is only 0.114, confirming load classes overlap heavily in feature space and supervised learning is the right approach here.
- **The interpretable linear equation** achieves R² = 0.870 with a fully transparent, exportable set of coefficients — a usable trade-off between accuracy and explainability.
- **The torque numbers are illustrative only** — they use an assumed forearm length (0.30 m) and elbow angle (90°), not measured per-subject values, and are explicitly labeled as such rather than presented as validated joint torque.
- **Feature importance techniques agree** — Random Forest, XGBoost, Permutation Importance, and SHAP all converge on similar informative channels/statistics (notably channel-2 energy).
- **No major overfitting** — subject-grouped learning curves and CNN training curves both show training and validation scores staying reasonably close.
- **Deep learning underperforms tree-based models** on this dataset size — expected, since MLPs and CNNs typically need far more samples than the few hundred available here.

### Results Folder Layout

Outputs are grouped into clearly named subfolders so nothing is confusing to navigate:

```
results/
├── eda/                 # Exploratory data analysis figures
│   ├── data_distribution.jpg
│   ├── eda_signal_waveforms.jpg
│   ├── eda_correlation.jpg
│   ├── per_subject_analysis.jpg
│   ├── eda_frequency_analysis.jpg
│   └── eda_mean_freq_boxplot.jpg
├── models/               # Classification, regression, interpretability & clustering figures
│   ├── supervised_10models.jpg
│   ├── regression_11models.jpg
│   ├── interpretable_model_equation.jpg
│   ├── approx_torque_model.jpg
│   ├── feature_importance.jpg
│   └── unsupervised_8models.jpg
├── validation/           # Overfitting checks & generalisation validation
│   ├── learning_curve_XGBoost.jpg
│   ├── learning_curve_Random_Forest.jpg
│   ├── cross_validation.jpg
│   └── loso_validation.jpg
├── deep_learning/        # Raw-signal 1D CNN results
│   ├── cnn_training_curves.jpg
│   └── cnn_confusion_matrix.jpg
└── tables/               # Exported numeric results
    ├── feature_importance.csv
    └── interpretable_equation_full.csv
```

| Folder | Contents |
|--------|----------|
| [`results/eda/`](results/eda/) | Raw waveform plots, histograms, correlation heatmaps, per-subject curves, PSD & spectrogram figures |
| [`results/models/`](results/models/) | Classifier & regressor comparisons, interpretable equation plot, torque illustration, feature importance, clustering |
| [`results/validation/`](results/validation/) | Subject-grouped learning curves, subject-grouped cross-validation, LOSO validation |
| [`results/deep_learning/`](results/deep_learning/) | 1D CNN training curves and confusion matrix on unseen test subjects |
| [`results/tables/`](results/tables/) | Full feature-importance table and full interpretable-equation coefficient table (CSV) |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/waqi786/sEMG_Load_Estimation.git
cd sEMG_Load_Estimation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Open and run the notebook top to bottom:

```bash
jupyter notebook notebooks/semg-joint-torque-estimation.ipynb
```

The notebook automatically downloads the dataset, engineers features, trains all models, runs every validation scheme, and regenerates every figure in `results/`.

---

## Repository Structure

```
sEMG_Load_Estimation/
├── notebooks/
│   └── semg-joint-torque-estimation.ipynb   # Full end-to-end analysis notebook
├── results/
│   ├── eda/                                  # Exploratory data analysis figures
│   ├── models/                                # Classification / regression / interpretability / clustering figures
│   ├── validation/                            # Learning curves, cross-validation, LOSO
│   ├── deep_learning/                         # 1D CNN training curves & confusion matrix
│   └── tables/                                # Exported CSV result tables
├── assets/                                    # Banner and closing image
├── requirements.txt                           # Python dependencies
├── LICENSE                                    # MIT License
└── README.md                                  # This file
```

---

## Honest Limitations

- **Only 10 subjects** — a property of the public dataset, not a choice made here. No subjects are invented or duplicated.
- **Torque is illustrative, not measured** — real per-subject forearm length and elbow angle were not recorded in this dataset, so the torque section uses stated, fixed assumptions (0.30 m moment arm, 90° elbow angle) purely to demonstrate the load-to-torque conversion. It should not be treated as a validated biomechanical torque measurement.
- **Small-sample deep learning** — both the MLP and CNN are working with only a few hundred samples/windows, so their scores are noisier and generally behind the tree-based models trained on engineered features.
- **Single-split scores can be optimistic** — always prefer the LOSO numbers when judging real-world generalisation.

---

## Future Work

- Collect a larger, multi-site dataset with more subjects to reduce variance in LOSO estimates.
- Record real per-subject forearm length and elbow angle (e.g., via goniometer or motion capture) to compute genuine, validated joint torque instead of an assumption-based illustration.
- Explore transfer learning / domain adaptation across subjects to improve CNN generalisation.
- Extend the CNN to a CNN-LSTM or Transformer-based sequence model for raw sEMG.
- Real-time deployment on edge/wearable devices for rehabilitation monitoring.

---

## Thank You

<div align="center">
  <img src="assets/last.png">
</div>
