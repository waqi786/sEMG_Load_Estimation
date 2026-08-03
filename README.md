<div align="center">

<img src="assets/banner.png">

<br><br>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-009639?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Estimating external load, joint torque (from each subject's real arm length and recorded joint angle), and a physiologically-grounded Hill-type EMG-to-force model, from surface Electromyography (sEMG) signals using 37 model input features, 10 classifiers, 11 regressors, an interpretable equation, a 1D CNN on raw signals, and subject-grouped / Leave-One-Subject-Out validation**

[Overview](#overview) · [Dataset](#dataset) · [Methodology](#methodology) · [Results](#results) · [Installation](#installation) · [Usage](#usage) · [Repository Structure](#repository-structure) · [Honest Limitations](#honest-limitations) · [Future Work](#future-work)

</div>

---

## Overview

This repository implements a complete, honestly-validated **machine learning pipeline** for estimating **external load** from **surface Electromyography (sEMG)** signals recorded at the elbow, extended with a **real, per-subject joint torque estimate** (using each subject's own measured arm length and their own recorded joint angle) and a **physiologically-grounded Hill-type EMG-to-force model**, including a calibration-free variant validated with Leave-One-Subject-Out testing.

The pipeline covers the full workflow end to end:

| Stage | Description |
|-------|-------------|
| **Data Loading** | Automatic download & extraction of the open EMG Elbow Dataset from Zenodo (10 subjects, 130 files) |
| **Feature Engineering** | 37 model input features: 10 statistical/shape/frequency features from each of the 2 filtered sEMG channels, 7 kinematic features from the joint angle, plus age, height, weight, arm length, hand length, exercise, and set type |
| **EDA** | Raw waveform plots, histograms, correlation heatmaps, per-subject response curves |
| **Frequency Analysis** | Welch PSD, spectrograms, median/mean frequency vs. load |
| **Subject-Wise Split** | Train/test split by subject (never by row) to avoid identity leakage |
| **Supervised Classification** | 10 classifiers for discrete load-level prediction |
| **Supervised Regression** | 11 regressors for continuous load estimation (grams) |
| **Interpretable Model** | Explicit polynomial equation linking features to load, full equation exported to CSV |
| **Real Per-Subject Torque** | Load converted to joint torque using each subject's own measured arm length and their own recorded joint angle (not an assumed constant) |
| **Hill-Type EMG-to-Force Model** | Physiologically-grounded Biceps/Triceps force model (F_max, alpha) fit on Flexion-Extension data, validated on held-out test subjects |
| **Calibration-Free Torque Model** | Anthropometric scaling of Hill-model strength parameters, validated with Leave-One-Subject-Out so a new subject needs zero personal EMG-torque calibration |
| **Feature Importance** | Random Forest, XGBoost, Permutation, and SHAP importances compared |
| **Unsupervised Clustering** | 8 clustering algorithms evaluated with ARI / Silhouette / Homogeneity |
| **Learning Curves** | Subject-grouped (`GroupKFold`) overfitting/underfitting diagnostics |
| **Cross-Validation** | Subject-grouped cross-validation for 6 regressors |
| **Leave-One-Subject-Out (LOSO)** | Every one of the 10 subjects used as the held-out test subject exactly once |
| **Deep Learning (MLP)** | Architecture search over 5 MLP configurations for classification & regression |
| **Deep Sequence Model (1D CNN)** | CNN trained directly on raw 3-channel sEMG windows, with subject-disjoint fit/validation/test sets |

> **Author:** Waqar Ali

---

## Dataset

We use the **[EMG Elbow Dataset](https://zenodo.org/record/7946782/files/EMG%20elbow%20dataset.zip)** hosted on Zenodo (Record `7946782`).

| Property | Value |
|----------|-------|
| **Subjects** | 10 (real subjects only — no synthetic or duplicated data) |
| **EMG Channels** | 5 per recording (2 raw, 2 filtered, 1 joint angle) |
| **Sampling Rate** | 2000 Hz |
| **Load Conditions** | 0 g · 1360 g · 2270 g |
| **Exercises** | Flexion–Extension (`flex`) · Pronation–Supination (`pronsup`) |
| **Total Recordings** | 130 raw text files across all subjects/conditions |
| **File Format** | `{subj}_{exercise}_{set_type}_{load}.txt` |
| **Subject Metadata** | Each subject's `subject_info` file provides age, height, weight, arm length, hand length, and sex as explicit `key:value` pairs, parsed by key name (not guessed by position) |
| **Target Variable** | Load (grams) — the primary ground-truth label; real torque (Nm) is then derived from each subject's own arm length and recorded joint angle |

The dataset is **automatically downloaded and extracted** at the start of the notebook — no manual setup required.

> Every split, score, and plot in this project is computed from these same 10 real subjects. This is a property of the public source data, not a design choice — using more subjects would require a larger dataset (see [Future Work](#future-work)).

---

## Methodology

### 1. Feature Engineering (37 Model Input Features)

**10 features per filtered channel × 2 channels (Channel 1 and Channel 2) + 7 joint-angle/kinematic features + 5 anthropometric fields + 2 encoded categoricals = 37 total model input columns.** Raw EMG is kept only to verify the filtered signal against it and is not used as an extra model input.

| Feature | Description |
|---------|-------------|
| `chN_mean` | Average amplitude |
| `chN_std` | Signal variability |
| `chN_max` / `chN_min` | Signal extrema |
| `chN_rms` | Root mean square amplitude |
| `chN_p2p` | Peak-to-peak amplitude (max − min) |
| `chN_energy` | Sum of squared samples |
| `chN_zcr` | Zero-crossing rate |
| `chN_skew` | Distribution skewness |
| `chN_kurt` | Distribution kurtosis |
| `chN_meanfreq` | Power-weighted mean frequency (Welch PSD) |
| `chN_peakfreq` | Dominant frequency (Welch PSD) |
| `angle_mean/std/min/max/p2p` | Joint angle summary statistics |
| `angle_mean_abs_vel` / `angle_max_abs_vel` | Joint angular velocity statistics |
| `age`, `height`, `weight`, `arm_length`, `hand_length` | Per-subject anthropometric fields (from `subject_info`) |
| `exercise`, `set_type` | Encoded metadata columns |

Muscle mapping depends on exercise: Channel 1/Channel 2 correspond to **Biceps Brachii / Triceps Brachii** during Flexion-Extension, and **Pronator Teres / Biceps Brachii** during Pronation-Supination.

### 2. Preprocessing & Splitting

- `StandardScaler` normalization fit on train only, applied to test
- `LabelEncoder` for categorical variables (`exercise`, `set_type`, load classes)
- **Subject-wise train/test split** (75% / 25% of subjects, `random_state=42`) — a subject's recordings never appear on both sides
- All cross-validation uses `GroupKFold` / `LeaveOneGroupOut`, grouped by subject, so no fold ever mixes one subject's data between train and validation
- Missing anthropometric values are filled with the dataset median (and logged); rows missing arm length are dropped only for the torque-specific models, with the dropped count reported honestly rather than silently

### 3. Models Evaluated

#### Supervised Classification (10 Models)
Logistic Regression · KNN (k=5) · KNN (k=7) · SVM (RBF) · SVM (Linear) · Decision Tree · Random Forest · Gradient Boosting · AdaBoost · XGBoost

#### Supervised Regression (11 Models)
Linear · Ridge · Lasso · ElasticNet · Bayesian Ridge · SVR (RBF) · SVR (Linear) · Decision Tree · Random Forest · Gradient Boosting · XGBoost

#### Interpretable Model
Polynomial `LinearRegression` (degree 1–3 searched), full coefficient table exported so the load equation can be read term-by-term rather than hidden in a black box.

#### Real Per-Subject Torque
`Torque = load(kg) * g * arm_length_m * sin(joint_angle)`, using each subject's own measured arm length and their own recorded joint angle — not a fixed assumed constant.

#### Hill-Type EMG-to-Force Model
`F = F_max * (1 - exp(-alpha * RMS))` fit per muscle (Biceps and Triceps) on Flexion-Extension data, combined as `Torque = arm_length * (F_Biceps - F_Triceps)`, giving physically meaningful parameters (peak force capacity and saturation rate) instead of opaque polynomial coefficients.

#### Calibration-Free Torque Model
Hill-model strength parameters (`F_max`) are regressed against each subject's arm length and body weight, then validated with Leave-One-Subject-Out — every prediction comes from a subject who contributed zero personal EMG-torque calibration data.

#### Unsupervised Clustering (8 Models)
KMeans (k=3) · KMeans (k=4) · Agglomerative · Gaussian Mixture · DBSCAN · OPTICS · Birch · MeanShift

#### Deep Learning
- **MLP architecture search** — 5 hidden-layer configurations, for both classification and regression, on the 37 engineered features
- **1D CNN** — trained directly on raw, per-channel-normalized 4000-sample (2s) windows of the 2 filtered EMG channels + joint angle channel, with early stopping on a subject-disjoint validation set and a fully held-out subject-disjoint test set

### 4. Validation Strategy

- **Single subject-wise split** — quick, fair estimate but sensitive to which subjects land in test
- **Subject-grouped cross-validation** (`GroupKFold`) — used for learning curves and 6-model CV comparison
- **Leave-One-Subject-Out (LOSO)** — the most robust estimate: every one of the 10 subjects is held out and tested exactly once, then results are averaged; also used to validate the calibration-free torque model

---

## Results

All figures are generated at 600 DPI and organized in [`results/`](results/) by category. Numbers below come directly from the notebook run bundled with this repository (no numbers are invented).

### Best Model Performance (Single Subject-Wise Split)

| Task | Best Model | Score |
|------|-----------|-------|
| **Classification** | Random Forest | **100.0% Accuracy** |
| **Regression** | Random Forest | **R² = 0.959**, RMSE = 189.4 g |
| **Interpretable Equation** | Best degree searched (1–3) | See notebook output for selected degree and R² |
| **Real Per-Subject Torque** | Linear (real arm length + real joint angle) | R² = 0.242, RMSE = 1.938 Nm |
| **Hill-Type Model (Flexion-Extension, calibrated)** | Biceps/Triceps Hill-type fit | R² = 0.835, RMSE = 0.850 Nm |
| **Calibration-Free Torque Model (LOSO)** | Anthropometric-scaled Hill model | R² = 0.307, RMSE = 1.737 Nm |
| **Unsupervised (best ARI)** | See notebook output | Clustering does not separate load classes well |
| **MLP (Deep Learning, features)** | (128,64,32) clf / (128,64,32) reg | Acc = 0.833 / R² = 0.836 |
| **1D CNN (raw signal, unseen subjects)** | — | Test Accuracy = 0.833 |

### Leave-One-Subject-Out (LOSO) — Most Trustworthy Estimate

| Task | Mean ± Std (across all 10 subjects) |
|------|--------------------------------------|
| **Classification Accuracy** | 0.875 ± 0.136 |
| **Regression R²** | 0.891 ± 0.123 |
| **Regression RMSE** | 267.5 g ± 152.9 g |

> The LOSO average is reasonably close to the single-split test accuracy, which supports that the earlier single-split result was not just a lucky split.

### Key Findings

- **EMG amplitude and mean frequency both increase with load** — confirming sEMG amplitude and spectral content both encode applied force.
- **Tree-based ensembles win** — Random Forest and XGBoost outperform linear models and simple neural networks on the 37 engineered features.
- **Subject-wise splitting matters** — subject-grouped CV and LOSO both give more conservative, trustworthy scores than plain row-wise splitting.
- **Clustering fails without labels** — load classes overlap heavily in feature space, confirming supervised learning is the right approach here.
- **The interpretable polynomial equation** gives a fully transparent, exportable set of coefficients — a usable trade-off between accuracy and explainability.
- **Real per-subject torque uses genuine measured values** — arm length and recorded joint angle come from each subject's own data, not an assumed constant, though the resulting R² (0.242) shows this rigid-pendulum approximation alone is a weak predictor of the real, noisy recorded joint angle.
- **The Hill-type model is far more physiologically informative** — fitting Biceps/Triceps force-activation curves directly (R² = 0.835 on held-out subjects) captures the true nonlinear EMG-to-force relationship much better than the simple kinematic torque formula.
- **The calibration-free model is an honest, harder proof-of-concept** — using only arm length and body weight for a brand-new subject (R² = 0.307 under LOSO) is expectedly lower than the subject-calibrated Hill model, since it removes the need for any personal calibration trial.
- **Feature importance techniques agree** — Random Forest, XGBoost, Permutation Importance, and SHAP all converge on similar informative channels/statistics.
- **Deep learning underperforms tree-based models** on this dataset size — expected, since MLPs and CNNs typically need far more samples than the few hundred available here; the CNN also shows a validation/training accuracy gap consistent with overfitting on this small sample size.

### Results Folder Layout

Outputs are grouped into clearly named subfolders so nothing is confusing to navigate:

```
results/
├── eda/                              # Exploratory data analysis figures
│   ├── data_distribution.jpg
│   ├── eda_signal_waveforms.jpg
│   ├── eda_correlation.jpg
│   ├── per_subject_analysis.jpg
│   ├── eda_frequency_analysis.jpg
│   ├── eda_mean_freq_boxplot.jpg
│   └── subject_info_relationships.jpg
├── models/                           # Classification, regression, torque, interpretability & clustering figures
│   ├── supervised_10models.jpg
│   ├── regression_11models.jpg
│   ├── interpretable_model_equation.jpg
│   ├── torque_model.jpg
│   ├── hill_type_model.jpg
│   ├── calibration_free_hill_model.jpg
│   ├── feature_importance.jpg
│   └── unsupervised_8models.jpg
├── validation/                       # Overfitting checks & generalisation validation
│   ├── learning_curve_XGBoost.jpg
│   ├── learning_curve_Random_Forest.jpg
│   ├── cross_validation.jpg
│   └── loso_validation.jpg
├── deep_learning/                    # Raw-signal 1D CNN results
│   ├── cnn_training_curves.jpg
│   └── cnn_confusion_matrix.jpg
└── tables/                           # Exported numeric results
    ├── feature_importance.csv
    └── interpretable_equation_full.csv
```

| Folder | Contents |
|--------|----------|
| [`results/eda/`](results/eda/) | Raw waveform plots, histograms, correlation heatmaps, per-subject curves, PSD & spectrogram figures, subject-info relationship plots |
| [`results/models/`](results/models/) | Classifier & regressor comparisons, interpretable equation plot, real torque model, Hill-type model, calibration-free torque model, feature importance, clustering |
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
│   ├── models/                                # Classification / regression / torque / interpretability / clustering figures
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
- **Real per-subject torque uses a rigid-pendulum approximation** — it uses each subject's own measured arm length and their own recorded joint angle (not an assumed constant), but the underlying rigid-pendulum formula and the angle's zero-reference are still standard modeling simplifications, and the resulting R² (0.242) shows the recorded joint angle alone does not tightly track true torque.
- **Hill-type model's alpha_t parameter is not well constrained** — during the calibrated fit, `alpha_t` landed at its upper bound (30), meaning the Triceps RMS range in this recording is too narrow to see clear saturation. This specific parameter should be treated as an upper-bound estimate, not a precise physiological constant.
- **Calibration-free torque model is a proof-of-concept** — with only 10 subjects, the anthropometric scaling from arm length and body weight is illustrative of the approach rather than a clinically validated calibration-free system.
- **Small-sample deep learning** — both the MLP and CNN are working with only a few hundred samples/windows, so their scores are noisier and generally behind the tree-based models trained on engineered features, and the CNN shows some overfitting (training/validation accuracy gap).
- **Single-split scores can be optimistic** — always prefer the LOSO numbers when judging real-world generalisation.

---

## Future Work

- Collect a larger, multi-site dataset with more subjects to reduce variance in LOSO estimates and better constrain Hill-type model parameters like `alpha_t`.
- Extend the calibration-free torque model with more anthropometric predictors (e.g., limb circumference, muscle cross-sectional area) to improve its Leave-One-Subject-Out accuracy.
- Explore transfer learning / domain adaptation across subjects to improve CNN generalisation and reduce the observed overfitting gap.
- Extend the CNN to a CNN-LSTM or Transformer-based sequence model for raw sEMG.
- Real-time deployment on edge/wearable devices for rehabilitation monitoring.

---

<div align="center">
  <img src="assets/last.png">
</div>
