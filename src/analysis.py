"""
sEMG-based Joint Torque & Load Estimation — Analysis Pipeline

Complete machine learning pipeline for estimating joint load from surface
Electromyography (sEMG) signals using the EMG Elbow Dataset (Zenodo 7946782).

Author: Waqar Ali
"""

import glob
import os
import re
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy import signal, stats
from scipy.signal import spectrogram, welch
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    homogeneity_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results"
DATASET_URL = (
    "https://zenodo.org/record/7946782/files/EMG%20elbow%20dataset.zip?download=1"
)
RANDOM_STATE = 42
N_CHANNELS = 5

plt.rcParams.update(
    {
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.linewidth": 1.5,
        "figure.figsize": (12, 6),
    }
)

PRIMARY = ["#FF7F0E", "#1F77B4"]
SECONDARY = ["#2CA02C", "#D62728", "#9467BD"]


def download_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract the EMG Elbow Dataset from Zenodo."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "emg_elbow_dataset.zip"
    extract_dir = data_dir / "emg"

    if extract_dir.exists() and any(extract_dir.rglob("*.txt")):
        print(f"Dataset already present at {extract_dir}")
        return extract_dir

    print("Downloading EMG Elbow Dataset from Zenodo...")
    response = requests.get(DATASET_URL, stream=True, timeout=120)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    zip_path.unlink(missing_ok=True)
    print(f"Dataset ready at {extract_dir}")
    return extract_dir


def extract_features(data_dir: Path) -> pd.DataFrame:
    """Extract time-domain statistical features from all EMG recordings."""
    all_records = []

    for filepath in sorted(data_dir.rglob("*.txt")):
        if "subject_info" in str(filepath):
            continue

        parts = filepath.stem.split("_")
        if len(parts) != 4:
            continue

        subj, exercise, set_type, load = int(parts[0]), parts[1], parts[2], int(parts[3])
        data = np.loadtxt(filepath)

        means = data.mean(axis=0)
        stds = data.std(axis=0)
        maxs = data.max(axis=0)
        mins = data.min(axis=0)
        rms = np.sqrt(np.mean(data**2, axis=0))
        p2p = maxs - mins
        energy = np.sum(data**2, axis=0)
        zcr = np.sum(np.diff(np.sign(data), axis=0) != 0, axis=0) / data.shape[0]

        row = [subj, exercise, set_type, load]
        row += list(means) + list(stds) + list(maxs) + list(mins)
        row += list(rms) + list(p2p) + list(energy) + list(zcr)
        all_records.append(row)

    cols = ["subj", "exercise", "set_type", "load"]
    for i in range(N_CHANNELS):
        cols.extend(
            [
                f"mean_{i}",
                f"std_{i}",
                f"max_{i}",
                f"min_{i}",
                f"rms_{i}",
                f"p2p_{i}",
                f"energy_{i}",
                f"zcr_{i}",
            ]
        )

    df = pd.DataFrame(all_records, columns=cols)
    print(f"Feature matrix shape: {df.shape}")
    return df


def prepare_classification_data(df: pd.DataFrame):
    """Prepare scaled train/test splits for load classification."""
    feat_cols = [c for c in df.columns if c not in ["subj", "exercise", "set_type", "load"]]
    X = df[feat_cols].copy()

    le_ex = LabelEncoder()
    le_set = LabelEncoder()
    X["exercise"] = le_ex.fit_transform(df["exercise"])
    X["set_type"] = le_set.fit_transform(df["set_type"])

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(df["load"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=RANDOM_STATE, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, le_y


def run_classification(X_train, X_test, y_train, y_test, le_y) -> dict:
    """Train and evaluate 5 supervised classification models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = le_y.inverse_transform(model.predict(X_test))
        y_true = le_y.inverse_transform(y_test)
        acc = accuracy_score(y_true, y_pred)
        results[name] = acc
        print(f"  {name}: {acc:.1%} accuracy")

    return results


def run_regression(df: pd.DataFrame) -> dict:
    """Train and evaluate regression models for continuous load estimation."""
    feat_cols = [c for c in df.columns if c not in ["subj", "exercise", "set_type", "load"]]
    X = df[feat_cols].copy()

    le_ex = LabelEncoder()
    le_set = LabelEncoder()
    X["exercise"] = le_ex.fit_transform(df["exercise"])
    X["set_type"] = le_set.fit_transform(df["set_type"])
    y = df["load"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "SVR (RBF)": SVR(kernel="rbf"),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150, max_depth=6, random_state=RANDOM_STATE
        ),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {"r2": r2, "rmse": rmse}
        print(f"  {name}: R² = {r2:.3f}, RMSE = {rmse:.1f} g")

    return results


def run_unsupervised(X_scaled: np.ndarray, y_true: np.ndarray) -> dict:
    """Evaluate 5 unsupervised clustering algorithms."""
    models = {
        "KMeans": KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=3),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
        "Gaussian Mixture": GaussianMixture(n_components=3, random_state=RANDOM_STATE),
        "Birch": Birch(n_clusters=3),
    }

    results = {}
    for name, model in models.items():
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        ari = adjusted_rand_score(y_true, labels)
        hom = homogeneity_score(y_true, labels)
        results[name] = {"silhouette": sil, "ari": ari, "homogeneity": hom}
        print(f"  {name}: Silhouette = {sil:.3f}, ARI = {ari:.3f}")

    return results


def main():
    """Run the complete sEMG load estimation pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("sEMG-based Joint Torque & Load Estimation")
    print("=" * 60)

    data_dir = download_dataset()
    df = extract_features(data_dir)

    print("\n── Supervised Classification (5 Models) ──")
    X_train, X_test, y_train, y_test, le_y = prepare_classification_data(df)
    clf_results = run_classification(X_train, X_test, y_train, y_test, le_y)

    print("\n── Regression (5 Models) ──")
    reg_results = run_regression(df)

    print("\n── Unsupervised Clustering (5 Models) ──")
    X_all = StandardScaler().fit_transform(
        df[[c for c in df.columns if c not in ["subj", "exercise", "set_type", "load"]]]
    )
    unsup_results = run_unsupervised(X_all, df["load"].values)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Best classifier: {max(clf_results, key=clf_results.get)} "
          f"({max(clf_results.values()):.1%})")
    best_reg = max(reg_results, key=lambda k: reg_results[k]["r2"])
    print(f"Best regressor:  {best_reg} (R² = {reg_results[best_reg]['r2']:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
