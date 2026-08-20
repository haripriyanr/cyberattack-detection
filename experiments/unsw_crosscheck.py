"""UNSW-NB15 cross-check: does the NSL-KDD approach work on a modern dataset?

Tries to download UNSW-NB15 CSVs from public mirrors. If the network is
blocked, it falls back to a documented manual-download path and still exits
cleanly (so CI does not break).

   python experiments/unsw_crosscheck.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from common import save_metrics, save_fig

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
UNSW_TRAIN_URLS = [
    "https://github.com/Nir-J/ML-Projects/raw/refs/heads/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv?download=true",
    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_training-set.csv?download=true",
]
UNSW_TEST_URLS = [
    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_testing-set.csv?download=true",
    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_testing-set.csv?download=true",
    "https://github.com/Nir-J/ML-Projects/raw/refs/heads/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
]


def try_download(urls, dest: Path) -> bool:
    if dest.exists():
        print(f"Found {dest}")
        return True
    for url in urls:
        try:
            print(f"Trying {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            print(f"Downloaded to {dest}")
            return True
        except Exception as e:
            print(f"  failed: {e}")
    return False


def preprocess_unsw(train_path: Path, test_path: Path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    # UNSW has label (0 normal, 1 attack) and attack_cat
    for df in (train, test):
        df.columns = [c.strip() for c in df.columns]
    # Drop non-numeric / id columns that leak
    drop_cols = [c for c in ["id", "attack_cat"] if c in train.columns]
    y_train = train["label"].astype(int)
    y_test = test["label"].astype(int)
    X_train = train.drop(columns=drop_cols + ["label"])
    X_test = test.drop(columns=drop_cols + ["label"])
    # One-hot categoricals
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    X_all = pd.concat([X_train, X_test], axis=0)
    X_all = pd.get_dummies(X_all, columns=cat_cols)
    X_tr = X_all.iloc[: len(X_train)]
    X_te = X_all.iloc[len(X_train):]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_te_s, y_train.values, y_test.values


def main() -> None:
    print("=" * 60)
    print("UNSW-NB15 cross-check")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

    ok_train = try_download(UNSW_TRAIN_URLS, train_path)
    ok_test = try_download(UNSW_TEST_URLS, test_path)

    if not (ok_train and ok_test):
        print("\nUNSW-NB15 not available automatically.")
        print("Manual download:")
        print("  1. Go to https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print("  2. Download The UNSW-NB15_* CSV Files.zip, unzip into data/raw/")
        print("  3. Re-run: python experiments/unsw_crosscheck.py")
        # Still write a placeholder so reports/ exists
        save_metrics(
            pd.DataFrame([{"note": "UNSW-NB15 not downloaded - manual step required"}]),
            "unsw_crosscheck.csv",
        )
        return

    X_tr, X_te, y_tr, y_te = preprocess_unsw(train_path, test_path)
    print(f"UNSW shapes: train {X_tr.shape} test {X_te.shape}")

    # Train Isolation Forest on UNSW normal-only
    normal_idx = np.where(y_tr == 0)[0]
    rng = np.random.RandomState(42)
    if len(normal_idx) > 40000:
        normal_idx = rng.choice(normal_idx, 40000, replace=False)
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(X_tr[normal_idx])
    scores = -iso.decision_function(X_te)
    auc = roc_auc_score(y_te, scores)
    # best F1
    best_f1 = 0
    for t in np.percentile(scores, np.linspace(0, 100, 101)):
        f1 = f1_score(y_te, scores >= t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    prec = precision_score(y_te, scores >= best_t)
    rec = recall_score(y_te, scores >= best_t)
    print(f"\nUNSW-NB15 IsolationForest - AUC={auc:.4f} F1={best_f1:.4f} prec={prec:.4f} rec={rec:.4f}")

    # Compare to NSL-KDD anomaly for reference (if reports exist)
    save_metrics(
        pd.DataFrame(
            [
                {
                    "dataset": "UNSW-NB15",
                    "model": "IsolationForest (normal-only)",
                    "roc_auc": round(auc, 4),
                    "f1": round(best_f1, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                }
            ]
        ),
        "unsw_crosscheck.csv",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["UNSW-NB15"], [auc], color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Isolation Forest on UNSW-NB15 (normal-only training)")
    fig.tight_layout()
    save_fig(fig, "unsw_crosscheck.png")
    plt.close(fig)


if __name__ == "__main__":
    main()