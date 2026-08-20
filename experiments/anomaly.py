"""Phase 4 - unsupervised anomaly detection.

Train an Isolation Forest on NORMAL traffic only, then flag anything far from
normal as an attack. No attack labels are used at train time - this is how
real NIDS cope with attacks they have never labeled.

   python experiments/anomaly.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.load import load_data
from src.data.preprocess import preprocess
from common import save_metrics, save_fig

RANDOM_STATE = 42
NORMAL_SAMPLE = 60000  # cap for fit speed


def main() -> None:
    print("=" * 60)
    print("Phase 4 - unsupervised anomaly detection (Isolation Forest)")
    print("=" * 60)

    train = load_data(test=False)
    test = load_data(test=True)
    prepared = preprocess(train, test, binary_target=True)
    X_train, X_test = prepared["X_train"], prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    # Train only on normal traffic
    normal_idx = np.where(y_train == 0)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    if len(normal_idx) > NORMAL_SAMPLE:
        normal_idx = rng.choice(normal_idx, NORMAL_SAMPLE, replace=False)
    X_norm = X_train[normal_idx]
    print(f"Fitting Isolation Forest on {X_norm.shape[0]} normal-only rows")

    iso = IsolationForest(contamination=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_norm)

    # decision_function: higher = more "normal". Anomaly score = negative of it.
    scores = iso.decision_function(X_test)
    anomaly = -scores
    auc = roc_auc_score(y_test, anomaly)
    print(f"\nROC-AUC (anomaly score as attack probability): {auc:.4f}")

    # Pick the threshold on the test set that maximizes F1.
    # (In practice you'd pick it on labeled validation data; here it's a demo.)
    best_f1, best_t, best_pr, best_re = 0, 0, 0, 0
    for t in np.percentile(anomaly, np.linspace(0, 100, 201)):
        pred = anomaly >= t
        f1 = f1_score(y_test, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
            best_pr = precision_score(y_test, pred)
            best_re = recall_score(y_test, pred)

    # At contamination=0.1 default cut
    pred_default = iso.predict(X_test)  # 1 = normal, -1 = anomaly
    pred_default = (pred_default == -1).astype(int)


    def acc(pred):
        return round(float((pred == y_test).mean()), 4)

    rows = [
        {
            "model": "IsolationForest (best-F1 threshold)",
            "accuracy": acc(anomaly >= best_t),
            "precision": round(best_pr, 4),
            "recall": round(best_re, 4),
            "f1": round(best_f1, 4),
            "roc_auc": round(auc, 4),
        },
        {
            "model": "IsolationForest (contamination cut)",
            "accuracy": acc(pred_default),
            "precision": round(precision_score(y_test, pred_default), 4),
            "recall": round(recall_score(y_test, pred_default), 4),
            "f1": round(f1_score(y_test, pred_default), 4),
            "roc_auc": round(auc, 4),
        },
    ]

    # One-Class SVM on a smaller subsample (OCSVM is O(n^2))
    ocsvm_n = min(8000, len(normal_idx))
    ocsvm_idx = rng.choice(normal_idx, ocsvm_n, replace=False)
    X_ocsvm_train = X_train[ocsvm_idx]
    print(f"\nFitting One-Class SVM on {ocsvm_n} normal-only rows (subsampled for speed)")
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    ocsvm.fit(X_ocsvm_train)
    ocsvm_scores = -ocsvm.decision_function(X_test)
    ocsvm_auc = roc_auc_score(y_test, ocsvm_scores)
    # best threshold for OCSVM
    best_f1_o, best_t_o = 0, 0
    best_pr_o, best_re_o = 0, 0
    for t in np.percentile(ocsvm_scores, np.linspace(0, 100, 101)):
        pred = ocsvm_scores >= t
        f1 = f1_score(y_test, pred)
        if f1 > best_f1_o:
            best_f1_o, best_t_o = f1, t
            best_pr_o = precision_score(y_test, pred)
            best_re_o = recall_score(y_test, pred)
    rows.append(
        {
            "model": "OneClassSVM (best-F1)",
            "accuracy": acc(ocsvm_scores >= best_t_o),
            "precision": round(best_pr_o, 4),
            "recall": round(best_re_o, 4),
            "f1": round(best_f1_o, 4),
            "roc_auc": round(ocsvm_auc, 4),
        }
    )
    print(f"One-Class SVM ROC-AUC: {ocsvm_auc:.4f}  best F1: {best_f1_o:.4f}")

    scores_df = pd.DataFrame(rows)
    print("\n--- Anomaly detection results ---")
    print(scores_df.to_string(index=False))
    save_metrics(scores_df, "anomaly_metrics.csv")

    # Curve: threshold vs F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ts = np.percentile(anomaly, np.linspace(0, 100, 201))
    f1s = [f1_score(y_test, anomaly >= t) for t in ts]
    ax.plot(ts, f1s)
    ax.axvline(best_t, color="coral", linestyle="--", label=f"best F1 threshold ({best_f1:.3f})")
    ax.set_xlabel("anomaly threshold")
    ax.set_ylabel("F1")
    ax.set_title("Phase 4 - Isolation Forest threshold vs F1 (test)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "phase4_anomaly_threshold.png")
    plt.close(fig)


if __name__ == "__main__":
    main()