"""Phase 1 - tune the workshop models, calibrate the winner, persist it.

   python experiments/tune_models.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from src.data.load import load_data
from src.data.preprocess import preprocess
from src.models.persist import save_bundle
from src.models.calibration import calibrate_isotonic
from common import save_metrics, save_fig

RANDOM_STATE = 42
TUNING_SIZE = 30000  # subsample for the grid; refit on full train afterwards


def metrics_dict(y_true, y_pred, proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
    }


def tune_knn(X, y):
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=TUNING_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tune, y_tune, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tune
    )
    rows = []
    for k in [3, 5, 7, 9, 11, 13]:
        for weights in ["uniform", "distance"]:
            m = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1)
            m.fit(X_tr, y_tr)
            rows.append(
                {
                    "k": k,
                    "weights": weights,
                    "val_f1": round(f1_score(y_val, m.predict(X_val)), 4),
                    "val_auc": round(
                        roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]), 4
                    ),
                }
            )
    grid = pd.DataFrame(rows).sort_values(["val_f1", "val_auc"], ascending=False)
    print("\nKNN tuning grid (top 5):")
    print(grid.head().to_string(index=False))
    best = grid.iloc[0]
    return int(best["k"]), best["weights"]


def tune_logistic(X, y):
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=TUNING_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tune, y_tune, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tune
    )
    rows = []
    for c in [0.1, 1.0, 10.0]:
        for cw in [None, "balanced"]:
            m = LogisticRegression(C=c, class_weight=cw, max_iter=2000, random_state=RANDOM_STATE)
            m.fit(X_tr, y_tr)
            rows.append(
                {
                    "C": c,
                    "class_weight": str(cw),
                    "val_f1": round(f1_score(y_val, m.predict(X_val)), 4),
                    "val_auc": round(
                        roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]), 4
                    ),
                }
            )
    grid = pd.DataFrame(rows).sort_values(["val_f1", "val_auc"], ascending=False)
    print("\nLogistic tuning grid (top 5):")
    print(grid.head().to_string(index=False))
    best = grid.iloc[0]
    return float(best["C"]), best["class_weight"]


def main() -> None:
    print("=" * 60)
    print("Phase 1 - tuning + calibration")
    print("=" * 60)

    train = load_data(test=False)
    test = load_data(test=True)
    prepared = preprocess(train, test, binary_target=True)
    X_train, X_test = prepared["X_train"], prepared["X_test"]
    y_train, y_test = prepared["y_train"], prepared["y_test"]

    # Tune each model on a subsample
    best_k, best_w = tune_knn(X_train, y_train)
    best_c, best_cw = tune_logistic(X_train, y_train)

    # Refit on the full training set with best params
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, n_jobs=-1)
    knn.fit(X_train, y_train)
    print(f"\nKNN refit on full train: k={best_k}, weights={best_w}")

    lr = LogisticRegression(C=best_c, class_weight=best_cw, max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    print(f"Logistic refit on full train: C={best_c}, class_weight={best_cw}")

    from sklearn.naive_bayes import GaussianNB

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Calibrate the KNN (winner) on a held-out slice
    X_cal, _, y_cal, _ = train_test_split(
        X_train, y_train, train_size=20000, random_state=RANDOM_STATE, stratify=y_train
    )
    knn_cal = calibrate_isotonic(knn, X_cal, y_cal)

    # Evaluate all on the test set
    rows = []
    for name, m in [("KNN (tuned)", knn), ("Logistic (tuned)", lr), ("Naive Bayes", nb)]:
        proba = m.predict_proba(X_test)[:, 1]
        d = metrics_dict(y_test, m.predict(X_test), proba)
        d["model"] = name
        rows.append(d)
    rows.append(
        {"model": "KNN (tuned + calibrated)", **metrics_dict(y_test, knn_cal.predict(X_test), knn_cal.predict_proba(X_test)[:, 1])}
    )
    scores = pd.DataFrame(rows)[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    print("\n--- Test set comparison ---")
    print(scores.to_string(index=False))
    save_metrics(scores, "tuned_metrics.csv")

    # Calibration quality: brier score before/after
    raw_brier = brier_score_loss(y_test, knn.predict_proba(X_test)[:, 1])
    cal_brier = brier_score_loss(y_test, knn_cal.predict_proba(X_test)[:, 1])
    print(f"\nBrier (raw KNN):     {raw_brier:.4f}")
    print(f"Brier (calibrated):  {cal_brier:.4f}")

    # Persist the best model + the full transform pipeline
    best_model = knn_cal
    save_bundle(
        {
            "model": best_model,
            "scaler": prepared["scaler"],
            "feature_columns": prepared["feature_names"],
            "categorical": ["protocol_type", "service", "flag"],
            "metrics": scores.to_dict("records"),
        }
    )
    print("\n[persist] best model bundle saved to models/")

    # Comparison figure
    fig, ax = plt.subplots(figsize=(10, 6))
    scores.set_index("model")[["f1", "roc_auc", "recall"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title("Phase 1 - tuned models on NSL-KDD test")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "phase1_tuned_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()