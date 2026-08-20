"""Phase 3 - zero-day generalization.

Train a detector on a subset of attack families, then measure how it does on
families it never saw. This mimics the real IDS problem: an attack you have no
labels for shows up anyway.

   python experiments/zero_day.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from src.data.load import load_data
from src.data.preprocess import preprocess, parse_label, family_label
from common import save_metrics, save_fig

RANDOM_STATE = 42
SEEN = ["DoS", "Probe"]
UNSEEN = ["R2L", "U2R"]
K = 9
WEIGHTS = "distance"


def recall_on_family(y_pred, fam_mask):
    """Recall among test rows of a given family subset (all are attacks)."""
    n = fam_mask.sum()
    if n == 0:
        return np.nan
    return round(float(y_pred[fam_mask].mean()), 4)


def main() -> None:
    print("=" * 60)
    print("Phase 3 - zero-day generalization experiment")
    print("=" * 60)

    train = load_data(test=False)
    test = load_data(test=True)
    prepared = preprocess(train, test, binary_target=True)
    X_train, X_test = prepared["X_train"], prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    train_fam = family_label(train["label"])
    test_fam = family_label(test["label"])

    # Training subset: normal + the "known" attack families only
    keep = (train_fam == "normal") | train_fam.isin(SEEN)
    X_restricted = X_train[keep]
    y_restricted = y_train[keep]
    print(f"Restricted training rows: {X_restricted.shape[0]} "
          f"(normal + {SEEN}); dropped {len(train) - X_restricted.shape[0]} "
          f"rows of {UNSEEN}")

    seen_mask = test_fam.isin(SEEN)
    unseen_mask = test_fam.isin(UNSEEN)
    normal_mask = test_fam == "normal"
    print(f"Test composition - normal: {normal_mask.sum()}, "
          f"seen ({SEEN}): {seen_mask.sum()}, unseen ({UNSEEN}): {unseen_mask.sum()}")

    models = {
        "Logistic (restricted)": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "KNN (restricted)": KNeighborsClassifier(
            n_neighbors=K, weights=WEIGHTS, n_jobs=-1
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_restricted, y_restricted)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        row = {
            "model": name,
            "f1": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "seen_family_recall": recall_on_family(y_pred, seen_mask),
            "zero_day_recall": recall_on_family(y_pred, unseen_mask),
            "normal_fp_rate": round(float(y_pred[normal_mask].mean()), 4),
        }
        rows.append(row)
        print(f"\n{name}:")
        print(f"  seen-family recall = {row['seen_family_recall']}")
        print(f"  zero-day recall    = {row['zero_day_recall']}")
        print(f"  normal FP rate     = {row['normal_fp_rate']}")

    # Reference: same models trained on ALL families
    print("\n--- Reference (trained on everything) ---")
    for name, model in [
        ("Logistic (full)", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        )),
        ("KNN (full)", KNeighborsClassifier(n_neighbors=K, weights=WEIGHTS, n_jobs=-1)),
    ]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        row = {
            "model": name,
            "f1": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4),
            "seen_family_recall": recall_on_family(y_pred, seen_mask),
            "zero_day_recall": recall_on_family(y_pred, unseen_mask),
            "normal_fp_rate": round(float(y_pred[normal_mask].mean()), 4),
        }
        rows.append(row)
        print(f"{name}: zero-day recall = {row['zero_day_recall']}")

    scores = pd.DataFrame(rows)
    print("\n--- Zero-day experiment summary ---")
    print(scores.to_string(index=False))
    save_metrics(scores, "zero_day_metrics.csv")

    # Bar chart: seen vs zero-day recall, restricted vs full
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(scores))
    width = 0.35
    ax.bar(x - width / 2, scores["seen_family_recall"], width, label="seen families", color="steelblue")
    ax.bar(x + width / 2, scores["zero_day_recall"], width, label="zero-day (unseen)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(scores["model"], rotation=10)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 3 - detection of seen vs zero-day attack families")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "phase3_zero_day.png")
    plt.close(fig)


if __name__ == "__main__":
    main()