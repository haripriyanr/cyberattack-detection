"""Phase 2 - multi-class attack-family classification (5 classes).

   python experiments/multiclass.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
)

from src.data.load import load_data
from src.data.preprocess import preprocess, family_label
from common import save_metrics, save_fig

RANDOM_STATE = 42
K = 9  # from Phase 1 tuning
WEIGHTS = "distance"


def main() -> None:
    print("=" * 60)
    print("Phase 2 - multi-class attack family classification")
    print("=" * 60)

    train = load_data(test=False)
    test = load_data(test=True)

    # Features are scaled the same way; target becomes family labels
    prepared = preprocess(train, test, binary_target=False)
    X_train, X_test = prepared["X_train"], prepared["X_test"]

    y_train = family_label(pd.Series(prepared["y_train"]))
    y_test = family_label(pd.Series(prepared["y_test"]))
    print("Families in train:", sorted(y_train.unique()))
    print("Families in test: ", sorted(y_test.unique()))
    print("\nTrain family counts:\n", y_train.value_counts().to_string())

    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test]))
    y_tr = le.transform(y_train)
    y_te = le.transform(y_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=K, weights=WEIGHTS, n_jobs=-1),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Naive Bayes": GaussianNB(),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_tr)
        y_pred = model.predict(X_test)
        row = {
            "model": name,
            "accuracy": round(accuracy_score(y_te, y_pred), 4),
            "macro_f1": round(f1_score(y_te, y_pred, average="macro"), 4),
        }
        # per-family recall
        for fam in le.classes_:
            mask = y_te == le.transform([fam])[0]
            row[f"recall_{fam}"] = round(recall_score(y_te[mask], y_pred[mask], average="micro", zero_division=0), 4)
        results.append(row)
        print(f"\n--- {name} ---")
        print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))

    scores = pd.DataFrame(results)
    print("\n--- Multi-class comparison ---")
    print(scores.to_string(index=False))
    save_metrics(scores, "multiclass_metrics.csv")

    # Confusion matrix for the best model (KNN)
    best = models["KNN"]
    y_pred = best.predict(X_test)
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
    )
    ax.set_title("Multi-class confusion matrix - KNN (NSL-KDD test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    save_fig(fig, "phase2_multiclass_confusion.png")
    plt.close(fig)


if __name__ == "__main__":
    main()