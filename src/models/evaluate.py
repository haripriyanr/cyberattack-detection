"""Evaluate classifiers: core metrics for every model."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    """Compute core metrics for every model; return a comparison table."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, proba),
            }
        )
        print(f"[eval] {name}: f1={rows[-1]['f1']:.4f} auc={rows[-1]['roc_auc']:.4f}")
    return pd.DataFrame(rows)