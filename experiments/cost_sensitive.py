"""Cost-sensitive evaluation: FN (missed attack) costs 10x a FP.

   python experiments/cost_sensitive.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from src.data.load import load_data
from src.data.preprocess import preprocess
from src.models.persist import load_bundle
from common import save_metrics, save_fig

COST_FN = 10
COST_FP = 1


def cost_at_threshold(y_true, proba, thresh, cost_fn=COST_FN, cost_fp=COST_FP):
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return cost_fn * fn + cost_fp * fp, (tn, fp, fn, tp)


def main() -> None:
    print("=" * 60)
    print(f"Cost-sensitive evaluation (FN={COST_FN}, FP={COST_FP})")
    print("=" * 60)

    train = load_data(test=False)
    test = load_data(test=True)
    prepared = preprocess(train, test, binary_target=True)
    y_test = prepared["y_test"]

    bundle = load_bundle()
    model = bundle["model"]
    proba = model.predict_proba(prepared["X_test"])[:, 1]

    thresholds = np.linspace(0.05, 0.95, 181)
    costs, f1s = [], []
    from sklearn.metrics import f1_score

    for t in thresholds:
        c, _ = cost_at_threshold(y_test, proba, t)
        costs.append(c)
        f1s.append(f1_score(y_test, proba >= t))

    best_idx = int(np.argmin(costs))
    best_t = thresholds[best_idx]
    best_cost, (tn, fp, fn, tp) = cost_at_threshold(y_test, proba, best_t)
    default_cost, _ = cost_at_threshold(y_test, proba, 0.5)

    print(f"Default (0.5) total cost: {default_cost}")
    print(f"Best threshold {best_t:.2f} total cost: {best_cost}  (TN={tn} FP={fp} FN={fn} TP={tp})")
    print(f"Cost saving: {default_cost - best_cost} ({(1 - best_cost / max(default_cost, 1)) * 100:.1f}%)")

    rows = [
        {"threshold": 0.5, "total_cost": int(default_cost), "f1": round(float(f1s[np.argmin(np.abs(thresholds - 0.5))]), 4)},
        {"threshold": round(float(best_t), 3), "total_cost": int(best_cost), "f1": round(float(f1s[best_idx]), 4)},
    ]
    save_metrics(pd.DataFrame(rows), "cost_sensitive.csv")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, costs, color="coral", label="total cost")
    ax1.axvline(best_t, color="coral", linestyle="--")
    ax1.set_xlabel("threshold")
    ax1.set_ylabel("total cost", color="coral")
    ax2 = ax1.twinx()
    ax2.plot(thresholds, f1s, color="steelblue", label="F1")
    ax2.set_ylabel("F1", color="steelblue")
    ax1.set_title(f"Cost vs threshold (FN={COST_FN}, FP={COST_FP})")
    fig.tight_layout()
    save_fig(fig, "cost_sensitive.png")
    plt.close(fig)


if __name__ == "__main__":
    main()