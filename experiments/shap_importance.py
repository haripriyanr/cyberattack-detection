"""SHAP feature importance for the best calibrated model.

Uses KernelExplainer on a small background + sample for speed - this is
tabular data with 122 features, so we keep the sample small.

   python experiments/shap_importance.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.load import load_data
from src.data.preprocess import preprocess
from src.models.persist import load_bundle
from common import save_fig, save_metrics


def main() -> None:
    print("=" * 60)
    print("SHAP feature importance")
    print("=" * 60)

    try:
        import shap
    except ImportError:
        print("shap not installed. Run: pip install shap")
        sys.exit(1)

    train = load_data(test=False)
    test = load_data(test=True)
    prepared = preprocess(train, test, binary_target=True)
    X_test = prepared["X_test"]
    feature_names = prepared["feature_names"]
    bundle = load_bundle()
    model = bundle["model"].base if hasattr(bundle["model"], "base") else bundle["model"]

    # Small samples for speed
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_test), min(200, len(X_test)), replace=False)
    sample_idx = rng.choice(len(X_test), min(300, len(X_test)), replace=False)
    background = X_test[bg_idx]
    sample = X_test[sample_idx]

    # KNN has no TreeExplainer - use KernelExplainer (model-agnostic)
    # Wrap predict_proba for SHAP
    def f(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.KernelExplainer(f, background[:100])
    shap_values = explainer.shap_values(sample[:150], nsamples=100)

    # shap_values is 1D for binary
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    top_names = [feature_names[i] for i in order]
    top_vals = mean_abs[order]

    print("\nTop 20 features by mean |SHAP|:")
    for n, v in zip(top_names, top_vals):
        print(f"  {n:35s} {v:.4f}")

    save_metrics(
        pd.DataFrame({"feature": top_names, "mean_abs_shap": np.round(top_vals, 4)}),
        "shap_top20.csv",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names[::-1], top_vals[::-1], color="steelblue")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top 20 features - SHAP (KNN calibrated, NSL-KDD test)")
    fig.tight_layout()
    save_fig(fig, "shap_top20.png")
    plt.close(fig)

    # Beeswarm-style summary as bar (KernelExplainer summary plot is slow)
    shap.summary_plot(shap_values, sample[:150], feature_names=feature_names, show=False, max_display=12)
    fig = plt.gcf()
    fig.tight_layout()
    save_fig(fig, "shap_summary.png")
    plt.close(fig)


if __name__ == "__main__":
    main()