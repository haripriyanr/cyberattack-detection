"""Plotting helpers: confusion matrices, model comparison bars, reports."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def save_confusion_matrix(model, X_test, y_test, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = FIGURES_DIR / f"confusion_{name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[viz] Saved {path}")


def save_comparison_plot(scores) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores.set_index("model")[metrics].plot(kind="bar", figsize=(10, 6))
    plt.title("Model comparison - NSL-KDD cyberattack detection")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[viz] Saved {path}")


def save_report(models: dict, X_test, y_test, scores) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(METRICS_DIR / "metrics.csv", index=False)
    report = {}
    for name, model in models.items():
        report[name] = classification_report(
            y_test, model.predict(X_test), output_dict=True
        )
    with open(METRICS_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[viz] Saved metrics.csv and classification_report.json")