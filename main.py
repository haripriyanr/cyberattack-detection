"""Run the full cyberattack detection pipeline.

   python main.py
"""

from src.data.load import load_data, save_processed
from src.data.preprocess import preprocess
from src.models.train import train_all
from src.models.evaluate import evaluate_all
from src.viz.plots import (
    save_confusion_matrix,
    save_comparison_plot,
    save_report,
)


def main() -> None:
    print("=" * 60)
    print("Cyberattack Detection - NSL-KDD Pipeline")
    print("=" * 60)

    # 1. Load data
    train = load_data(test=False)
    test = load_data(test=True)
    save_processed(train, test)
    print(f"[data] Train shape={train.shape} | Test shape={test.shape}")
    print(f"[data] Attack ratio in train: {(train['label'] != 'normal').mean():.2%}")

    # 2. Preprocess
    prepared = preprocess(train, test, binary_target=True)
    X_train, X_test = prepared["X_train"], prepared["X_test"]
    y_train, y_test = prepared["y_train"], prepared["y_test"]
    print(f"[prep] Features after encoding: {X_train.shape[1]}")

    # 3. Train
    models = train_all(X_train, y_train, n_neighbors=5)

    # 4. Evaluate
    scores = evaluate_all(models, X_test, y_test)
    print("\n--- Metric comparison ---")
    print(scores.to_string(index=False))

    for name, model in models.items():
        save_confusion_matrix(model, X_test, y_test, name)
    save_comparison_plot(scores)
    save_report(models, X_test, y_test, scores)

    print("\nDone. Reports in reports/ and processed data in data/processed/.")


if __name__ == "__main__":
    main()