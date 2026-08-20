"""Smoke test: the pipeline wiring is correct and runnable on a tiny slice."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.data.load import COLUMNS
from src.data.preprocess import parse_label, family_label
from src.models.classifiers import MODEL_FACTORIES


def test_columns_count():
    # 41 features + label + difficulty
    assert len(COLUMNS) == 43


def test_parse_label():
    s = pd.Series(["normal", "neptune", "satan"])
    assert list(parse_label(s)) == [0, 1, 1]


def test_family_label():
    s = pd.Series(["normal", "neptune", "guess_passwd", "buffer_overflow"])
    assert list(family_label(s)) == ["normal", "DoS", "R2L", "U2R"]


def test_models_exist():
    assert set(MODEL_FACTORIES) == {
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
    }


if __name__ == "__main__":
    test_columns_count()
    test_parse_label()
    test_family_label()
    test_models_exist()
    print("All smoke tests passed.")