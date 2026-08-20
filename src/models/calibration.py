"""Isotonic calibration wrapper for an already-fitted classifier.

sklearn's CalibratedClassifierCV refits the estimator, which wastes the full
training fit. This wrapper trains a single IsotonicRegression on a held-out
slice and wraps a prefit base model, so the base keeps its full training fit.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self, base, iso: IsotonicRegression):
        self.base = base
        self.iso = iso
        self.classes_ = base.classes_

    def predict_proba(self, X) -> np.ndarray:
        p = self.iso.transform(self.base.predict_proba(X)[:, 1])
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_isotonic(base, X_cal, y_cal) -> IsotonicCalibrator:
    """Calibrate a prefit model's probabilities on a held-out set."""
    proba_cal = base.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_cal, y_cal)
    return IsotonicCalibrator(base, iso)