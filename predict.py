"""Score a single network connection with the saved model.

   python predict.py 0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.10,0.00,0.00,0.00,0.00,0.00

The value at index 1 (protocol_type) can be any of tcp/udp/icmp. Numeric
fields must be numbers, categoricals must be the exact strings the model saw.
"""

import sys
import pandas as pd

from src.data.load import COLUMNS
from src.models.persist import load_bundle

FEATURES = [c for c in COLUMNS if c not in ("label", "difficulty")]


def coerce(value):
    try:
        return float(value)
    except ValueError:
        return value


def predict_row(values, bundle) -> tuple:
    """Return (verdict, attack_probability)."""
    df = pd.DataFrame([values], columns=FEATURES)
    X = pd.get_dummies(df, columns=bundle["categorical"])
    X = X.reindex(columns=bundle["feature_columns"], fill_value=0)
    X = bundle["scaler"].transform(X)

    model = bundle["model"]
    cls = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    attack_prob = float(proba[1]) if model.classes_[1] == 1 else float(proba[0])
    return ("attack" if cls == 1 else "normal"), attack_prob


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    raw = sys.argv[1].split(",")
    if len(raw) != len(FEATURES):
        print(f"Expected {len(FEATURES)} fields, got {len(raw)}.")
        print("Format:", ", ".join(FEATURES))
        sys.exit(1)

    values = [coerce(v.strip()) for v in raw]
    bundle = load_bundle()
    verdict, prob = predict_row(values, bundle)
    print(f"verdict: {verdict}")
    print(f"attack probability: {prob:.4f}")


if __name__ == "__main__":
    main()