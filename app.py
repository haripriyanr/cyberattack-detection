"""Streamlit demo for the cyberattack detector.

   streamlit run app.py
"""

import streamlit as st
import pandas as pd

from src.data.load import COLUMNS
from src.models.persist import load_bundle

FEATURES = [c for c in COLUMNS if c not in ("label", "difficulty")]
CATEGORICAL = ["protocol_type", "service", "flag"]

st.set_page_config(page_title="Cyberattack Detector", layout="wide")
st.title("Cyberattack Detector - NSL-KDD")
st.caption("Calibrated KNN trained on NSL-KDD. Paste a connection's 41 features or tweak the example.")

try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Model not found. Run `python experiments/tune_models.py` first. ({e})")
    st.stop()

# Example rows
EXAMPLE_NORMAL = "0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.10,0.00,0.00,0.00,0.00,0.00"
EXAMPLE_ATTACK = "0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,33,1.00,1.00,0.00,0.00,0.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00"

col1, col2 = st.columns(2)
with col1:
    if st.button("Load normal example"):
        st.session_state["raw"] = EXAMPLE_NORMAL
with col2:
    if st.button("Load attack example"):
        st.session_state["raw"] = EXAMPLE_ATTACK

raw = st.text_area("41 comma-separated features", value=st.session_state.get("raw", EXAMPLE_NORMAL), height=100)


def coerce(v):
    try:
        return float(v)
    except ValueError:
        return v


if st.button("Score"):
    vals = [c.strip() for c in raw.split(",")]
    if len(vals) != len(FEATURES):
        st.error(f"Expected {len(FEATURES)} fields, got {len(vals)}.")
    else:
        row = [coerce(v) for v in vals]
        df = pd.DataFrame([row], columns=FEATURES)
        X = pd.get_dummies(df, columns=bundle["categorical"])
        X = X.reindex(columns=bundle["feature_columns"], fill_value=0)
        X = bundle["scaler"].transform(X)
        model = bundle["model"]
        proba = float(model.predict_proba(X)[0, 1]) if model.classes_[1] == 1 else float(model.predict_proba(X)[0, 0])
        verdict = "attack" if proba >= 0.5 else "normal"
        st.metric("Verdict", verdict, f"{proba:.1%} attack probability")
        st.progress(proba)
        if verdict == "attack":
            st.warning("Flagged as attack - would be forwarded to SOC review.")
        else:
            st.success("Looks normal.")

st.divider()
st.caption("Model: KNN (k=3, distance, isotonic-calibrated) | Features: 41 raw + one-hot -> 122 | Test F1=0.787")
