# Project Plan — Cyberattack Detection

## 1. Objective

Build a reproducible network intrusion detection pipeline on NSL-KDD using
supervised and unsupervised methods. The pipeline covers data ingestion,
preprocessing, model training, evaluation, and single-record inference, with
experiments that test generalization to unseen attack families and to a modern
benchmark dataset.

## 2. Dataset

- **Primary:** NSL-KDD — 125,973 training and 22,544 test records, 41 features
  plus label and difficulty. Test set contains attack types absent from training.
- **Cross-check:** UNSW-NB15 — 175,341 training and 82,332 test records, 49
  features. Used to validate that the approach is not specific to NSL-KDD.

Raw files are not committed. `src/data/load.py` downloads them on first run;
processed files are written to `data/processed/`.

## 3. Baseline

Binary classification (normal vs. attack) with three classifiers from the
workshop:

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes

Pipeline entry point: `python main.py` — loads data, one-hot encodes
categorical features, standardizes numerics, trains all three models, and
writes metrics and figures to `reports/`.

## 4. Known Limitations

1. Default hyperparameters — no tuning or calibration.
2. Binary target only — does not distinguish attack families.
3. No isolated evaluation on unseen attack families.
4. Rare families (R2L, U2R) are underrepresented in training (~1%).
5. No unsupervised baseline for label-free detection.
6. No persisted artifacts for inference or deployment.

## 5. Roadmap

### Phase 1 — Hyperparameter Tuning and Calibration

- **Objective:** Select hyperparameters on a held-out validation split and
  calibrate probability estimates.
- **Approach:** Grid search over KNN (k, weighting) and Logistic Regression
  (C, class weight) on a 30k subsample; refit best configuration on the full
  training set. Isotonic calibration of the best KNN on a held-out slice.
- **Status:** Complete. KNN (k=3, distance, isotonic) achieves F1 0.787 on
  test; calibration adds +0.025 F1.
- **Artifacts:** `experiments/tune_models.py`, `reports/metrics/tuned_metrics.csv`,
  `reports/figures/phase1_tuned_comparison.png`.

### Phase 2 — Multi-Class Attack Family Classification

- **Objective:** Classify five categories: normal, DoS, Probe, R2L, U2R.
- **Approach:** Map raw labels to families, train KNN, Logistic Regression, and
  Naive Bayes, report per-class precision and recall.
- **Status:** Complete. Logistic Regression (balanced) achieves macro-F1 0.484;
  R2L recall 0.18 and U2R recall 0.48 reflect data scarcity.
- **Artifacts:** `experiments/multiclass.py`, `reports/metrics/multiclass_metrics.csv`.

### Phase 3 — Zero-Day Generalization

- **Objective:** Measure detection of attack families not seen during training.
- **Approach:** Train on normal + DoS + Probe only; evaluate separately on seen
  families versus held-out R2L/U2R. Compare against models trained on all
  families.
- **Status:** Complete. Models trained on seen families achieve 82–84% recall
  on seen attacks and 1–2% on unseen attacks.
- **Artifacts:** `experiments/zero_day.py`, `reports/metrics/zero_day_metrics.csv`.

### Phase 4 — Unsupervised Anomaly Detection

- **Objective:** Provide a label-free baseline that does not rely on attack
  signatures.
- **Approach:** Isolation Forest and One-Class SVM trained on normal-only
  traffic. Evaluate with ROC-AUC and threshold-tuned F1.
- **Status:** Complete. One-Class SVM achieves F1 0.893 and ROC-AUC 0.935;
  Isolation Forest achieves F1 0.880 and ROC-AUC 0.937.
- **Artifacts:** `experiments/anomaly.py`, `reports/metrics/anomaly_metrics.csv`.

### Phase 5 — Persistence and Inference

- **Objective:** Persist the best model and enable single-record scoring.
- **Approach:** Save calibrated KNN, scaler, and feature schema with joblib;
  provide `predict.py` for single-connection inference. Include cost-sensitive
  threshold analysis, SHAP feature importance, Streamlit demo, and UNSW-NB15
  cross-check as supporting analyses.
- **Status:** Complete.
  - `models/best_model.joblib` and `predict.py` — calibrated KNN scoring
  - `experiments/cost_sensitive.py` — threshold optimized for 10:1 FN:FP cost
  - `experiments/shap_importance.py` — SHAP top-20 features
  - `app.py` — Streamlit interface
  - `experiments/unsw_crosscheck.py` — UNSW-NB15 validation (Isolation Forest
    ROC-AUC 0.795, F1 0.777)
- **Artifacts:** `reports/metrics/cost_sensitive.csv`, `reports/metrics/shap_top20.csv`,
  `reports/metrics/unsw_crosscheck.csv`.

## 6. Out of Scope

- Deep learning on tabular features.
- Real-time packet capture and streaming inference.
- Optimization for inflated accuracy on leaked features.

## 7. Future Work

- Additional modern benchmarks (e.g., CICIDS2017).
- Formal model monitoring and drift detection.
- Containerized deployment for the Streamlit demo.
