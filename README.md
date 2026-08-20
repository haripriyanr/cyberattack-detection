# cyberattack-detection

[![ci](https://github.com/haripriyanr/cyberattack-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/haripriyanr/cyberattack-detection/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)

Detect network intrusions in NSL-KDD traffic using the classifiers from the
hands-on ML workshop: Logistic Regression, KNN and Naive Bayes.

The project goes further than a baseline: it tunes the models, classifies
attack *families*, tests the hard question (how well do we catch attacks the
model has never seen?), and throws in an unsupervised baseline that ends up
being the best detector of all.

## Results

All numbers are on the held-out test set (22,544 connections). NSL-KDD's test
set contains attack types absent from training, so scores sit lower than the
inflated numbers most tutorials report.

### Phase 1 - tuned binary detection

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| KNN (tuned k=3, distance + isotonic calibration) | 0.789 | 0.924 | 0.686 | **0.787** | 0.807 |
| KNN (tuned, uncalibrated) | 0.769 | 0.922 | 0.649 | 0.762 | 0.807 |
| Logistic Regression (C=10, balanced) | 0.756 | 0.917 | 0.627 | 0.745 | 0.779 |
| Naive Bayes | 0.555 | 0.980 | 0.222 | 0.362 | 0.799 |

Calibration is the real win here: same model, better decision boundary, F1 up
from 0.762 to 0.787. Naive Bayes still gets crushed on recall - its
independence assumption doesn't survive correlated traffic features.

### Phase 2 - attack family classification (5 classes)

| Model | Accuracy | Macro-F1 | Recall R2L | Recall U2R |
|-------|----------|----------|------------|------------|
| Logistic Regression (balanced) | **0.782** | **0.484** | 0.178 | 0.478 |
| KNN | 0.736 | 0.456 | 0.038 | 0.313 |
| Naive Bayes | 0.419 | 0.246 | 0.324 | 0.672 |

The story: R2L and U2R are nearly absent from training (995 and 52 rows out of
126k), so every model struggles to recall them. Overall accuracy hides this -
which is the point of reporting per-family recall.

### Phase 3 - zero-day generalization

Trained only on `normal + DoS + Probe`, then measured on the held-out families
R2L/U2R as if they were novel attacks.

| Model | Seen-family recall | **Zero-day recall** | Normal FP rate |
|-------|--------------------|--------------------|----------------|
| Logistic (restricted) | 0.836 | **0.012** | 0.075 |
| KNN (restricted) | 0.821 | **0.009** | 0.072 |
| Logistic (full training) | 0.807 | 0.018 | 0.075 |
| KNN (full training) | 0.821 | 0.056 | 0.072 |

The model catches 82-84% of the attacks it was trained on and **1-2% of the
attacks it has never seen**. This is the honest, uncomfortable answer to "can
an IDS detect novel attacks?" - and it's why real systems lean on anomaly
detection.

### Phase 4 - unsupervised anomaly detection

Trained on **normal traffic only** - no attack labels at train time.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| One-Class SVM (best-F1) | 0.877 | 0.885 | 0.902 | **0.893** | 0.935 |
| Isolation Forest (best-F1 threshold) | 0.861 | 0.868 | 0.892 | 0.880 | **0.937** |
| Isolation Forest (contamination cut) | 0.801 | 0.925 | 0.707 | 0.801 | 0.937 |

The unsupervised models trained on zero attack examples beat every supervised
classifier. One-Class SVM edges out Isolation Forest on F1; both show that
not memorizing attack signatures is how you generalize to novel attacks.

### Extra: cost-sensitive threshold

Missing an attack costs 10x a false alarm in a real SOC. Optimizing the
threshold for total cost instead of F1:

- Default threshold 0.50: total cost 41062
- Best threshold 0.05: total cost **40472** (saving 1.4%, 590 fewer cost units)

The low threshold is aggressive - it catches more attacks at the price of more
false alarms, which is exactly the right trade when FN is expensive.

### Extra: SHAP feature importance

SHAP (KernelExplainer on the calibrated KNN) shows which features drive
predictions. Top drivers on the test set: `flag_RSTR`, `service_ecr_i`,
`dst_host_srv_count`, `service_private`, `srv_serror_rate`. These are the
connection-error and service flags you'd expect a SOC analyst to look at.

Plots: `reports/figures/shap_top20.png` and `shap_summary.png`.

### Extra: UNSW-NB15 cross-check (modern dataset)

Same Isolation Forest approach on the 2015-era UNSW-NB15 (175k train / 82k
test, 49 raw features). No code changes to the method, just a different
dataset:

| Dataset | ROC-AUC | F1 | Precision | Recall |
|---------|---------|-----|-----------|--------|
| NSL-KDD | 0.937 | 0.880 | 0.868 | 0.892 |
| UNSW-NB15 | 0.795 | 0.777 | 0.728 | 0.833 |

The drop is real - modern traffic is harder - but the anomaly approach still
works, which is the point worth making in a review.

## Layout

```
├── main.py               # runs the baseline pipeline
├── predict.py            # score a single connection with the saved model
├── app.py                # Streamlit demo (streamlit run app.py)
├── experiments/          # tune, multiclass, zero-day, anomaly, SHAP, cost, UNSW
├── src/
│   ├── data/             # loading + preprocessing
│   ├── models/           # classifiers, training, evaluation, calibration, persist
│   └── viz/              # plotting + report export
├── notebooks/            # 01_eda.ipynb - exploration walkthrough
├── reports/
│   ├── figures/          # plots (generated, gitignored)
│   └── metrics/          # csv results (generated, gitignored)
├── data/                 # raw + processed (gitignored, see data/README.md)
├── models/               # saved artifacts (gitignored)
├── .github/workflows/    # CI
└── tests/                # smoke tests
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

python main.py                     # baseline comparison
python experiments/tune_models.py  # Phase 1
python experiments/multiclass.py   # Phase 2
python experiments/zero_day.py     # Phase 3
python experiments/anomaly.py      # Phase 4 (+ One-Class SVM)
python experiments/cost_sensitive.py  # cost-sensitive threshold
python experiments/shap_importance.py # SHAP top-20
python experiments/unsw_crosscheck.py # UNSW-NB15 (downloads ~45 MB)
```

First run downloads NSL-KDD (~21 MB total) into `data/raw/`. No manual setup.
UNSW-NB15 is downloaded separately by `unsw_crosscheck.py` (~45 MB).

Smoke tests and Streamlit demo:

```bash
.venv\Scripts\python tests\test_smoke.py
.venv\Scripts\streamlit run app.py   # opens http://localhost:8501
```

## Score one connection

`predict.py` reloads the saved, calibrated KNN model and classifies a single
connection given its 41 comma-separated features:

```bash
.venv\Scripts\python predict.py 0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.10,0.00,0.00,0.00,0.00,0.00
# verdict: normal  |  attack probability: 0.0000
```

## Why metrics beyond accuracy

NSL-KDD test has a different attack mix than train (that's intentional in this
dataset). Accuracy alone makes Naive Bayes look "fine" until you look at recall
and see it's catching one in five attacks. Every experiment in this repo
reports F1, per-family recall, or both, because that's where the interesting
failures live.

## Roadmap

See [PLAN.md](PLAN.md). All phases plus the extras (cost-sensitive, SHAP,
One-Class SVM, Streamlit demo, UNSW-NB15 cross-check) are implemented.

## Data

NSL-KDD is a cleaned re-release of the 1999 KDD Cup dataset. It removes the
redundant records that made the original biased. It's still old (2009) and
doesn't reflect modern traffic, which is a fair criticism to keep in mind.
See `data/README.md`.