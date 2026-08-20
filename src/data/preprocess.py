"""Preprocess NSL-KDD: clean, encode, scale, split."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CATEGORICAL = ["protocol_type", "service", "flag"]


def parse_label(label: pd.Series) -> pd.Series:
    """Map 4 attack families (and normal) to binary normal vs attack."""
    return label.apply(lambda x: 0 if x == "normal" else 1)


ATTACK_FAMILY = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "mailbomb": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
    "worm": "U2R",
}

UNKNOWN_FAMILY = "other"


def family_label(label: pd.Series) -> pd.Series:
    """Map each attack label to its family (DoS/Probe/R2L/U2R)."""
    return label.apply(lambda x: "normal" if x == "normal" else ATTACK_FAMILY.get(x, UNKNOWN_FAMILY))


def preprocess(train: pd.DataFrame, test: pd.DataFrame, binary_target: bool = True) -> dict:
    """Return dict with scaled feature matrices and encoded targets."""
    y_train = parse_label(train["label"]) if binary_target else train["label"]
    y_test = parse_label(test["label"]) if binary_target else test["label"]

    X_train = train.drop(columns=["label"])
    X_test = test.drop(columns=["label"])

    # One-hot encode categorical features (fitted on train only)
    X_all = pd.concat([X_train, X_test], axis=0)
    X_all = pd.get_dummies(X_all, columns=CATEGORICAL)

    X_train_enc = X_all.iloc[: len(X_train)]
    X_test_enc = X_all.iloc[len(X_train):]

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "feature_names": list(X_train_enc.columns),
        "scaler": scaler,
        "train_df": train,
        "test_df": test,
    }