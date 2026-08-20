"""Load the NSL-KDD network intrusion detection dataset."""

from pathlib import Path
import urllib.request
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

NSL_KDD_BASE = "https://github.com/HoaNP/NSL-KDD-DataSet/raw/master"

COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label", "difficulty",
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_file(file_name: str) -> Path:
    """Download one NSL-KDD .txt file if not present; return local path."""
    ensure_dirs()
    local_path = RAW_DIR / file_name
    if local_path.exists():
        return local_path
    url = f"{NSL_KDD_BASE}/{file_name}"
    print(f"[load] Downloading {file_name} from {url} ...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; cyberattack-detection)"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp, open(local_path, "wb") as f:
        f.write(resp.read())
    print("[load] Download complete.")
    return local_path


def load_data(test: bool = False) -> pd.DataFrame:
    """Load train (or test) NSL-KDD data as a DataFrame."""
    file_name = "KDDTest+.txt" if test else "KDDTrain+.txt"
    path = download_file(file_name)
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    return df


def save_processed(train: pd.DataFrame, test: pd.DataFrame) -> None:
    ensure_dirs()
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"[load] Processed data saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    print(load_data(test=False).head())