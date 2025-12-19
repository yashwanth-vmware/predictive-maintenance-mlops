# Data Preparation script for Predictive Maintenance (Engine Health)
# Option 1 (BEST for your setup): Everything stays in the HF SPACE repo (data + splits),
# and data is LOADED via Hugging Face SPACE URLs.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ----------------------------
# Hugging Face Space (single source of truth)
# ----------------------------
HF_SPACE_REPO_ID = "Yashwanthsairam/predictive-maintenance-mlops"
BASE_URL = "https://huggingface.co/spaces/{repo}/resolve/main".format(repo=HF_SPACE_REPO_ID)

DATA_URL  = "{base}/data/engine_data.csv".format(base=BASE_URL)     # raw dataset in Space
TRAIN_IN_REPO = "splits/train_engine_data.csv"                      # where we upload train
TEST_IN_REPO  = "splits/test_engine_data.csv"                       # where we upload test

# Token required for upload (read can work without token if public)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise EnvironmentError(
        "❌ HF_TOKEN is not set. Add HF_TOKEN in Colab Secrets (or env vars) and restart runtime."
    )

api = HfApi(token=HF_TOKEN)

# ----------------------------
# Local paths (temporary)
# ----------------------------
PROJECT_DIR = os.path.join(os.getcwd(), "maintenance_predictive")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train_engine_data.csv")
test_path  = os.path.join(DATA_DIR, "test_engine_data.csv")

# ----------------------------
# Load dataset from HF Space URL
# ----------------------------
df = pd.read_csv(DATA_URL)
print("✅ Dataset loaded from HF Space URL:", df.shape)

# ----------------------------
# Standardize column names
# ----------------------------
df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

# ----------------------------
# Target column
# ----------------------------
target_col = "Engine_Condition"
if target_col not in df.columns:
    raise KeyError(
        f"❌ Target column '{target_col}' not found. Available columns: {list(df.columns)}"
    )

# ----------------------------
# Split (train/test includes target)
# ----------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() > 1 else None,  # safe if only 1 class by mistake
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("✅ Saved:", train_path, "shape:", train_df.shape)
print("✅ Saved:", test_path,  "shape:", test_df.shape)

# ----------------------------
# Upload splits back to HF Space repo under /splits
# ----------------------------
api.upload_file(
    path_or_fileobj=train_path,
    path_in_repo=TRAIN_IN_REPO,
    repo_id=HF_SPACE_REPO_ID,
    repo_type="space",
)
print("⬆️ Uploaded:", TRAIN_IN_REPO)

api.upload_file(
    path_or_fileobj=test_path,
    path_in_repo=TEST_IN_REPO,
    repo_id=HF_SPACE_REPO_ID,
    repo_type="space",
)
print("⬆️ Uploaded:", TEST_IN_REPO)

print("\n✅ Done. You can load them via URLs:")
print("{base}/{p}".format(base=BASE_URL, p=TRAIN_IN_REPO))
print("{base}/{p}".format(base=BASE_URL, p=TEST_IN_REPO))
