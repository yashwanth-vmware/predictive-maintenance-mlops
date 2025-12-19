from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Register data inside your Hugging Face SPACE repo (Files & Versions).
HF_SPACE_REPO_ID = "Yashwanthsairam/predictive-maintenance-mlops"
REPO_TYPE = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Ensure the Space exists (no-op if it already exists)
try:
    api.repo_info(repo_id=HF_SPACE_REPO_ID, repo_type=REPO_TYPE)
    print(f"✅ Space exists: {HF_SPACE_REPO_ID}")
except RepositoryNotFoundError:
    print(f"⚠️ Space not found. Creating: {HF_SPACE_REPO_ID}")
    create_repo(repo_id=HF_SPACE_REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"✅ Space created: {HF_SPACE_REPO_ID}")

# Upload local data folder into the Space repo under /data
api.upload_folder(
    folder_path="maintenance_predictive/data",
    repo_id=HF_SPACE_REPO_ID,
    repo_type=REPO_TYPE,
    path_in_repo="data",
)

print("✅ Uploaded dataset files to the Space under /data")
