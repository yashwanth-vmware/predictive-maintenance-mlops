"""
Deploy Docker-based Hugging Face Space files to a Hugging Face Space repo.

Option 1 design:
- Single Space repo is the source of truth
- Dockerfile, requirements.txt, app.py are uploaded to repo ROOT
- Works in Colab, local Python, and GitHub Actions

Required:
- HF_TOKEN (Colab Secrets or env var)

Optional:
- HF_SPACE_ID (env var). Defaults to Yashwanthsairam/predictive-maintenance-mlops
"""

import os
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, create_repo

# Hub compatibility (older versions)
try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    class HfHubHTTPError(Exception):
        pass


# ----------------------------
# Defaults (Option 1)
# ----------------------------
DEFAULT_SPACE_ID = "Yashwanthsairam/predictive-maintenance-mlops"
DEFAULT_DEPLOY_DIR = "maintenance_predictive/deployment"


# ----------------------------
# Token handling (Option A)
# ----------------------------
def get_hf_token() -> str:
    # 1) env var
    token = (os.getenv("HF_TOKEN") or "").strip()
    if token:
        return token

    # 2) Colab Secrets fallback
    try:
        from google.colab import userdata
        token = (userdata.get("HF_TOKEN") or "").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            return token
    except Exception:
        pass

    raise EnvironmentError(
        "❌ HF_TOKEN not found. Add it as:\n"
        "- Colab: Secrets → HF_TOKEN\n"
        "- GitHub Actions: secrets.HF_TOKEN\n"
        "- Local: export HF_TOKEN=..."
    )


def get_space_id() -> str:
    """
    GitHub Actions vars.HF_SPACE_ID may exist but be empty.
    Always fallback safely.
    """
    space_id = (os.getenv("HF_SPACE_ID") or "").strip()
    return space_id if space_id else DEFAULT_SPACE_ID


# ----------------------------
# Space validation
# ----------------------------
def ensure_space_exists(api: HfApi, space_id: str, token: str) -> None:
    try:
        api.repo_info(repo_id=space_id, repo_type="space")
        print(f"✅ Space exists: {space_id}")
        return
    except Exception:
        print(f"ℹ️ Space not found. Creating: {space_id}")

    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            private=False,
            token=token,
            exist_ok=True,
        )
        print(f"✅ Created Space: {space_id}")
    except HfHubHTTPError as e:
        raise RuntimeError(f"❌ Failed to create/access Space '{space_id}': {e}") from e


# ----------------------------
# Upload files to Space root
# ----------------------------
def upload_files(
    api: HfApi,
    space_id: str,
    deploy_dir: Path,
    files: List[Tuple[str, str]],
) -> None:
    """
    Upload local deployment files to Space repo root.
    Docker Spaces REQUIRE Dockerfile at repo root.
    """
    for local_name, repo_path in files:
        local_fp = deploy_dir / local_name
        if not local_fp.exists():
            raise FileNotFoundError(f"❌ Missing required file: {local_fp}")

        print(f"📤 Uploading {local_fp} → {space_id}/{repo_path}")
        api.upload_file(
            path_or_fileobj=str(local_fp),
            path_in_repo=repo_path,
            repo_id=space_id,
            repo_type="space",
        )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    token = get_hf_token()
    api = HfApi(token=token)

    space_id = get_space_id()
    if not space_id:
        raise ValueError("❌ Resolved HF_SPACE_ID is empty.")

    deploy_dir = Path(os.getenv("DEPLOY_DIR", DEFAULT_DEPLOY_DIR))

    files_to_upload = [
        ("Dockerfile", "Dockerfile"),
        ("requirements.txt", "requirements.txt"),
        ("app.py", "app.py"),
    ]

    print("========== DEPLOY TO HF SPACE ==========")
    print(f"Space ID : {space_id}")
    print(f"Deploy   : {deploy_dir.resolve()}")
    print("========================================")

    ensure_space_exists(api, space_id, token)
    upload_files(api, space_id, deploy_dir, files_to_upload)

    print("✅ Deployment completed.")
    print("➡️ Open the Space UI and wait for the Docker build to finish.")


if __name__ == "__main__":
    main()
