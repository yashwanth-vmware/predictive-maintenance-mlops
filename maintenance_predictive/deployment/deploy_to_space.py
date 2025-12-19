"""
Deploy Docker-based Hugging Face Space files to a Hugging Face Space repo.

Fix:
- If Dockerfile/requirements.txt/app.py are not in DEPLOY_DIR, fallback to repo root.
- Uploads files to Space repo ROOT (required for Docker Spaces).
"""

import os
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, create_repo

try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    class HfHubHTTPError(Exception):
        pass


DEFAULT_SPACE_ID = "Yashwanthsairam/predictive-maintenance-mlops"
DEFAULT_DEPLOY_DIR = "maintenance_predictive/deployment"


def get_hf_token() -> str:
    token = (os.getenv("HF_TOKEN") or "").strip()
    if token:
        return token

    # Colab fallback
    try:
        from google.colab import userdata
        token = (userdata.get("HF_TOKEN") or "").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            return token
    except Exception:
        pass

    raise EnvironmentError(
        "❌ HF_TOKEN not found. Set it via GitHub Secrets (HF_TOKEN) or locally."
    )


def get_space_id() -> str:
    space_id = (os.getenv("HF_SPACE_ID") or "").strip()
    return space_id if space_id else DEFAULT_SPACE_ID


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


def resolve_local_file(deploy_dir: Path, repo_root: Path, filename: str) -> Path:
    """
    Prefer deploy_dir/filename. If missing, fallback to repo_root/filename.
    """
    p1 = deploy_dir / filename
    if p1.exists():
        return p1

    p2 = repo_root / filename
    if p2.exists():
        return p2

    raise FileNotFoundError(
        "❌ Missing required file: '{f}'\nChecked:\n- {a}\n- {b}".format(
            f=filename, a=str(p1), b=str(p2)
        )
    )


def upload_files(api: HfApi, space_id: str, deploy_dir: Path, files: List[Tuple[str, str]]) -> None:
    """
    Upload to Space ROOT. Local files may come from deploy_dir or repo root (fallback).
    """
    # repo_root = .../predictive-maintenance-mlops (3 parents above this script folder)
    repo_root = Path(__file__).resolve().parents[2]  # maintenance_predictive
    repo_root = repo_root.parent                     # repo root

    for local_name, path_in_repo in files:
        local_fp = resolve_local_file(deploy_dir, repo_root, local_name)

        print(f"📤 Uploading {local_fp} → {space_id}/{path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(local_fp),
            path_in_repo=path_in_repo,
            repo_id=space_id,
            repo_type="space",
        )


def main() -> None:
    token = get_hf_token()
    api = HfApi(token=token)

    space_id = get_space_id()
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

    print("✅ Deployment completed. Check Space build logs in Hugging Face.")


if __name__ == "__main__":
    main()
