"""
Run this script from inside the openenv-support-env/ folder.
It will:
  1. Create the HF Space (if it doesn't exist)
  2. Upload all project files to the Space

Usage:
    cd openenv-support-env
    python ../push_to_hf.py
"""

import os
import sys

try:
    from huggingface_hub import HfApi, whoami
except Exception as e:
    print("REAL ERROR:", repr(e))
    raise

REPO_ID = "Arnav100904/customer-support-triage"
REPO_TYPE = "space"

api = HfApi()

# Check login
try:
    whoami()
except Exception:
    print("ERROR: Not logged in. Run: huggingface-cli login")
    sys.exit(1)

# Create space
print(f"Creating Space: {REPO_ID} ...")
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print("  Space ready.")
except Exception as e:
    print(f"  Space creation note: {e}")

# Upload all files
SKIP_DIRS = {"__pycache__", ".git", "dist", "build", ".venv", "venv", ".egg-info"}
SKIP_EXTS = {".pyc", ".pyo"}

uploaded = 0
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
    for fname in files:
        ext = os.path.splitext(fname)[1]
        if ext in SKIP_EXTS:
            continue
        local_path = os.path.join(root, fname)
        # path_in_repo: strip leading ./ or .\
        path_in_repo = local_path.replace("\\", "/").lstrip("./")
        print(f"  Uploading: {path_in_repo}")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            uploaded += 1
        except Exception as e:
            print(f"  WARNING: Could not upload {path_in_repo}: {e}")

print(f"\nDone! {uploaded} files uploaded.")
print(f"Space URL: https://huggingface.co/spaces/{REPO_ID}")
print(f"API URL:   https://arnav100904-customer-support-triage.hf.space")
print("\nThe Space will build in ~2-3 minutes. Watch the logs at the URL above.")
