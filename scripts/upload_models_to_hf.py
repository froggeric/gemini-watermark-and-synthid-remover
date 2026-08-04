#!/usr/bin/env python3
"""
Upload wmr models to HuggingFace.

This script uploads the CoreML SDXL models to the froggeric/wmr repository.
Run with: HF_TOKEN=... python scripts/upload_models_to_hf.py

Or authenticate first:
    huggingface-cli login
Then run without HF_TOKEN.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, HfFolder
except ImportError:
    print("Error: huggingface_hub not installed. Install it via:")
    print("  pip install 'huggingface_hub[cli]'")
    sys.exit(1)


def main():
    repo_id = "froggeric/wmr"
    upload_dir = Path.home() / ".cache" / "wmr" / "coreml-sdxl" / "_hf_upload"

    if not upload_dir.exists():
        print(f"Error: Upload directory not found: {upload_dir}")
        print("Ensure you've created the archives and README first.")
        sys.exit(1)

    # Get token from env or huggingface-cli cache
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = HfFolder.get_token()
    if not token:
        print("Error: No HF_TOKEN found. Set HF_TOKEN or run: huggingface-cli login")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        repo_info = api.repo_info(repo_id)
        print(f"Using existing repo: {repo_info}")
    except Exception:
        print(f"Creating repository: {repo_id}")
        api.create_repo(repo_id, repo_type="model", private=False)
        print(f"Created public repo: {repo_id}")

    # Files to upload
    files = [
        "README.md",
        "coreml-sdxl-unet.mlpackage.tar.gz",
        "coreml-sdxl-vae-encoder.mlpackage.tar.gz",
        "coreml-sdxl-vae-decoder.mlpackage.tar.gz",
        "empty_prompt_embeds.bin",
    ]

    for filename in files:
        filepath = upload_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found in {upload_dir}, skipping")
            continue

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"Uploading {filename} ({size_mb:.1f} MB)...")

        try:
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded: {filename}")
        except Exception as e:
            print(f"  Error uploading {filename}: {e}")
            sys.exit(1)

    print("\nUpload complete!")
    print(f"View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
