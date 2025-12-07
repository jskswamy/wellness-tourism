"""
Register Dataset to Hugging Face Hub

This script uploads the tourism dataset to Hugging Face Dataset Hub.
It uploads the raw CSV file directly, avoiding schema conflicts with
processed train/test splits that may exist in the same repo.
"""

import os
import argparse
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


def register_dataset(
    data_path: str,
    repo_id: str,
    hf_token: str = None,
    private: bool = False
):
    """
    Upload dataset to Hugging Face Hub as a raw CSV file.

    This function uploads the raw CSV file directly to avoid schema conflicts
    with the datasets library. The prep.py script will then load this CSV
    and create proper train/test splits.

    Args:
        data_path: Path to the CSV file
        repo_id: Hugging Face repo ID (e.g., 'username/dataset-name')
        hf_token: Hugging Face API token
        private: Whether to make the dataset private
    """
    # Authenticate with Hugging Face
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("Warning: No HF token provided. Using cached credentials.")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Dataset repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating dataset repo: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="dataset", private=private)

    # Upload CSV file directly (avoids schema conflicts with datasets library)
    print(f"Uploading {data_path} to {repo_id}")
    api.upload_file(
        path_or_fileobj=data_path,
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload raw tourism dataset"
    )

    print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Register dataset to Hugging Face Hub")
    parser.add_argument(
        "--data-path",
        type=str,
        default="tourism.csv",
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/wellness-tourism-data')"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )

    args = parser.parse_args()

    register_dataset(
        data_path=args.data_path,
        repo_id=args.repo_id,
        hf_token=args.hf_token,
        private=args.private
    )


if __name__ == "__main__":
    main()
