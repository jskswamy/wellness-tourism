"""
Hugging Face Spaces Deployment Module

This module handles the deployment of the Streamlit prediction application
to Hugging Face Spaces. It automates the process of creating the Space
repository (if needed) and uploading the deployment artifacts.

The deployment uses Docker SDK for containerized execution on HF Spaces.
"""

import os
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Configuration from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
SPACE_NAME = os.environ.get("HF_SPACE_NAME", "wellness-tourism-predictor")

REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"


def main():
    """
    Deploy Streamlit application to Hugging Face Spaces.

    This function performs the following steps:
        1. Authenticates with Hugging Face Hub using HF_TOKEN
        2. Creates the Space repository if it doesn't exist (Docker SDK)
        3. Sets the HF_USERNAME environment variable on the Space
        4. Uploads the deployment/ folder contents to the Space

    Environment Variables:
        HF_TOKEN: Hugging Face API token with write access (required)
        HF_USERNAME: Hugging Face username for model repository lookup
        HF_SPACE_NAME: Name for the Space (default: wellness-tourism-predictor)

    Raises:
        ValueError: If HF_TOKEN environment variable is not set

    Returns:
        None. Prints deployment status and final app URL.
    """
    # Authenticate with Hugging Face Hub
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        raise ValueError("HF_TOKEN environment variable is required")

    api = HfApi()

    # Check if space exists, create if not
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="space")
        print(f"Space '{REPO_ID}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating space '{REPO_ID}'...")
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            private=False
        )
        print(f"Space '{REPO_ID}' created.")

    # Set HF_USERNAME as a space variable for model repository lookup
    try:
        api.add_space_variable(
            repo_id=REPO_ID,
            key="HF_USERNAME",
            value=HF_USERNAME
        )
        print(f"Set HF_USERNAME variable on space")
    except Exception as e:
        print(f"Note: Could not set space variable: {e}")

    # Upload deployment folder to the Space
    print(f"Uploading deployment folder to {REPO_ID}...")
    api.upload_folder(
        folder_path="deployment",
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo="",
        commit_message="Deploy Streamlit app"
    )

    print(f"\nDeployed successfully!")
    print(f"App URL: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
