"""
Data Preparation Script

This script handles data cleaning, preprocessing, and train/test split.
Uploads the prepared data back to Hugging Face Hub as CSV files.
"""

import os
import argparse
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import login


RANDOM_STATE = 42


def load_data_from_hf(repo_id: str) -> pd.DataFrame:
    """Load raw CSV dataset from Hugging Face Hub.

    The data_register.py script uploads the raw CSV file directly to the repo,
    so we load it using hf_hub_download instead of the datasets library.
    """
    from huggingface_hub import hf_hub_download

    print(f"Loading dataset from: {repo_id}")

    # Download the raw CSV file from HF Hub
    csv_path = hf_hub_download(
        repo_id=repo_id,
        filename="tourism.csv",
        repo_type="dataset"
    )

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")

    return df


def load_data_from_csv(data_path: str) -> pd.DataFrame:
    """Load dataset from local CSV file."""
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset.

    - Remove unnecessary columns
    - Handle missing values
    - Add engineered features
    """
    print("Cleaning data...")

    # Remove ID columns
    cols_to_drop = ["Unnamed: 0", "CustomerID"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    # Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != "ProdTaken"]
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Impute numerical with median
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed {col} with median: {median_val:.2f}")

    # Impute categorical with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Imputed {col} with mode: {mode_val}")

    # Feature engineering
    # Age groups
    def create_age_group(age):
        if age < 25:
            return "Young"
        elif age < 35:
            return "Young Adult"
        elif age < 45:
            return "Middle Age"
        elif age < 55:
            return "Senior"
        else:
            return "Elder"

    df["AgeGroup"] = df["Age"].apply(create_age_group)

    # Income groups
    def create_income_group(income):
        if income < 15000:
            return "Low"
        elif income < 25000:
            return "Medium"
        elif income < 35000:
            return "High"
        else:
            return "Very High"

    df["IncomeGroup"] = df["MonthlyIncome"].apply(create_income_group)

    # Total travelers
    df["TotalTravelers"] = df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]

    print(f"Final dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.3) -> tuple:
    """Split data into train and test sets with stratification."""
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Combine features and target back
    train_df = X_train.copy()
    train_df["ProdTaken"] = y_train.values

    test_df = X_test.copy()
    test_df["ProdTaken"] = y_test.values

    print(f"Training set: {len(train_df)} records")
    print(f"Testing set: {len(test_df)} records")
    print(f"Target distribution (train): {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Target distribution (test): {y_test.value_counts(normalize=True).to_dict()}")

    return train_df, test_df


def save_locally(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data"):
    """Save train and test sets locally."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved training data to: {train_path}")
    print(f"Saved testing data to: {test_path}")


def upload_to_hf(train_df: pd.DataFrame, test_df: pd.DataFrame, repo_id: str, hf_token: str = None):
    """Upload train/test splits as CSV files to Hugging Face Hub.

    Uploads CSV files directly to avoid schema conflicts with the datasets library.
    """
    from huggingface_hub import HfApi

    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()

    # Save to temp files and upload
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.csv")
        test_path = os.path.join(tmpdir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Uploading to Hugging Face Hub: {repo_id}")

        api.upload_file(
            path_or_fileobj=train_path,
            path_in_repo="train.csv",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload train split"
        )

        api.upload_file(
            path_or_fileobj=test_path,
            path_in_repo="test.csv",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload test split"
        )

    print(f"Data successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for model training")
    parser.add_argument(
        "--input-source",
        type=str,
        default="csv",
        choices=["csv", "hf"],
        help="Input source: 'csv' for local file, 'hf' for Hugging Face Hub"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="tourism.csv",
        help="Path to local CSV file (if input-source is 'csv')"
    )
    parser.add_argument(
        "--input-repo-id",
        type=str,
        help="Hugging Face repo ID to load data from (if input-source is 'hf')"
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        help="Hugging Face repo ID to upload prepared data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Local directory to save prepared data"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test set proportion (default: 0.3)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token"
    )

    args = parser.parse_args()

    # Load data
    if args.input_source == "csv":
        df = load_data_from_csv(args.data_path)
    else:
        if not args.input_repo_id:
            raise ValueError("--input-repo-id is required when input-source is 'hf'")
        df = load_data_from_hf(args.input_repo_id)

    # Clean data
    df_clean = clean_data(df)

    # Split data
    train_df, test_df = split_data(df_clean, test_size=args.test_size)

    # Save locally
    save_locally(train_df, test_df, args.output_dir)

    # Upload to HF if repo ID provided
    if args.output_repo_id:
        upload_to_hf(train_df, test_df, args.output_repo_id, args.hf_token)


if __name__ == "__main__":
    main()
