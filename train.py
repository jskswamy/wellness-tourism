"""
Model Training Script with MLflow Tracking

This script trains multiple classification models, tracks experiments with MLflow,
and registers the best model to Hugging Face Model Hub.
"""

import os
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier

from huggingface_hub import HfApi, login


warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def load_data(train_path: str, test_path: str = None):
    """Load training and optionally test data."""
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)

    if test_path:
        print(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    return train_df, None


def load_data_from_hf(repo_id: str):
    """Load train/test CSV files from Hugging Face Hub.

    The prep.py script uploads train.csv and test.csv files directly,
    so we load them using hf_hub_download instead of the datasets library.
    """
    from huggingface_hub import hf_hub_download

    print(f"Loading data from Hugging Face: {repo_id}")

    # Download train and test CSV files from HF Hub
    train_path = hf_hub_download(
        repo_id=repo_id,
        filename="train.csv",
        repo_type="dataset"
    )
    train_df = pd.read_csv(train_path)

    try:
        test_path = hf_hub_download(
            repo_id=repo_id,
            filename="test.csv",
            repo_type="dataset"
        )
        test_df = pd.read_csv(test_path)
    except Exception:
        print("No test.csv found, will split from training data")
        test_df = None

    return train_df, test_df


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    """Prepare data for training."""
    # Separate features and target
    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]

    if test_df is not None:
        X_test = test_df.drop(columns=["ProdTaken"])
        y_test = test_df["ProdTaken"]
    else:
        # Split from training data
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train,
            test_size=0.3,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def create_preprocessor(X_train: pd.DataFrame):
    """Create preprocessing pipeline."""
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="passthrough")

    return preprocessor, numerical_features, categorical_features


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model and return metrics."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None
    }

    return metrics


def train_models(X_train, X_test, y_train, y_test, preprocessor, experiment_name: str):
    """Train multiple models with MLflow tracking."""
    mlflow.set_experiment(experiment_name)

    # Define models and hyperparameter grids
    models = {
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            "params": {
                "classifier__max_depth": [3, 5, 7, 10],
                "classifier__min_samples_split": [5, 10, 20],
                "classifier__min_samples_leaf": [2, 5, 10]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, 15],
                "classifier__min_samples_split": [5, 10],
                "classifier__min_samples_leaf": [2, 5]
            }
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                random_state=RANDOM_STATE
            ),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__min_samples_split": [5, 10]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            ),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__min_child_weight": [1, 3, 5]
            }
        }
    }

    results = {}
    best_model = None
    best_f1 = 0
    best_model_name = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, config in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print("=" * 60)

        with mlflow.start_run(run_name=name):
            # Create pipeline
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", config["model"])
            ])

            # Hyperparameter tuning
            if name in ["GradientBoosting", "XGBoost"]:
                # Use RandomizedSearchCV for larger grids
                search = RandomizedSearchCV(
                    pipeline,
                    config["params"],
                    n_iter=15,
                    cv=cv,
                    scoring="f1",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbose=1
                )
            else:
                search = GridSearchCV(
                    pipeline,
                    config["params"],
                    cv=cv,
                    scoring="f1",
                    n_jobs=-1,
                    verbose=1
                )

            search.fit(X_train, y_train)

            # Best model
            best_pipeline = search.best_estimator_

            # Log parameters
            mlflow.log_params(search.best_params_)
            mlflow.log_param("cv_best_score", search.best_score_)

            # Evaluate
            metrics = evaluate_model(
                best_pipeline,
                X_train, X_test,
                y_train, y_test
            )

            # Log metrics
            for metric_name, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(metric_name, value)

            # Log model with descriptive name
            model_artifact_name = f"{name.lower()}_model"
            mlflow.sklearn.log_model(best_pipeline, name=model_artifact_name)

            print(f"\n{name} Results:")
            print(f"  Best CV Score: {search.best_score_:.4f}")
            for metric_name, value in metrics.items():
                if value is not None:
                    print(f"  {metric_name}: {value:.4f}")

            results[name] = {
                "pipeline": best_pipeline,
                "params": search.best_params_,
                "cv_score": search.best_score_,
                "metrics": metrics
            }

            # Track best model
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = best_pipeline
                best_model_name = name

    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name} (F1 Score: {best_f1:.4f})")
    print("=" * 60)

    return results, best_model, best_model_name


def save_model(model, output_dir: str, numerical_features: list, categorical_features: list):
    """Save model and feature information."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "wellness_tourism_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save feature info
    feature_info = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "model_type": type(model.named_steps["classifier"]).__name__,
        "saved_at": datetime.now().isoformat()
    }

    feature_info_path = os.path.join(output_dir, "feature_info.json")
    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"Feature info saved to: {feature_info_path}")

    return model_path, feature_info_path


def upload_model_to_hf(model_path: str, repo_id: str, metrics: dict, model_name: str, hf_token: str = None):
    """Upload model to Hugging Face Model Hub with model card."""
    from huggingface_hub import create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
    import tempfile

    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Model repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", private=False)

    # Create model card with model-index for evaluation results
    model_card = f'''---
library_name: sklearn
license: mit
tags:
  - tabular-classification
  - sklearn
  - wellness-tourism
datasets:
  - {repo_id.split("/")[0]}/wellness-tourism-data
metrics:
  - accuracy
  - f1
  - precision
  - recall
  - roc_auc
model-index:
  - name: {model_name}
    results:
      - task:
          type: tabular-classification
          name: Binary Classification
        dataset:
          name: Wellness Tourism Dataset
          type: {repo_id.split("/")[0]}/wellness-tourism-data
        metrics:
          - name: Accuracy
            type: accuracy
            value: {metrics.get("test_accuracy", 0):.4f}
          - name: F1 Score
            type: f1
            value: {metrics.get("f1_score", 0):.4f}
          - name: Precision
            type: precision
            value: {metrics.get("precision", 0):.4f}
          - name: Recall
            type: recall
            value: {metrics.get("recall", 0):.4f}
          - name: ROC AUC
            type: roc_auc
            value: {metrics.get("roc_auc", 0):.4f}
---

# Wellness Tourism Package Prediction Model

This model predicts whether a customer will purchase the Wellness Tourism Package offered by "Visit with Us" travel company.

## Model Description

- **Model Type:** {model_name}
- **Task:** Binary Classification
- **Framework:** scikit-learn

## Intended Use

This model is designed to help travel companies identify potential customers for wellness tourism packages based on customer demographics and travel history.

## Training Data

The model was trained on the Wellness Tourism Dataset containing customer information including:
- Demographics (Age, Gender, Marital Status)
- Professional details (Occupation, Designation, Monthly Income)
- Travel history (Number of Trips, Passport status)
- Pitch information (Duration, Property Star preference, Satisfaction Score)

## Evaluation Results

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get("test_accuracy", 0):.4f} |
| F1 Score | {metrics.get("f1_score", 0):.4f} |
| Precision | {metrics.get("precision", 0):.4f} |
| Recall | {metrics.get("recall", 0):.4f} |
| ROC AUC | {metrics.get("roc_auc", 0):.4f} |

## Usage

```python
import joblib
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="wellness_tourism_model.joblib"
)

# Load and use
model = joblib.load(model_path)
predictions = model.predict(your_data)
```

## Limitations

- Model performance may vary on data significantly different from the training distribution
- Predictions should be used as one factor among many in business decisions
'''

    print(f"Uploading model to: {repo_id}")

    # Upload model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="wellness_tourism_model.joblib",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload trained model"
    )

    # Upload model card
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(model_card)
        readme_path = f.name

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card with evaluation results"
    )

    print(f"Model uploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Train models with MLflow tracking")
    parser.add_argument(
        "--input-source",
        type=str,
        default="csv",
        choices=["csv", "hf"],
        help="Input source: 'csv' for local files, 'hf' for Hugging Face Hub"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/train.csv",
        help="Path to training data (if input-source is 'csv')"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/test.csv",
        help="Path to test data (if input-source is 'csv')"
    )
    parser.add_argument(
        "--data-repo-id",
        type=str,
        help="Hugging Face dataset repo ID (if input-source is 'hf')"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="wellness_tourism_prediction",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for saved model"
    )
    parser.add_argument(
        "--model-repo-id",
        type=str,
        help="Hugging Face model repo ID for uploading"
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
        train_df, test_df = load_data(args.train_path, args.test_path)
    else:
        if not args.data_repo_id:
            raise ValueError("--data-repo-id is required when input-source is 'hf'")
        train_df, test_df = load_data_from_hf(args.data_repo_id)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)

    # Create preprocessor
    preprocessor, numerical_features, categorical_features = create_preprocessor(X_train)

    # Train models
    results, best_model, best_model_name = train_models(
        X_train, X_test, y_train, y_test,
        preprocessor, args.experiment_name
    )

    # Save best model
    model_path, feature_info_path = save_model(
        best_model, args.output_dir,
        numerical_features, categorical_features
    )

    # Upload to Hugging Face if repo ID provided
    if args.model_repo_id:
        best_metrics = results[best_model_name]["metrics"]
        upload_model_to_hf(
            model_path, args.model_repo_id,
            best_metrics, best_model_name,
            args.hf_token
        )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
