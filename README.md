# Wellness Tourism MLOps Pipeline

[![MLOps Pipeline](https://github.com/jskswamy/wellness-tourism-mlops/actions/workflows/pipeline.yml/badge.svg)](https://github.com/jskswamy/wellness-tourism-mlops/actions/workflows/pipeline.yml)

## Project Overview

This repository implements an end-to-end MLOps pipeline for predicting customer purchases of the Wellness Tourism Package for "Visit With Us" travel company.

### Business Problem

The company wants to identify customers likely to purchase the newly introduced Wellness Tourism Package, enabling targeted marketing and improved conversion rates.

### Solution Architecture

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Data Register  │────▶│  Data Prepare   │────▶│  Model Train    │
    │  (HF Dataset)   │     │  (Clean/Split)  │     │  (MLflow Track) │
    └─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                             │
                                                             ▼
                                                   ┌─────────────────┐
                                                   │  Deploy to HF   │
                                                   │  Spaces         │
                                                   └─────────────────┘

## Model Information

| Attribute | Value |
|-----------|-------|
| **Algorithm** | XGBoost Classifier |
| **Target Variable** | ProdTaken (Binary: 0/1) |
| **Features** | 18 customer attributes |
| **Tracking** | MLflow Experiment Tracking |

### Key Features Used

- **Demographics**: Age, Gender, MaritalStatus, Occupation
- **Travel History**: NumberOfTrips, Passport, PreferredPropertyStar
- **Sales Interaction**: TypeofContact, DurationOfPitch, NumberOfFollowups
- **Product**: ProductPitched, PitchSatisfactionScore

## Repository Structure

    .
    ├── data_register.py      # Upload raw data to HF Dataset Hub
    ├── prep.py               # Data cleaning and train/test split
    ├── train.py              # Model training with MLflow tracking
    ├── deployment/
    │   ├── app.py            # Streamlit prediction interface
    │   ├── Dockerfile        # Container configuration
    │   └── requirements.txt  # App dependencies
    ├── hosting/
    │   └── hosting.py        # Deploy to HF Spaces
    ├── .github/workflows/
    │   └── pipeline.yml      # CI/CD automation
    ├── requirements.txt      # Pipeline dependencies
    ├── README.md             # This file
    └── tourism.csv           # Raw dataset

## CI/CD Pipeline

The GitHub Actions workflow automatically executes on push to `main`:

1. **register-dataset**: Uploads raw CSV to Hugging Face Dataset Hub
2. **data-prep**: Cleans data, creates train/test split, uploads to HF
3. **model-training**: Trains XGBoost with MLflow tracking, uploads model to HF Model Hub
4. **deploy-hosting**: Deploys Streamlit app to Hugging Face Spaces

## Deployed Resources

| Resource | URL |
|----------|-----|
| Dataset | https://huggingface.co/datasets/jskswamy/wellness-tourism-data |
| Model | https://huggingface.co/jskswamy/wellness-tourism-model |
| App | https://huggingface.co/spaces/jskswamy/wellness-tourism-predictor |

## Local Development

    # Set environment variables
    export HF_TOKEN="your-token"
    export HF_USERNAME="jskswamy"

    # Run pipeline steps
    python data_register.py
    python prep.py
    python train.py
    python hosting/hosting.py

## Technologies Used

- **ML Framework**: scikit-learn, XGBoost
- **Experiment Tracking**: MLflow
- **Model Registry**: Hugging Face Model Hub
- **Data Versioning**: Hugging Face Dataset Hub
- **Deployment**: Streamlit on Hugging Face Spaces
- **CI/CD**: GitHub Actions
