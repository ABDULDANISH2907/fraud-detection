# Fraud Detection System

## Project Overview

This repository contains a production-ready fraud detection application built with Python and Streamlit. It uses a trained machine learning model to score transaction risk and provide confidence information for fraud detection decisions.

## Problem Statement

Fraud detection is a critical challenge for financial institutions. This application evaluates individual transactions and identifies suspicious activity using a pre-trained fraud classification model.

## Tech Stack

- Python
- Streamlit
- scikit-learn
- XGBoost
- pandas
- numpy
- joblib

## Folder Structure

```
fraud-detection/
├── app.py
├── models/
│    └── fraud_model_v1.pkl
├── utils/
│    ├── helpers.py
│    ├── predict.py
│    └── preprocess.py
├── requirements.txt
└── README.md
```

## How to Run Locally

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the environment:
   - Windows:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - macOS / Linux:
     ```bash
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## App Features

- Loads the production-trained fraud model from `models/fraud_model_v1.pkl`
- Recreates the preprocessing pipeline used during model training
- Validates user inputs for transaction amount, account age, risk score, and more
- Displays fraud / not fraud prediction with a confidence score
- Provides feature importance or explanation details
- Includes class imbalance awareness and risk guidance

## Screenshots

> Add screenshots here after running the app locally.

## Future Improvements

- Add support for real transaction event data streaming
- Integrate with an API backend for live scoring
- Implement model monitoring, drift detection, and alerting
- Add user authentication and audit logging for predictions
- Improve explainability with SHAP and feature contribution breakdowns


cd "C:\danish (data)\training_notebooks\fraud-detection"

.\.venv\Scripts\activate.bat

streamlit run app.py