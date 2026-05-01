from pathlib import Path
import warnings
import tempfile
import os
import joblib

import xgboost as xgb


def _normalize_xgb_model(model):
    if not isinstance(model, xgb.XGBModel):
        return model

    if not hasattr(model, "save_model") or not hasattr(model, "load_model"):
        return model

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temporary_file:
        temp_path = temporary_file.name

    try:
        model.save_model(temp_path)
        normalized_model = xgb.XGBClassifier()
        normalized_model.load_model(temp_path)
        return normalized_model
    except Exception:
        return model
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def load_model_package(model_path=None):
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent / "models" / "fraud_model_v1.pkl"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*If you are loading a serialized model.*",
            category=UserWarning,
        )
        package = joblib.load(model_path)

    expected_keys = {"model", "scaler", "feature_names"}
    if not expected_keys.issubset(package.keys()):
        raise ValueError("Loaded package is missing required model components.")

    package["model"] = _normalize_xgb_model(package["model"])
    return package


def validate_amount(amount):
    if amount is None or amount <= 0:
        raise ValueError("Transaction amount must be greater than zero.")
    return float(amount)


def validate_integer(value, name, minimum=0):
    if value is None:
        raise ValueError(f"{name} must be provided.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def validate_score(score):
    if score is None:
        raise ValueError("Risk score must be provided.")
    if score < 0 or score > 100:
        raise ValueError("Risk score must be between 0 and 100.")
    return float(score)


def format_probability(value):
    return f"{value * 100:.1f}%"


def get_prediction_label(score, threshold=0.5):
    return "Fraud" if score >= threshold else "Not Fraud"


def get_risk_message():
    return (
        "This model was trained on a highly imbalanced fraud dataset. "
        "Review low-confidence predictions with additional business rules."
    )
