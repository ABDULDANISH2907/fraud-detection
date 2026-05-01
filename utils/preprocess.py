import pandas as pd

TRANSACTION_TYPE_MAP = {
    "Online purchase": 1.0,
    "POS payment": 0.5,
    "ATM withdrawal": -0.5,
    "Transfer": -1.0,
    "Mobile purchase": 0.2,
}


def build_feature_vector(user_inputs, feature_names):
    base_features = {name: 0.0 for name in feature_names}

    if "Amount" in base_features:
        base_features["Amount"] = float(user_inputs["amount"])

    if "Time" in base_features:
        base_features["Time"] = float(user_inputs.get("transaction_age_minutes", 0.0))

    if "V1" in base_features:
        base_features["V1"] = TRANSACTION_TYPE_MAP.get(
            user_inputs.get("transaction_type"), 0.0
        )

    if "V2" in base_features:
        base_features["V2"] = float(user_inputs.get("account_age_days", 0.0)) / 365.0

    if "V3" in base_features:
        base_features["V3"] = float(user_inputs.get("previous_alerts", 0.0))

    if "V4" in base_features:
        base_features["V4"] = float(user_inputs.get("account_risk_score", 0.0)) / 100.0

    raw_features = pd.DataFrame([base_features], columns=feature_names)
    return raw_features


def preprocess_input(model_package, user_inputs):
    scaler = model_package["scaler"]
    feature_names = model_package["feature_names"]

    raw_features = build_feature_vector(user_inputs, feature_names)
    scaled_features = scaler.transform(raw_features)
    return scaled_features, raw_features
