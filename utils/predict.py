import numpy as np


def predict_transaction(model, scaled_input):
    probabilities = model.predict_proba(scaled_input)
    fraud_probability = float(probabilities[0, 1])
    label = "Fraud" if fraud_probability >= 0.5 else "Not Fraud"
    confidence = max(fraud_probability, 1.0 - fraud_probability)
    return label, fraud_probability, confidence


def explain_features(model, raw_features):
    feature_names = raw_features.columns.tolist()

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefficient = np.asarray(model.coef_).ravel()
        scores = np.abs(coefficient)
    else:
        return []

    importance = sorted(
        zip(feature_names, scores), key=lambda pair: pair[1], reverse=True
    )

    return [
        {"feature": feature, "score": float(score)}
        for feature, score in importance[:5]
        if score > 0
    ]
