import streamlit as st
from pathlib import Path

from utils.helpers import (
    format_probability,
    get_risk_message,
    load_model_package,
    validate_amount,
    validate_integer,
    validate_score,
)
from utils.predict import explain_features, predict_transaction
from utils.preprocess import preprocess_input

MODEL_FILE = Path(__file__).resolve().parent /"fraud_model_v1.pkl"
TRANSACTION_TYPES = [
    "Online purchase",
    "POS payment",
    "ATM withdrawal",
    "Transfer",
    "Mobile purchase",
]


def load_model():
    return load_model_package(MODEL_FILE)


def inject_style():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            font-family: Inter, sans-serif;
        }

        .stApp {
            background: #070b18;
            color: #e8edf7;
        }

        .main > div.block-container {
            padding-top: 24px;
            padding-bottom: 36px;
            max-width: 920px;
        }

        .hero-card,
        .section-card,
        .result-card {
            border-radius: 24px;
            padding: 28px;
            background: rgba(13, 18, 35, 0.96);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.06);
            margin-bottom: 24px;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.04em;
        }

        .hero-subtitle {
            color: #a9b2d7;
            font-size: 1rem;
            line-height: 1.7;
            margin-top: 0;
        }

        .section-label {
            color: #7c89b6;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.75rem;
            margin-bottom: 12px;
        }

        .stButton>button {
            width: 100%;
            border-radius: 16px;
            padding: 14px 18px;
            background: linear-gradient(135deg, #4f7dff, #6ab4ff);
            color: white;
            font-size: 1rem;
            font-weight: 700;
            border: none;
            box-shadow: 0 14px 32px rgba(37, 83, 255, 0.18);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 38px rgba(37, 83, 255, 0.26);
        }

        .input-label {
            display: block;
            margin-bottom: 8px;
            color: #c1c9e4;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .hint-text {
            color: #6d79a0;
            font-size: 0.88rem;
            margin-top: 4px;
        }

        .callout {
            background: rgba(65, 92, 170, 0.12);
            border-left: 4px solid #5f8cff;
            padding: 14px 18px;
            border-radius: 14px;
            color: #d0d7f2;
            margin-top: 20px;
        }

        .result-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 10px 18px;
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 18px;
            width: fit-content;
        }

        .result-success {
            background: rgba(57, 189, 98, 0.14);
            color: #b8f0b4;
        }

        .result-danger {
            background: rgba(255, 108, 108, 0.14);
            color: #ffb7b7;
        }

        .progress-bar {
            height: 14px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255,255,255,0.08);
            margin-top: 12px;
        }

        .progress-bar-inner {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #62d5ff, #6a8cff);
            transition: width 0.5s ease;
        }

        .value-box {
            display: inline-flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            margin-top: 18px;
        }

        .label-small {
            color: #8fa5d0;
            font-size: 0.88rem;
        }

        .section-compact {
            gap: 18px;
        }

        @media (max-width: 768px) {
            .hero-title {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class='hero-card'>
            <div class='section-label'>Fraud Intelligence</div>
            <div class='hero-title'>🚨 Transaction Risk Monitor</div>
            <p class='hero-subtitle'>Fast, consistent fraud scoring with the same preprocessing pipeline used in production.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info():
    st.markdown(
        """
        <div class='callout'>
            <strong>Class imbalance awareness:</strong> this model was trained on a highly imbalanced dataset. Use the probability score and follow business rules for low-confidence cases.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(label, probability, confidence, prediction_inputs, importance_rows):
    status_class = "result-danger" if label == "Fraud" else "result-success"
    percentage = int(probability * 100)

    st.markdown(
        """
        <div class='result-card'>
            <div class='result-badge {status_class}'>{emoji} {label}</div>
            <div class='value-box'>
                <div>
                    <div class='label-small'>Confidence score</div>
                    <div style='font-size: 1.95rem; font-weight: 700;'>{confidence_text}</div>
                </div>
                <div style='text-align: right;'>
                    <div class='label-small'>Fraud probability</div>
                    <div style='font-size: 1.2rem; font-weight: 700; color: #ffffff;'>{prob_text}</div>
                </div>
            </div>
            <div class='progress-bar'>
                <div class='progress-bar-inner' style='width: {percentage}%'></div>
            </div>
            <p style='margin-top: 20px; color: #b0bce7;'>This scoring uses the same trained model and preprocessing path as production. Review the risk message before escalation.</p>
        </div>
        """.format(
            status_class=status_class,
            emoji="🔥" if label == "Fraud" else "✅",
            label=label,
            confidence_text=format_probability(confidence),
            prob_text=format_probability(probability),
            percentage=percentage,
        ),
        unsafe_allow_html=True,
    )

    if importance_rows:
        st.markdown("### Key feature contributions")
        st.table(
            {
                "Feature": [row["feature"] for row in importance_rows],
                "Relative score": [f"{row['score']:.4f}" for row in importance_rows],
            }
        )

    with st.expander("Prediction details"):
        st.write(f"**Transaction type:** {prediction_inputs['transaction_type']}")
        st.write(f"**Account age (days):** {prediction_inputs['account_age_days']}")
        st.write(f"**Previous alerts:** {prediction_inputs['previous_alerts']}")
        st.write(f"**Account risk score:** {prediction_inputs['account_risk_score']}")
        st.write(f"**Transaction age (minutes):** {prediction_inputs['transaction_age_minutes']}")
        st.info(get_risk_message())


def main():
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🚨",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    inject_style()
    render_header()
    render_info()

    try:
        model_package = load_model()
    except Exception as error:
        st.error(f"Unable to load model: {error}")
        return

    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### Transaction profile")

        with st.form(key="prediction_form"):
            col1, col2 = st.columns([2, 1], gap="large")
            with col1:
                amount = st.number_input(
                    label="Amount ($)",
                    min_value=0.01,
                    value=58.40,
                    step=0.25,
                    format="%.2f",
                )
                st.markdown("<div class='hint-text'>Total transaction value.</div>", unsafe_allow_html=True)

                transaction_type = st.selectbox("Transaction type", TRANSACTION_TYPES)
                st.markdown("<div class='hint-text'>Choose the category that best matches the transaction.</div>", unsafe_allow_html=True)

            with col2:
                account_age_days = st.slider("Account age (days)", 0, 5000, 420)
                st.markdown("<div class='hint-text'>Number of days the account has been active.</div>", unsafe_allow_html=True)

                previous_alerts = st.number_input(
                    "Previous alerts",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1,
                )
                st.markdown("<div class='hint-text'>Historic fraud flags for this account.</div>", unsafe_allow_html=True)

            col3, col4 = st.columns([1, 1], gap="large")
            with col3:
                account_risk_score = st.slider("Account risk score", 0, 100, 12)
                st.markdown("<div class='hint-text'>Internal risk rating between 0 and 100.</div>", unsafe_allow_html=True)

            with col4:
                transaction_age_minutes = st.slider(
                    "Transaction age (minutes)",
                    0,
                    43200,
                    120,
                )
                st.markdown("<div class='hint-text'>Minutes since the account was created.</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;margin-top:18px;'>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Predict fraud risk")
            st.markdown("</div>", unsafe_allow_html=True)

        if submit_button:
            with st.spinner("Analyzing transaction..."):
                try:
                    prediction_inputs = {
                        "amount": validate_amount(amount),
                        "transaction_type": transaction_type,
                        "account_age_days": validate_integer(account_age_days, "Account age"),
                        "previous_alerts": validate_integer(previous_alerts, "Previous fraud alerts"),
                        "account_risk_score": validate_score(account_risk_score),
                        "transaction_age_minutes": validate_integer(
                            transaction_age_minutes, "Transaction age"
                        ),
                    }

                    scaled_input, raw_features = preprocess_input(model_package, prediction_inputs)
                    label, probability, confidence = predict_transaction(
                        model_package["model"], scaled_input
                    )
                    importance_rows = explain_features(model_package["model"], raw_features)

                    render_result(label, probability, confidence, prediction_inputs, importance_rows)

                except Exception as error:
                    st.error(f"Input error: {error}")

    st.markdown(
        """
        <div class='callout'>
            <strong>Note:</strong> All values are scored with the production preprocessing path. Use this interface for quick fraud triage and review borderline cases carefully.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
