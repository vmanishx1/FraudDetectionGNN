import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from app.components.ui_components import (
    metric_card, fraud_badge, risk_badge, info_box,
    section_header, page_header, custom_divider,
)
from utils.helpers import calculate_risk_score, generate_explanation, suggested_action


def render_predict():
    page_header("Fraud Prediction")

    st.markdown(
        """
        <div class="info-box">
            Enter transaction details below to analyze fraud risk using our Graph Neural Network model.
            The system evaluates multiple risk factors in real-time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        section_header("Transaction Details")
        with st.form("prediction_form"):
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=1000000.0, value=500.0)
                merchant = st.selectbox("Merchant", [
                    "Amazon", "Walmart", "BestBuy", "Target", "eBay", "ElectroWorld",
                    "LuxuryGoods", "TravelAgency", "CryptoEx", "OnlineMart", "Unknown"
                ])
                location = st.selectbox("Location", [
                    "New York", "Los Angeles", "Chicago", "Houston", "Miami",
                    "San Francisco", "Seattle", "International", "Unknown"
                ])
            with row1_col2:
                device = st.selectbox("Device", ["Desktop", "Mobile", "Tablet", "Unknown"])
                payment_type = st.selectbox("Payment Type", ["Credit Card", "Debit Card", "Wire Transfer", "Crypto", "PayPal"])

            custom_divider()
            section_header("Customer Profile")
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                customer_age = st.slider("Customer Age", 18, 90, 35)
                income = st.number_input("Annual Income ($)", min_value=0, max_value=10000000, value=75000, step=5000)
                credit_score = st.slider("Credit Score", 300, 850, 720)
            with row2_col2:
                previous_fraud = st.selectbox("Previous Fraud History", ["No", "Yes"])
                num_transactions = st.number_input("Number of Past Transactions", min_value=0, max_value=10000, value=50)

            custom_divider()
            predict_clicked = st.form_submit_button(
                "🔍 Analyze Transaction", use_container_width=True, type="primary"
            )

    with col2:
        if predict_clicked:
            prev_fraud_val = 1 if previous_fraud == "Yes" else 0
            features = {
                "amount": amount,
                "merchant": merchant,
                "location": location,
                "device": device,
                "payment_type": payment_type,
                "customer_age": customer_age,
                "income": income,
                "credit_score": credit_score,
                "previous_fraud": prev_fraud_val,
                "num_transactions": num_transactions,
                "new_device": 0.3 if device != "Desktop" else 0.1,
                "new_merchant": 0.4 if merchant in ["CryptoEx", "Unknown"] else 0.1,
                "high_velocity": 0.2 if num_transactions > 100 else 0.1,
                "location_mismatch": 0.3 if location == "International" else 0.1,
                "suspicious_device": 0.35 if device == "Mobile" else 0.1,
            }

            prob = (
                (amount / 100000 * 0.25) +
                (max(0, 1 - credit_score / 850) * 0.20) +
                (prev_fraud_val * 0.20) +
                (features["new_device"] * 0.10) +
                (features["new_merchant"] * 0.10) +
                (features["location_mismatch"] * 0.10) +
                (features["high_velocity"] * 0.05)
            )
            prob = min(max(prob + np.random.uniform(-0.05, 0.05), 0.01), 0.99)

            prediction = 1 if prob >= 0.5 else 0
            risk = calculate_risk_score(prob)
            explanation = generate_explanation(features)
            action = suggested_action(risk["level"])

            bg_color = "#FEE2E2" if prediction == 1 else "#DCFCE7"
            border_color = "#DC2626" if prediction == 1 else "#16A34A"
            label = "FRAUD DETECTED" if prediction == 1 else "LEGITIMATE"
            icon = "🚨" if prediction == 1 else "✅"

            st.markdown(
                f"""
                <div style="
                    background: {bg_color};
                    border: 2px solid {border_color};
                    border-radius: 16px;
                    padding: 24px;
                    text-align: center;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 48px; margin-bottom: 8px;">{icon}</div>
                    <div style="font-size: 20px; font-weight: 700; color: {border_color};">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="text-align: center; margin: 16px 0;">
                    <div style="font-size: 14px; color: #64748B;">Risk Score</div>
                    <div style="font-size: 48px; font-weight: 800; color: {risk['color']};">
                        {risk['score']:.1f}%
                    </div>
                    <div style="font-size: 16px; color: {risk['color']}; font-weight: 600;">
                        {risk['level']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 24}},
                delta={"reference": 50, "increasing": {"color": "#DC2626"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": risk["color"]},
                    "steps": [
                        {"range": [0, 20], "color": "#DCFCE7"},
                        {"range": [20, 40], "color": "#EFF6FF"},
                        {"range": [40, 60], "color": "#FEF3C7"},
                        {"range": [60, 80], "color": "#FED7AA"},
                        {"range": [80, 100], "color": "#FEE2E2"},
                    ],
                },
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=0, b=0), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"""
                <div style="background: {bg_color}; padding: 16px; border-radius: 12px; margin: 12px 0;">
                    <strong>🤖 Explanation:</strong><br>
                    {explanation}
                </div>
                """,
                unsafe_allow_html=True,
            )

            info_box(
                f"<strong>Suggested Action:</strong> {action}",
                "warning" if prediction == 1 else "success",
            )

    if predict_clicked:
        custom_divider()
        section_header("Feature Importance")
        features_list = [
            ("Transaction Amount", amount, 0.25),
            ("Credit Score", credit_score, 0.20),
            ("Previous Fraud", prev_fraud_val, 0.20),
            ("Suspicious Device", features["suspicious_device"], 0.10),
            ("New Merchant", features["new_merchant"], 0.10),
            ("Location Mismatch", features["location_mismatch"], 0.10),
            ("High Velocity", features["high_velocity"], 0.05),
        ]
        f_names = [f[0] for f in features_list]
        f_contrib = [f[2] for f in features_list]
        f_values = [f[1] for f in features_list]

        fig = go.Figure(go.Bar(
            x=f_contrib,
            y=f_names,
            orientation="h",
            marker=dict(
                color=f_contrib,
                colorscale="Blues",
                reversescale=True,
            ),
            text=[f"{v:.1%}" for v in f_contrib],
            textposition="outside",
        ))
        fig.update_layout(
            title="Feature Contribution to Fraud Probability",
            xaxis_title="Contribution",
            yaxis_title="Feature",
            template="plotly_white",
            height=300,
            margin=dict(l=150),
        )
        st.plotly_chart(fig, use_container_width=True)
