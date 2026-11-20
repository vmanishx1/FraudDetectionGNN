import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any

from app.components.ui_components import metric_card, fraud_badge, risk_badge, info_box, section_header, page_header, custom_divider


def render_dashboard():
    page_header("Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Total Transactions", "48,592", "+12.5% vs last month", "#1E3A8A")
    with col2:
        metric_card("Fraud Transactions", "1,284", "2.64% rate", "#DC2626")
    with col3:
        metric_card("Normal Transactions", "47,308", "97.36% rate", "#16A34A")

    col4, col5, col6, col7, col8 = st.columns(5)
    with col4:
        metric_card("Accuracy", "97.8%", "", "#2563EB")
    with col5:
        metric_card("Precision", "94.2%", "", "#2563EB")
    with col6:
        metric_card("Recall", "91.5%", "", "#2563EB")
    with col7:
        metric_card("F1 Score", "92.8%", "", "#2563EB")
    with col8:
        metric_card("ROC AUC", "98.1%", "", "#2563EB")

    custom_divider()

    col1, col2 = st.columns(2)

    with col1:
        section_header("Fraud Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[97.36, 2.64],
            hole=0.4,
            marker=dict(colors=["#16A34A", "#DC2626"]),
            textinfo="label+percent",
        )])
        fig.update_layout(template="plotly_white", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Monthly Fraud Trend")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fraud_trend = np.random.randint(80, 200, 12)
        normal_trend = np.random.randint(3000, 5000, 12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=fraud_trend, mode="lines+markers",
                                 name="Fraud", line=dict(color="#DC2626", width=3)))
        fig.add_trace(go.Scatter(x=months, y=normal_trend, mode="lines+markers",
                                 name="Normal", line=dict(color="#16A34A", width=3)))
        fig.update_layout(template="plotly_white", height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    custom_divider()

    section_header("Recent Alerts")
    alerts = [
        {"id": "TXN-2024-001", "amount": "$12,500", "merchant": "ElectroWorld", "risk": "Critical", "time": "2 min ago"},
        {"id": "TXN-2024-002", "amount": "$8,200", "merchant": "LuxuryGoods", "risk": "High", "time": "15 min ago"},
        {"id": "TXN-2024-003", "amount": "$5,000", "merchant": "OnlineMart", "risk": "Medium", "time": "1 hr ago"},
        {"id": "TXN-2024-004", "amount": "$15,000", "merchant": "CryptoEx", "risk": "Critical", "time": "2 hrs ago"},
        {"id": "TXN-2024-005", "amount": "$3,200", "merchant": "TravelAgency", "risk": "High", "time": "3 hrs ago"},
    ]
    for alert in alerts:
        col_a, col_b, col_c, col_d, col_e = st.columns([2, 1.5, 1.5, 1, 1])
        with col_a:
            st.markdown(f"**{alert['id']}**")
        with col_b:
            st.markdown(alert['amount'])
        with col_c:
            st.markdown(alert['merchant'])
        with col_d:
            st.markdown(risk_badge(alert['risk']), unsafe_allow_html=True)
        with col_e:
            st.markdown(f"_{alert['time']}_")

    custom_divider()

    st.subheader("🤖 Model Performance Comparison")
    models_df = pd.DataFrame({
        "Model": ["GCN", "GraphSAGE", "GAT"],
        "Accuracy": [0.978, 0.972, 0.981],
        "Precision": [0.942, 0.935, 0.951],
        "Recall": [0.915, 0.908, 0.927],
        "F1 Score": [0.928, 0.921, 0.939],
        "ROC AUC": [0.981, 0.976, 0.987],
    })
    fig = px.bar(models_df, x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                 barmode="group", color_discrete_sequence=["#2563EB", "#16A34A", "#DC2626", "#F59E0B", "#8B5CF6"])
    fig.update_layout(template="plotly_white", height=400, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
