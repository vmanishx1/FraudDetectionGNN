import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from app.components.ui_components import section_header, page_header, custom_divider, metric_card


def render_analytics():
    page_header("Analytics & Insights")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Avg Fraud Amount", "$2,847", "+8.3%", "#DC2626")
    with col2:
        metric_card("Top Risk City", "New York", "142 cases", "#F59E0B")
    with col3:
        metric_card("Riskiest Merchant", "CryptoEx", "12.4% fraud rate", "#DC2626")
    with col4:
        metric_card("Peak Fraud Hour", "2 AM", "3.2x avg rate", "#F59E0B")

    custom_divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 Location Analytics", "🏪 Merchant Analytics",
        "💳 Payment Analytics", "📅 Time Analytics"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            section_header("Top Fraud Locations")
            cities = ["New York", "Los Angeles", "Chicago", "Miami", "Houston", "San Francisco", "Seattle", "International"]
            fraud_counts = np.random.randint(20, 200, len(cities))
            total_counts = fraud_counts / np.random.uniform(0.02, 0.10, len(cities))
            df_locations = pd.DataFrame({
                "City": cities, "Fraud": fraud_counts, "Total": total_counts.astype(int),
                "Rate": (fraud_counts / total_counts * 100).round(2)
            }).sort_values("Fraud", ascending=True)

            fig = px.bar(df_locations, y="City", x="Fraud",
                         color="Rate", color_continuous_scale="Reds",
                         text="Fraud", orientation="h")
            fig.update_layout(template="plotly_white", height=400,
                              title="Fraud Cases by Location")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_header("Fraud Heatmap by Region")
            z = np.random.randint(0, 100, (6, 8))
            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=["NY", "LA", "CHI", "MIA", "HOU", "SF", "SEA", "INT"],
                y=["North", "South", "East", "West", "Central", "International"],
                colorscale="Reds",
                hovertemplate="Region: %{y}<br>City: %{x}<br>Cases: %{z}<extra></extra>"
            ))
            fig.update_layout(template="plotly_white", height=400,
                              title="Fraud Cases Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section_header("Fraud by Merchant Category")
        merchants_data = pd.DataFrame({
            "Merchant": ["CryptoEx", "ElectroWorld", "LuxuryGoods", "TravelAgency",
                         "OnlineMart", "Amazon", "Walmart", "BestBuy", "Target", "eBay"],
            "Fraud Rate": np.random.uniform(1, 15, 10),
            "Total Transactions": np.random.randint(500, 10000, 10),
            "Fraud Cases": np.random.randint(10, 500, 10),
        }).sort_values("Fraud Rate", ascending=False)

        fig = px.bar(merchants_data, x="Merchant", y="Fraud Rate",
                     color="Fraud Cases", color_continuous_scale="Reds",
                     text=merchants_data["Fraud Rate"].round(1).astype(str) + "%")
        fig.update_layout(template="plotly_white", height=400,
                          title="Fraud Rate by Merchant")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            section_header("Merchant Risk Distribution")
            risk_levels = ["Safe", "Low", "Medium", "High", "Critical"]
            risk_counts = np.random.randint(5, 50, 5)
            fig = go.Figure(go.Pie(
                labels=risk_levels, values=risk_counts,
                hole=0.4,
                marker=dict(colors=["#16A34A", "#2563EB", "#F59E0B", "#F97316", "#DC2626"]),
                textinfo="label+percent",
            ))
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_header("Top Risky Customers")
            customers = pd.DataFrame({
                "Customer ID": [f"CUST-{i}" for i in range(1, 11)],
                "Risk Score": np.random.randint(50, 100, 10),
                "Fraud Attempts": np.random.randint(1, 10, 10),
            }).sort_values("Risk Score", ascending=False)
            fig = px.bar(customers.head(5), y="Customer ID", x="Risk Score",
                         color="Fraud Attempts", color_continuous_scale="Reds",
                         orientation="h", text="Risk Score")
            fig.update_layout(template="plotly_white", height=300,
                              title="High-Risk Customers")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            section_header("Fraud by Payment Method")
            payment_types = ["Credit Card", "Debit Card", "Wire Transfer", "Crypto", "PayPal"]
            payment_frauds = np.random.randint(20, 200, 5)
            fig = go.Figure(go.Pie(
                labels=payment_types, values=payment_frauds, hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo="label+percent",
            ))
            fig.update_layout(template="plotly_white", height=350,
                              title="Fraud Distribution by Payment Type")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_header("Payment Method Risk Analysis")
            df_payment = pd.DataFrame({
                "Method": payment_types,
                "Fraud Rate": np.random.uniform(0.5, 8, 5),
                "Avg Amount": np.random.randint(100, 5000, 5),
            }).sort_values("Fraud Rate", ascending=False)
            fig = px.bar(df_payment, x="Method", y="Fraud Rate",
                         color="Avg Amount", color_continuous_scale="Blues",
                         text=df_payment["Fraud Rate"].round(1).astype(str) + "%")
            fig.update_layout(template="plotly_white", height=350,
                              title="Fraud Rate by Payment Method")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            section_header("Fraud Over Time (Daily)")
            days = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
            daily_frauds = np.random.randint(5, 30, 30)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days, y=daily_frauds, mode="lines+markers",
                fill="tozeroy", name="Fraud Cases",
                line=dict(color="#DC2626", width=2),
            ))
            fig.update_layout(template="plotly_white", height=350,
                              title="Daily Fraud Trend")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_header("Fraud by Hour of Day")
            hours = list(range(24))
            hourly_frauds = np.random.randint(5, 50, 24)
            hourly_frauds[0:6] = hourly_frauds[0:6] * 2
            fig = go.Figure(go.Bar(
                x=hours, y=hourly_frauds,
                marker=dict(
                    color=hourly_frauds,
                    colorscale="Reds",
                ),
            ))
            fig.update_layout(template="plotly_white", height=350,
                              title="Fraud Pattern by Hour",
                              xaxis_title="Hour of Day",
                              yaxis_title="Fraud Cases")
            st.plotly_chart(fig, use_container_width=True)

    custom_divider()

    section_header("Export Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("📥 Export CSV", use_container_width=True)
    with col2:
        st.button("📊 Export PDF Report", use_container_width=True)
    with col3:
        st.button("📈 Export Charts", use_container_width=True)
