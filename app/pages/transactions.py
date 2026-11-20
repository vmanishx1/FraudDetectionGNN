import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.components.ui_components import metric_card, fraud_badge, risk_badge, info_box, section_header, page_header, custom_divider


def render_transactions():
    page_header("Transaction Search & History")

    with st.expander("🔍 Search Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            search_query = st.text_input("Search", placeholder="Transaction ID, Customer, Merchant...")
        with col2:
            status_filter = st.selectbox("Status", ["All", "Completed", "Flagged", "Blocked", "Under Review"])
        with col3:
            date_range = st.date_input("Date Range", [datetime.now() - timedelta(days=30), datetime.now()])

        col4, col5, col6 = st.columns(3)
        with col4:
            min_amount = st.number_input("Min Amount ($)", min_value=0.0, value=0.0)
        with col5:
            max_amount = st.number_input("Max Amount ($)", min_value=0.0, value=100000.0)
        with col6:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

    custom_divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Filtered Results", "1,247", "25.6% of total", "#1E3A8A")
    with col2:
        metric_card("Fraud Found", "42", "3.4% rate", "#DC2626")
    with col3:
        metric_card("Total Amount", "$3.2M", "Avg $2,567", "#2563EB")

    custom_divider()

    section_header("Transaction List")

    np.random.seed(42)
    transactions = []
    for i in range(20):
        is_fraud = np.random.random() < 0.15
        tx = {
            "ID": f"TXN-{np.random.randint(10000, 99999)}",
            "Date": (datetime.now() - timedelta(hours=np.random.randint(1, 720))).strftime("%Y-%m-%d %H:%M"),
            "Amount": f"${np.random.randint(100, 50000):,}",
            "Merchant": np.random.choice(["Amazon", "Walmart", "BestBuy", "Target", "eBay", "Unknown"]),
            "Location": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Miami", "International"]),
            "Status": np.random.choice(["Completed", "Flagged", "Blocked"], p=[0.7, 0.2, 0.1]),
            "Risk": np.random.choice(["Safe", "Low", "Medium", "High", "Critical"], p=[0.4, 0.3, 0.15, 0.1, 0.05]),
            "Fraud": is_fraud,
        }
        transactions.append(tx)

    df = pd.DataFrame(transactions)
    for _, row in df.iterrows():
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5, 1.5, 1, 1.2, 1, 0.8, 0.8])
        with col1:
            st.markdown(f"**{row['ID']}**")
        with col2:
            st.markdown(f"<small>{row['Date']}</small>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"**{row['Amount']}**")
        with col4:
            st.markdown(row['Merchant'])
        with col5:
            st.markdown(row['Location'])
        with col6:
            st.markdown(risk_badge(row['Risk']), unsafe_allow_html=True)
        with col7:
            st.markdown(fraud_badge(row['Fraud']), unsafe_allow_html=True)
        st.markdown('<hr style="margin: 4px 0; opacity: 0.2;">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; gap: 8px; margin-top: 16px;">
                <button style="background: #1E3A8A; color: white; border: none; padding: 8px 16px; border-radius: 8px;">Previous</button>
                <button style="background: #1E3A8A; color: white; border: none; padding: 8px 16px; border-radius: 8px;">1</button>
                <button style="background: #E2E8F0; border: none; padding: 8px 16px; border-radius: 8px;">2</button>
                <button style="background: #E2E8F0; border: none; padding: 8px 16px; border-radius: 8px;">3</button>
                <button style="background: #1E3A8A; color: white; border: none; padding: 8px 16px; border-radius: 8px;">Next</button>
            </div>
            """,
            unsafe_allow_html=True,
        )

    custom_divider()

    with st.expander("📋 Transaction Details", expanded=False):
        tx_id = st.selectbox("Select Transaction ID", [t["ID"] for t in transactions])
        if tx_id:
            tx = next(t for t in transactions if t["ID"] == tx_id)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Transaction ID:** {tx['ID']}")
                st.markdown(f"**Date:** {tx['Date']}")
                st.markdown(f"**Amount:** {tx['Amount']}")
                st.markdown(f"**Merchant:** {tx['Merchant']}")
                st.markdown(f"**Location:** {tx['Location']}")
            with col2:
                st.markdown(f"**Status:** {tx['Status']}")
                st.markdown(f"**Risk Level:** {tx['Risk']}")
                st.markdown(f"**Fraud:** {'Yes' if tx['Fraud'] else 'No'}")
            if tx['Fraud']:
                info_box(
                    "🚨 This transaction was flagged as fraudulent. "
                    "High transaction amount from an unusual location using a new device.",
                    "danger",
                )
            else:
                info_box("✓ This transaction appears legitimate.", "success")
