import streamlit as st

from app.components.ui_components import section_header, page_header, custom_divider, info_box


def render_settings():
    page_header("Settings")

    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Profile", "🔐 Security", "🎨 Appearance", "⚙️ System"
    ])

    with tab1:
        section_header("Profile Information")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Full Name", value="John Doe")
            st.text_input("Email", value="john.doe@bank.com")
        with col2:
            st.text_input("Username", value="johndoe")
            st.text_input("Role", value="Fraud Analyst", disabled=True)
        if st.button("Update Profile", type="primary"):
            st.success("Profile updated successfully!")

        custom_divider()
        section_header("Preferences")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Default Dashboard View", ["Overview", "Analytics", "Transactions"])
            st.number_input("Items per Page", min_value=10, max_value=100, value=50, step=10)
        with col2:
            st.selectbox("Notification Frequency", ["Real-time", "Hourly", "Daily", "Weekly"])
            st.multiselect("Email Alerts", ["Fraud Detected", "High Risk", "Model Update", "Report Ready"],
                           default=["Fraud Detected", "High Risk"])

    with tab2:
        section_header("Security Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Current Password", type="password")
            st.text_input("New Password", type="password")
        with col2:
            st.text_input("Confirm New Password", type="password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Change Password", type="primary"):
                st.success("Password changed successfully!")

        custom_divider()
        section_header("Session Management")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Remember Me", value=True)
            st.number_input("Session Timeout (minutes)", min_value=5, max_value=480, value=60)
        with col2:
            st.checkbox("Two-Factor Authentication", value=False)
            st.button("Revoke All Sessions", type="secondary")

        custom_divider()
        section_header("Login History")
        st.markdown("""
        - **Current Session:** Active since 2026-07-13 10:30 AM
        - **Last Login:** 2026-07-13 09:15 AM from 192.168.1.1
        - **Previous:** 2026-07-12 from 10.0.0.1 (Chrome on Windows)
        - **Failed Attempts:** 0
        """)

    with tab3:
        section_header("Theme Settings")
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("Theme", ["Light", "Dark", "System Default"])
            st.color_picker("Primary Color", "#1E3A8A")
        with col2:
            st.selectbox("Layout", ["Default", "Compact", "Wide"])
            st.selectbox("Font Size", ["Small", "Medium", "Large"])
        if st.button("Apply Theme", type="primary"):
            st.success("Theme applied!")

        custom_divider()
        section_header("Dashboard Widgets")
        st.multiselect(
            "Visible Widgets",
            ["Quick Stats", "Fraud Alerts", "Performance Charts", "Recent Predictions",
             "Risk Overview", "Model Comparison", "Transaction Feed"],
            default=["Quick Stats", "Fraud Alerts", "Performance Charts", "Recent Predictions", "Risk Overview"],
        )

    with tab4:
        section_header("System Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Model Auto-Train", ["Disabled", "Daily", "Weekly", "Monthly"])
            st.slider("Fraud Alert Threshold", 0, 100, 70)
        with col2:
            st.number_input("Max Batch Size", min_value=16, max_value=512, value=64, step=16)
            st.slider("Risk Score Sensitivity", 0.0, 1.0, 0.5)

        custom_divider()
        section_header("Data Management")
        col1, col2 = st.columns(2)
        with col1:
            st.button("🗑 Clear Cache", use_container_width=True)
            st.button("📤 Export All Data", use_container_width=True)
        with col2:
            st.button("🔄 Reset to Defaults", use_container_width=True)
            info_box("Exporting data may take a few minutes for large datasets.", "info")
