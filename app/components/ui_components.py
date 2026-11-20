import streamlit as st
from typing import Dict, Any, Optional, List
from utils.config import THEME


def metric_card(label: str, value: str, delta: Optional[str] = None, color: str = "#1E3A8A"):
    st.markdown(
        f"""
        <div class="metric-card fade-in">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            {f'<div style="font-size: 12px; color: #64748B;">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def fraud_badge(is_fraud: bool):
    if is_fraud:
        return '<span class="badge-fraud">⚠ Fraud</span>'
    return '<span class="badge-safe">✓ Legitimate</span>'


def risk_badge(level: str):
    colors = {
        "Critical": "badge-fraud",
        "High": "badge-warning",
        "Medium": "badge-warning",
        "Low": "badge-safe",
        "Safe": "badge-safe",
    }
    css_class = colors.get(level, "badge-safe")
    return f'<span class="{css_class}">{level}</span>'


def info_box(message: str, type: str = "info"):
    box_class = {
        "info": "info-box",
        "warning": "warning-box",
        "danger": "danger-box",
        "success": "success-box",
    }.get(type, "info-box")
    st.markdown(f'<div class="{box_class}">{message}</div>', unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def page_header(title: str):
    st.markdown(f'<div class="page-header">{title}</div>', unsafe_allow_html=True)


def custom_divider():
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def sidebar_menu():
    menu_items = {
        "Dashboard": {"icon": "📊", "id": "dashboard"},
        "Transactions": {"icon": "💳", "id": "transactions"},
        "Graph View": {"icon": "🔗", "id": "graph"},
        "Fraud Prediction": {"icon": "🔍", "id": "predict"},
        "Analytics": {"icon": "📈", "id": "analytics"},
        "Reports": {"icon": "📄", "id": "reports"},
        "Settings": {"icon": "⚙️", "id": "settings"},
    }

    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px 0;">
                <h2 style="color: white; margin: 0;">🛡️ FraudDetect</h2>
                <p style="color: #93C5FD; font-size: 12px; margin: 4px 0 0 0;">AI-Powered Fraud Detection</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        selected = None
        for label, info in menu_items.items():
            if st.button(
                f"{info['icon']}  {label}",
                key=f"menu_{info['id']}",
                use_container_width=True,
            ):
                selected = info["id"]

        st.markdown("---")

        if st.button("🚪  Logout", key="menu_logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown(
            f"""
            <div style="position: fixed; bottom: 20px; left: 20px; right: 20px; text-align: center;">
                <p style="color: #93C5FD; font-size: 10px;">v1.0.0 | © 2026 FraxSecure</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected


def auth_ui():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class="auth-container">
                <div class="auth-title">🛡️ FraudDetect</div>
                <div class="auth-subtitle">AI-Powered Fraud Detection System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    remember = st.checkbox("Remember me")
                with col_b:
                    st.markdown(
                        '<div style="text-align: right; padding-top: 4px;">'
                        '<a href="#" style="color: #2563EB; text-decoration: none; font-size: 13px;">Forgot password?</a>'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                if submitted:
                    if username and password:
                        return {"action": "login", "username": username, "password": password}
                    else:
                        st.error("Please fill in all fields")

        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_email = st.text_input("Email", placeholder="Enter your email")
                new_password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
                full_name = st.text_input("Full Name (optional)", placeholder="Enter your full name")
                submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif new_username and new_email and new_password:
                        return {
                            "action": "signup",
                            "username": new_username,
                            "email": new_email,
                            "password": new_password,
                            "full_name": full_name,
                        }
                    else:
                        st.error("Please fill in required fields")
    return None


def notification_badge(count: int):
    if count > 0:
        st.markdown(
            f"""
            <span style="
                background: #DC2626;
                color: white;
                padding: 2px 8px;
                border-radius: 9999px;
                font-size: 11px;
                font-weight: 700;
                margin-left: 4px;
            ">{min(count, 99)}{'+' if count > 99 else ''}</span>
            """,
            unsafe_allow_html=True,
        )
