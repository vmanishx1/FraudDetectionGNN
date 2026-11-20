import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.styles.theme import apply_theme
from app.components.ui_components import auth_ui
from app.pages.dashboard import render_dashboard
from app.pages.transactions import render_transactions
from app.pages.graph_view import render_graph_view
from app.pages.predict import render_predict
from app.pages.analytics_page import render_analytics
from app.pages.reports_page import render_reports
from app.pages.settings_page import render_settings
from app.pages.training_page import render_training
from utils.config import APP_NAME


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_theme()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


if not st.session_state.authenticated:
    result = auth_ui()
    if result:
        if result["action"] == "login":
            st.session_state.authenticated = True
            st.session_state.username = result["username"]
            st.success(f"Welcome back, {result['username']}!")
            st.rerun()
        elif result["action"] == "signup":
            st.session_state.authenticated = True
            st.session_state.username = result["username"]
            st.success(f"Account created! Welcome, {result['username']}!")
            st.rerun()
else:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px 0;">
                <h2 style="color: white; margin: 0;">🛡️ FraudDetect</h2>
                <p style="color: #93C5FD; font-size: 12px;">AI-Powered Fraud Detection</p>
                <div style="margin-top: 12px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <p style="color: white; margin: 0; font-size: 14px;">👤 {st.session_state.username}</p>
                    <p style="color: #93C5FD; margin: 0; font-size: 11px;">Fraud Analyst</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        menu_items = {
            "dashboard": {"icon": "📊", "label": "Dashboard"},
            "transactions": {"icon": "💳", "label": "Transactions"},
            "graph": {"icon": "🔗", "label": "Graph View"},
            "predict": {"icon": "🔍", "label": "Fraud Prediction"},
            "analytics": {"icon": "📈", "label": "Analytics"},
            "training": {"icon": "🤖", "label": "Model Training"},
            "reports": {"icon": "📄", "label": "Reports"},
            "settings": {"icon": "⚙️", "label": "Settings"},
        }

        st.markdown("---")
        st.markdown('<p style="color: #93C5FD; font-size: 11px; padding-left: 8px;">NAVIGATION</p>', unsafe_allow_html=True)

        for page_id, info in menu_items.items():
            btn_class = "primary" if st.session_state.current_page == page_id else "secondary"
            if st.button(
                f"{info['icon']}  {info['label']}",
                key=f"nav_{page_id}",
                use_container_width=True,
                type=btn_class,
            ):
                st.session_state.current_page = page_id
                st.rerun()

        st.markdown("---")
        st.markdown('<p style="color: #93C5FD; font-size: 11px; padding-left: 8px;">PREFERENCES</p>', unsafe_allow_html=True)
        st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

        st.markdown("---")
        if st.button("🚪  Logout", key="logout_btn", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown(
            """
            <div style="position: fixed; bottom: 20px; left: 20px; right: 20px; text-align: center;">
                <p style="color: #93C5FD; font-size: 10px;">v1.0.0 | © 2026 FraudDetect</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    page = st.session_state.current_page
    if page == "dashboard":
        render_dashboard()
    elif page == "transactions":
        render_transactions()
    elif page == "graph":
        render_graph_view()
    elif page == "predict":
        render_predict()
    elif page == "analytics":
        render_analytics()
    elif page == "training":
        render_training()
    elif page == "reports":
        render_reports()
    elif page == "settings":
        render_settings()
