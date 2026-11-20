import streamlit as st

THEME_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A8A 0%, #1E40AF 100%);
        color: white;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    [data-testid="stSidebar"] .stSelectbox label {
        color: white;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid #E2E8F0;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin: 8px 0;
    }

    .metric-label {
        font-size: 13px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Status Badges */
    .badge-fraud {
        background: #FEE2E2;
        color: #DC2626;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }

    .badge-safe {
        background: #DCFCE7;
        color: #16A34A;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }

    .badge-warning {
        background: #FEF3C7;
        color: #F59E0B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Info Boxes */
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }

    .warning-box {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }

    .danger-box {
        background: #FEE2E2;
        border-left: 4px solid #DC2626;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }

    .success-box {
        background: #DCFCE7;
        border-left: 4px solid #16A34A;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }

    /* Headers */
    .page-header {
        color: #1E3A8A;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    .section-header {
        color: #334155;
        font-size: 18px;
        font-weight: 600;
        margin: 16px 0 12px 0;
    }

    /* Tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }

    /* Progress Bars */
    .risk-bar {
        height: 8px;
        border-radius: 4px;
    }

    /* Risk Score Display */
    .risk-score {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
    }

    .risk-label {
        font-size: 16px;
        text-align: center;
        font-weight: 600;
    }

    /* Metrics grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.3s ease-out;
    }

    /* Login/Signup */
    .auth-container {
        max-width: 420px;
        margin: 80px auto;
        padding: 40px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .auth-title {
        text-align: center;
        color: #1E3A8A;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .auth-subtitle {
        text-align: center;
        color: #64748B;
        margin-bottom: 32px;
    }

    /* Dark Mode */
    .dark-mode {
        background-color: #0F172A;
        color: #E2E8F0;
    }

    .dark-mode .metric-card {
        background: #1E293B;
        border-color: #334155;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .metrics-grid {
            grid-template-columns: 1fr;
        }
        .metric-value {
            font-size: 20px;
        }
    }

    /* Search Box */
    .search-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 16px;
        width: 100%;
    }

    /* Graph Container */
    .graph-container {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        border: 1px solid #E2E8F0;
    }

    /* Filter Panel */
    .filter-panel {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        border: 1px solid #E2E8F0;
    }

    /* Custom divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
        margin: 24px 0;
    }

    /* Prediction Result */
    .prediction-result {
        text-align: center;
        padding: 32px;
        border-radius: 16px;
        margin: 16px 0;
    }
</style>
"""


DARK_THEME_CSS = """
<style>
    [data-theme="dark"] .stApp {
        background-color: #0F172A;
    }
    [data-theme="dark"] .metric-card {
        background: #1E293B;
        color: #E2E8F0;
        border-color: #334155;
    }
    [data-theme="dark"] .metric-label {
        color: #94A3B8;
    }
    [data-theme="dark"] .page-header {
        color: #60A5FA;
    }
</style>
"""


def apply_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            /* Hide default Streamlit elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
