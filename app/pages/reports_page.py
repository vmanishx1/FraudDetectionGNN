import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np

from app.components.ui_components import section_header, page_header, custom_divider, info_box


def render_reports():
    page_header("Reports")

    tab1, tab2 = st.tabs(["📄 Generate Report", "📚 Report History"])

    with tab1:
        st.markdown(
            """
            <div class="info-box">
                Generate a professional PDF report with comprehensive fraud analysis,
                model performance metrics, transaction analytics, and recommendations.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            report_title = st.text_input("Report Title", "Fraud Detection Analysis Report")
            date_range = st.date_input(
                "Report Period",
                [datetime(2026, 1, 1), datetime.now()],
            )
            model_select = st.selectbox(
                "Model to Include", ["Best Model (Auto)", "GCN", "GraphSAGE", "GAT", "All Models"]
            )

        with col2:
            include_sections = st.multiselect(
                "Include Sections",
                ["Executive Summary", "Transaction Analytics", "Model Performance",
                 "Fraud Analysis", "Graph Visualization", "Recommendations"],
                default=["Executive Summary", "Transaction Analytics", "Model Performance",
                         "Fraud Analysis", "Recommendations"],
            )
            report_format = st.selectbox("Report Format", ["PDF", "HTML", "CSV"])
            include_graphs = st.checkbox("Include Graphs and Charts", value=True)

        custom_divider()

        st.subheader("📋 Report Preview")
        st.markdown(f"""
        - **Title:** {report_title}
        - **Period:** {date_range[0]} to {date_range[1]}
        - **Model:** {model_select}
        - **Format:** {report_format}
        - **Sections:** {', '.join(include_sections)}
        - **Graphs:** {'Yes' if include_graphs else 'No'}
        """)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📄 Generate Report", use_container_width=True, type="primary"):
                with st.spinner("Generating report..."):
                    st.success("✅ Report generated successfully!")
                    st.download_button(
                        "⬇ Download PDF Report",
                        data=b"PDF content placeholder",
                        file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                    )

    with tab2:
        section_header("Previous Reports")
        reports_data = []
        for i in range(10):
            reports_data.append({
                "Report ID": f"RPT-{1000+i}",
                "Title": f"Fraud Analysis Report {datetime.now().strftime('%b %Y')}",
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Type": ["Daily", "Weekly", "Monthly"][i % 3],
                "Pages": np.random.randint(5, 20),
                "Status": "Completed",
            })

        df_reports = pd.DataFrame(reports_data)

        for _, report in df_reports.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([1.5, 2, 1, 0.8, 1, 1.5])
            with col1:
                st.markdown(f"**{report['Report ID']}**")
            with col2:
                st.markdown(report['Title'])
            with col3:
                st.markdown(report['Date'])
            with col4:
                st.markdown(report['Type'])
            with col5:
                st.markdown(f"{report['Pages']} pages")
            with col6:
                if st.button(f"⬇ Download", key=f"dl_{i}", use_container_width=True):
                    st.info(f"Downloading {report['Report ID']}...")
            st.markdown('<hr style="margin: 4px 0; opacity: 0.2;">', unsafe_allow_html=True)
