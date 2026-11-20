from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, ListFlowable, ListItem,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from datetime import datetime
from pathlib import Path
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sqlalchemy.orm import Session
import tempfile

from utils.logger import logger
from utils.config import REPORT_DIR, APP_NAME
from backend.database.models import Transaction, Prediction, User, ModelHistory


class ReportService:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name="Title1",
            parent=self.styles["Title"],
            fontSize=24,
            textColor=colors.HexColor("#1E3A8A"),
            spaceAfter=12,
        ))
        self.styles.add(ParagraphStyle(
            name="Heading1Custom",
            parent=self.styles["Heading1"],
            textColor=colors.HexColor("#1E3A8A"),
            fontSize=16,
        ))
        self.styles.add(ParagraphStyle(
            name="BodyCustom",
            parent=self.styles["Normal"],
            fontSize=10,
            leading=14,
        ))

    def generate_pdf_report(self, db: Session) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"fraud_detection_report_{timestamp}.pdf"
        filepath = REPORT_DIR / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            title=f"{APP_NAME} - Fraud Detection Report",
            author=APP_NAME,
        )

        story = []
        self._add_header(story)
        self._add_executive_summary(story, db)
        self._add_analytics_section(story, db)
        self._add_model_performance(story, db)
        self._add_recent_predictions(story, db)
        self._add_recommendations(story)
        self._add_footer(story)

        doc.build(story)
        logger.info(f"PDF report generated: {filepath}")
        return filepath

    def _add_header(self, story):
        story.append(Paragraph(f"{APP_NAME}", self.styles["Title1"]))
        story.append(Paragraph(
            f"Fraud Detection Report - {datetime.utcnow().strftime('%B %d, %Y')}",
            self.styles["Heading1Custom"],
        ))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            "This report provides a comprehensive analysis of fraudulent transactions detected "
            "by the AI-powered Graph Neural Network system.",
            self.styles["BodyCustom"],
        ))
        story.append(Spacer(1, 20))

    def _add_executive_summary(self, story, db: Session):
        story.append(Paragraph("Executive Summary", self.styles["Heading1Custom"]))
        story.append(Spacer(1, 8))

        total_tx = db.query(Transaction).count()
        fraud_tx = db.query(Transaction).filter(Transaction.is_fraud == 1).count()
        normal_tx = total_tx - fraud_tx
        fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0

        data = [
            ["Metric", "Value"],
            ["Total Transactions", str(total_tx)],
            ["Fraud Transactions", str(fraud_tx)],
            ["Normal Transactions", str(normal_tx)],
            ["Fraud Rate", f"{fraud_rate:.2f}%"],
        ]
        t = Table(data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        self._generate_pie_chart(fraud_tx, normal_tx, "fraud_distribution.png")
        story.append(Image("fraud_distribution.png", width=300, height=200))
        story.append(Spacer(1, 20))

    def _generate_pie_chart(self, fraud, normal, filename):
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Fraud", "Normal"]
        sizes = [fraud, normal]
        colors_chart = ["#DC2626", "#16A34A"]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors_chart)
        ax.set_title("Transaction Distribution")
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()

    def _add_analytics_section(self, story, db: Session):
        story.append(Paragraph("Transaction Analytics", self.styles["Heading1Custom"]))
        story.append(Spacer(1, 8))

        from sqlalchemy import func
        top_merchants = (
            db.query(Transaction.merchant, func.count(Transaction.id).label("cnt"),
                     func.sum(Transaction.is_fraud).label("fr"))
            .group_by(Transaction.merchant)
            .order_by(func.count(Transaction.id).desc())
            .limit(5)
            .all()
        )

        if top_merchants:
            data = [["Merchant", "Total", "Frauds", "Fraud Rate"]]
            for m in top_merchants:
                rate = (m.fr / m.cnt * 100) if m.cnt > 0 else 0
                data.append([str(m[0] or "Unknown"), str(m.cnt), str(m.fr), f"{rate:.1f}%"])
            t = Table(data, colWidths=[120, 80, 80, 80])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Paragraph("Top Merchants by Transaction Volume:", self.styles["BodyCustom"]))
            story.append(Spacer(1, 6))
            story.append(t)
            story.append(Spacer(1, 20))

    def _add_model_performance(self, story, db: Session):
        story.append(Paragraph("Model Performance", self.styles["Heading1Custom"]))
        story.append(Spacer(1, 8))

        models = db.query(ModelHistory).order_by(ModelHistory.created_at.desc()).all()
        if models:
            data = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]]
            for m in models:
                data.append([
                    m.model_type,
                    f"{m.accuracy:.4f}" if m.accuracy else "N/A",
                    f"{m.precision:.4f}" if m.precision else "N/A",
                    f"{m.recall:.4f}" if m.recall else "N/A",
                    f"{m.f1_score:.4f}" if m.f1_score else "N/A",
                    f"{m.roc_auc:.4f}" if m.roc_auc else "N/A",
                ])
            t = Table(data, colWidths=[80, 70, 70, 70, 70, 70])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
        else:
            story.append(Paragraph("No trained models found in database.", self.styles["BodyCustom"]))
        story.append(Spacer(1, 20))

    def _add_recent_predictions(self, story, db: Session):
        story.append(Paragraph("Recent Predictions", self.styles["Heading1Custom"]))
        story.append(Spacer(1, 8))

        recent = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(10).all()
        if recent:
            data = [["ID", "Prediction", "Probability", "Risk Level", "Timestamp"]]
            for p in recent:
                label = "Fraud" if p.prediction == 1 else "Legitimate"
                ts = p.timestamp.strftime("%Y-%m-%d %H:%M") if p.timestamp else "N/A"
                data.append([str(p.id), label, f"{p.probability:.4f}", p.risk_level or "Unknown", ts])
            t = Table(data, colWidths=[40, 80, 80, 80, 120])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
        else:
            story.append(Paragraph("No predictions found.", self.styles["BodyCustom"]))
        story.append(Spacer(1, 20))

    def _add_recommendations(self, story):
        story.append(Paragraph("Recommendations", self.styles["Heading1Custom"]))
        story.append(Spacer(1, 8))
        recommendations = [
            "Implement real-time monitoring for high-risk transactions.",
            "Review flagged transactions within 24 hours to minimize false positives.",
            "Update model retraining schedule based on new fraud patterns.",
            "Consider multi-factor authentication for high-value transactions.",
            "Regularly audit merchant accounts with high fraud rates.",
        ]
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles["BodyCustom"]))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 20))

    def _add_footer(self, story):
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"Generated by {APP_NAME} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            ParagraphStyle("Footer", parent=self.styles["Normal"], fontSize=8, textColor=colors.gray,
                           alignment=1),
        ))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Bank Officer Signature: ____________________________",
            self.styles["BodyCustom"],
        ))
