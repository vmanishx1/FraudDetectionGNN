import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, classification_report
)
from typing import Dict, Any, Tuple, Optional, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

from utils.logger import logger
from utils.config import MODEL_DIR


class Evaluator:
    def __init__(self):
        self.metrics: Dict[str, Any] = {}

    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "specificity": float(self._specificity(y_true, y_pred)),
            "false_positive_rate": float(self._false_positive_rate(y_true, y_pred)),
            "false_negative_rate": float(self._false_negative_rate(y_true, y_pred)),
            "positive_predictive_value": float(precision_score(y_true, y_pred, zero_division=0)),
            "negative_predictive_value": float(self._npv(y_true, y_pred)),
        }
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
        self.metrics = metrics
        logger.info(f"Evaluation metrics computed: Acc={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}")
        return metrics

    def _specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def _false_positive_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        fp = np.sum((y_true == 0) & (y_pred == 1))
        n = np.sum(y_true == 0)
        return fp / n if n > 0 else 0.0

    def _false_negative_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        fn = np.sum((y_true == 1) & (y_pred == 0))
        p = np.sum(y_true == 1)
        return fn / p if p > 0 else 0.0

    def _npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0

    def plot_confusion_matrix(self, cm: np.ndarray) -> go.Figure:
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Fraud"],
            y=["Actual Normal", "Actual Fraud"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=500,
            height=400,
            template="plotly_white",
        )
        return fig

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.4f})",
                                 line=dict(color="#2563EB", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=500,
            height=400,
            template="plotly_white",
            showlegend=True,
        )
        return fig

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                 fill="tozeroy", name="PR Curve",
                                 line=dict(color="#16A34A", width=3)))
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=500,
            height=400,
            template="plotly_white",
        )
        return fig

    def plot_loss_curve(self, history: Dict[str, list]) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.get("train_loss", []), mode="lines",
                                 name="Train Loss", line=dict(color="#2563EB")))
        fig.add_trace(go.Scatter(y=history.get("val_loss", []), mode="lines",
                                 name="Val Loss", line=dict(color="#DC2626")))
        fig.update_layout(
            title="Loss Curves",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
            showlegend=True,
        )
        return fig

    def plot_accuracy_curve(self, history: Dict[str, list]) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.get("train_acc", []), mode="lines",
                                 name="Train Accuracy", line=dict(color="#2563EB")))
        fig.add_trace(go.Scatter(y=history.get("val_acc", []), mode="lines",
                                 name="Val Accuracy", line=dict(color="#16A34A")))
        fig.update_layout(
            title="Accuracy Curves",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            template="plotly_white",
            showlegend=True,
        )
        return fig

    def plot_metrics_comparison(self, models_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        fig = go.Figure()
        colors = ["#2563EB", "#16A34A", "#DC2626", "#F59E0B", "#8B5CF6"]
        for idx, (model_name, metrics) in enumerate(models_metrics.items()):
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            fig.add_trace(go.Bar(
                name=model_name,
                x=[m.replace("_", " ").title() for m in metrics_to_plot],
                y=values,
                marker_color=colors[idx % len(colors)],
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
            ))
        fig.update_layout(
            title="Model Comparison",
            barmode="group",
            template="plotly_white",
            yaxis_range=[0, 1],
            showlegend=True,
        )
        return fig
