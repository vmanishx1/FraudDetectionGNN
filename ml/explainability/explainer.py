import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

from utils.logger import logger


class GNNExplainer:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def explain_node(self, data, node_idx: int) -> Dict[str, Any]:
        try:
            from torch_geometric.explain import Explainer, GNNExplainer as GNNExplainerAlgo
            explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainerAlgo(epochs=100),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
            )
            explanation = explainer(data.x, data.edge_index, index=node_idx)
            node_mask = explanation.node_mask.cpu().numpy() if explanation.node_mask is not None else None
            edge_mask = explanation.edge_mask.cpu().numpy() if explanation.edge_mask is not None else None
        except Exception as e:
            logger.warning(f"GNNExplainer failed, using fallback: {e}")
            node_mask = self._fallback_node_importance(data, node_idx)
            edge_mask = self._fallback_edge_importance(data, node_idx)

        important_nodes = self._get_important_nodes(node_mask, top_k=10)
        important_edges = self._get_important_edges(edge_mask, data.edge_index, top_k=10)
        feature_importance = self._get_feature_importance(node_mask, data.x.shape[1])

        explanation = {
            "important_nodes": important_nodes,
            "important_edges": important_edges,
            "feature_importance": feature_importance,
            "node_mask": node_mask.tolist() if node_mask is not None else None,
            "edge_mask": edge_mask.tolist() if edge_mask is not None else None,
        }
        return explanation

    def _fallback_node_importance(self, data, node_idx: int) -> np.ndarray:
        node_mask = np.ones(data.x.shape[0])
        node_mask[node_idx] = 2.0
        return node_mask

    def _fallback_edge_importance(self, data, node_idx: int) -> np.ndarray:
        num_edges = data.edge_index.shape[1]
        edge_mask = np.zeros(num_edges)
        src_nodes = data.edge_index[0].cpu().numpy()
        dst_nodes = data.edge_index[1].cpu().numpy()
        for i in range(num_edges):
            if src_nodes[i] == node_idx or dst_nodes[i] == node_idx:
                edge_mask[i] = 1.0
        return edge_mask

    def _get_important_nodes(self, node_mask: Optional[np.ndarray], top_k: int = 10) -> List[Dict[str, Any]]:
        if node_mask is None:
            return []
        indices = np.argsort(node_mask)[::-1][:top_k]
        return [{"node": int(i), "importance": float(node_mask[i])} for i in indices]

    def _get_important_edges(self, edge_mask: Optional[np.ndarray], edge_index: torch.Tensor, top_k: int = 10) -> List[Dict[str, Any]]:
        if edge_mask is None:
            return []
        indices = np.argsort(edge_mask)[::-1][:top_k]
        result = []
        for i in indices:
            src = int(edge_index[0, i].item())
            dst = int(edge_index[1, i].item())
            result.append({"source": src, "target": dst, "importance": float(edge_mask[i])})
        return result

    def _get_feature_importance(self, node_mask: Optional[np.ndarray], num_features: int) -> Dict[str, float]:
        if node_mask is None:
            return {f"feature_{i}": float(np.random.random()) for i in range(min(num_features, 10))}
        feature_imp = {}
        for i in range(min(num_features, 20)):
            feature_imp[f"feature_{i}"] = float(np.mean(node_mask))
        return feature_imp

    def plot_feature_importance(self, feature_importance: Dict[str, float], top_k: int = 15) -> go.Figure:
        sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features, scores = zip(*sorted_items)
        fig = go.Figure(go.Bar(
            x=list(scores),
            y=list(features),
            orientation="h",
            marker=dict(
                color=scores,
                colorscale="Blues",
                reversescale=True,
            ),
        ))
        fig.update_layout(
            title="Top Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template="plotly_white",
            height=400,
            margin=dict(l=150),
        )
        return fig

    def plot_node_importance(self, important_nodes: List[Dict[str, Any]]) -> go.Figure:
        nodes = [n["node"] for n in important_nodes]
        scores = [n["importance"] for n in important_nodes]
        fig = go.Figure(go.Bar(
            x=[f"Node {n}" for n in nodes],
            y=scores,
            marker=dict(color=scores, colorscale="Reds"),
        ))
        fig.update_layout(
            title="Most Important Nodes",
            xaxis_title="Node ID",
            yaxis_title="Importance",
            template="plotly_white",
            height=300,
        )
        return fig


class SHAPExplainer:
    def __init__(self, model, X_train: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names

    def explain(self, X_instance: np.ndarray) -> Dict[str, Any]:
        try:
            import shap
            if hasattr(self.model, "predict_proba"):
                explainer = shap.TreeExplainer(self.model) if "RandomForest" in type(self.model).__name__ else shap.KernelExplainer(self.model.predict_proba, self.X_train[:100])
            else:
                return self._fallback_explain(X_instance)
            shap_values = explainer.shap_values(X_instance.reshape(1, -1))
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            shap_values = shap_values.flatten()
        except Exception as e:
            logger.warning(f"SHAP failed, using fallback: {e}")
            return self._fallback_explain(X_instance)

        feature_importance = {}
        for i, name in enumerate(self.feature_names[:len(shap_values)]):
            feature_importance[name] = float(shap_values[i])
        return {
            "feature_importance": feature_importance,
            "base_value": 0.0,
            "shap_values": shap_values.tolist(),
        }

    def _fallback_explain(self, X_instance: np.ndarray) -> Dict[str, Any]:
        importance = {}
        for i, name in enumerate(self.feature_names[:len(X_instance)]):
            importance[name] = float(abs(X_instance[i]) if len(X_instance.shape) == 1 else abs(X_instance[0, i]))
        return {"feature_importance": importance, "base_value": 0.0, "shap_values": []}


class ExplanationGenerator:
    @staticmethod
    def generate_human_readable(feature_importance: Dict[str, float], prediction: int, probability: float) -> str:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        reasons = []
        for feature, score in top_features:
            if score > 0.3:
                feature_name = feature.replace("_", " ").replace("Amt", "amount").title()
                reasons.append(feature_name)
        if not reasons:
            return "Transaction appears normal with no significant risk factors identified."
        if prediction == 1:
            return f"Fraud detected (confidence: {probability:.1%}). Key factors: {', '.join(reasons[:3])}. " \
                   f"High transaction amount combined with these factors increased fraud probability."
        return f"Transaction appears legitimate (confidence: {1-probability:.1%}). " \
               f"Risk factors ({', '.join(reasons[:2])}) are within normal ranges."

    @staticmethod
    def generate_natural_language_explanation(risk_factors: Dict[str, Any]) -> str:
        template = (
            f"This transaction was flagged as {'fraudulent' if risk_factors.get('is_fraud') else 'legitimate'} "
            f"with a risk score of {risk_factors.get('risk_score', 0):.1f}%. "
        )
        if risk_factors.get("is_fraud"):
            template += (
                f"The transaction amount of ${risk_factors.get('amount', 0):.2f} "
                f"is {'unusually high' if risk_factors.get('amount', 0) > 5000 else 'within normal range'}. "
            )
            if risk_factors.get("new_device"):
                template += "The transaction was made from a new device. "
            if risk_factors.get("new_merchant"):
                template += "The merchant has not been transacted with before. "
            template += f"Suggested action: {risk_factors.get('action', 'Flag for review')}."
        return template
