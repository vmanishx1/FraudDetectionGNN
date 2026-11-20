import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from utils.logger import logger
from utils.config import MODEL_DIR
from utils.helpers import calculate_risk_score, generate_explanation, suggested_action
from ml.models.gnn_models import ModelFactory
from ml.explainability.explainer import GNNExplainer, ExplanationGenerator


class FraudPredictor:
    def __init__(self, model_type: str = "gcn", model_path: Optional[Path] = None):
        self.model_type = model_type
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.explainer = None
        self.feature_names = []
        if model_path and model_path.exists():
            self.load_model(model_path)

    def load_model(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model_type = checkpoint.get("model_type", self.model_type)
        in_channels = checkpoint.get("in_channels", 64)
        self.model = ModelFactory.create_model(
            model_type=self.model_type, in_channels=in_channels
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.explainer = GNNExplainer(self.model, self.device)
        logger.info(f"Model loaded from {path}")

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            return self._rule_based_prediction(features)

        feature_tensor = self._prepare_features(features)
        with torch.no_grad():
            feature_tensor = feature_tensor.to(self.device)
            output = self.model(feature_tensor, torch.zeros((2, 0), dtype=torch.long).to(self.device))
            prob = torch.softmax(output, dim=1)[0, 1].item()

        prediction = 1 if prob >= 0.5 else 0
        risk = calculate_risk_score(prob)
        explanation_text = generate_explanation(features)
        action = suggested_action(risk["level"])

        return {
            "prediction": prediction,
            "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
            "probability": round(prob, 4),
            "risk_score": risk["score"],
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "explanation": explanation_text,
            "suggested_action": action,
            "confidence": round(abs(prob - 0.5) * 2, 4),
        }

    def _rule_based_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        risk_score = 0.0
        reasons = []
        if features.get("amount", 0) > 10000:
            risk_score += 0.3
            reasons.append("High transaction amount")
        if features.get("previous_fraud", 0) == 1:
            risk_score += 0.3
            reasons.append("Previous fraud history")
        if features.get("credit_score", 700) < 500:
            risk_score += 0.2
            reasons.append("Low credit score")
        if features.get("new_device", 0) == 1:
            risk_score += 0.15
            reasons.append("New device")

        prob = min(risk_score, 0.95)
        prediction = 1 if prob >= 0.5 else 0
        risk = calculate_risk_score(prob)
        explanation_text = ", ".join(reasons) if reasons else "Transaction appears normal"
        action = suggested_action(risk["level"])

        return {
            "prediction": prediction,
            "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
            "probability": round(prob, 4),
            "risk_score": risk["score"],
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "explanation": f"Rule-based: {explanation_text}",
            "suggested_action": action,
            "confidence": round(abs(prob - 0.5) * 2, 4),
        }

    def _prepare_features(self, features: Dict[str, Any]) -> torch.Tensor:
        default_features = [
            features.get("amount", 0) / 10000,
            features.get("age", 30) / 100,
            features.get("income", 50000) / 200000,
            features.get("credit_score", 700) / 850,
            float(features.get("previous_fraud", 0)),
            features.get("num_transactions", 10) / 100,
            float(features.get("new_device", 0)),
            float(features.get("new_merchant", 0)),
            float(features.get("location_mismatch", 0)),
            float(features.get("high_velocity", 0)),
        ]
        return torch.tensor([default_features], dtype=torch.float)

    def batch_predict(self, features_list: list) -> list:
        return [self.predict_single(f) for f in features_list]
