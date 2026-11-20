import pytest
from utils.helpers import (
    calculate_risk_score, generate_explanation, suggested_action,
    format_currency, format_percentage,
)


class TestHelpers:
    def test_calculate_risk_score(self):
        assert calculate_risk_score(0.9)["level"] == "Critical"
        assert calculate_risk_score(0.7)["level"] == "High"
        assert calculate_risk_score(0.5)["level"] == "Medium"
        assert calculate_risk_score(0.3)["level"] == "Low"
        assert calculate_risk_score(0.1)["level"] == "Safe"

    def test_risk_score_color(self):
        assert calculate_risk_score(0.9)["color"] == "#DC2626"
        assert calculate_risk_score(0.1)["color"] == "#16A34A"

    def test_generate_explanation_high_risk(self):
        features = {
            "amount": 0.8, "suspicious_device": 0.7,
            "new_merchant": 0.6, "high_velocity": 0.4,
        }
        explanation = generate_explanation(features)
        assert "High transaction amount" in explanation
        assert "suspicious device" in explanation

    def test_generate_explanation_safe(self):
        features = {"amount": 0.1, "credit_score": 0.9}
        explanation = generate_explanation(features)
        assert "normal" in explanation.lower()

    def test_suggested_action(self):
        assert "block" in suggested_action("Critical").lower()
        assert "normal" in suggested_action("Safe").lower()

    def test_format_currency(self):
        assert format_currency(1234.5) == "$1,234.50"

    def test_format_percentage(self):
        assert "50.00" in format_percentage(50)
