import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

def save_json(data: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: Path) -> Any:
    with open(filepath, "r") as f:
        return json.load(f)

def save_pickle(data: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load_pickle(filepath: Path) -> Any:
    with open(filepath, "rb") as f:
        return pickle.load(f)

def hash_id(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()[:16]

def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    return f"{value:.2f}%"

def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def calculate_risk_score(probability: float) -> Dict[str, Any]:
    if probability >= 0.8:
        level = "Critical"
        color = "#DC2626"
    elif probability >= 0.6:
        level = "High"
        color = "#F59E0B"
    elif probability >= 0.4:
        level = "Medium"
        color = "#F97316"
    elif probability >= 0.2:
        level = "Low"
        color = "#2563EB"
    else:
        level = "Safe"
        color = "#16A34A"
    return {"level": level, "color": color, "score": round(probability * 100, 2)}

def generate_explanation(features: Dict[str, float]) -> str:
    reasons = []
    if features.get("amount", 0) > 0.6:
        reasons.append("High transaction amount")
    if features.get("suspicious_device", 0) > 0.5:
        reasons.append("suspicious device")
    if features.get("new_merchant", 0) > 0.5:
        reasons.append("unusual/new merchant")
    if features.get("high_velocity", 0) > 0.5:
        reasons.append("high transaction velocity")
    if features.get("location_mismatch", 0) > 0.5:
        reasons.append("location mismatch")
    if features.get("previous_fraud", 0) > 0.5:
        reasons.append("previous fraud history")
    if features.get("credit_score", 0) < 0.3:
        reasons.append("low credit score")
    if not reasons:
        return "Transaction appears normal with no significant risk factors."
    return f"High transaction amount combined with {', '.join(reasons)} increased fraud probability."

def suggested_action(risk_level: str) -> str:
    actions = {
        "Critical": "Immediately block transaction and flag account for review. Notify security team.",
        "High": "Flag for manual review. Require additional authentication from customer.",
        "Medium": "Monitor transaction closely. Apply step-up authentication.",
        "Low": "Allow transaction with standard monitoring.",
        "Safe": "Process transaction normally.",
    }
    return actions.get(risk_level, "Process normally.")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
