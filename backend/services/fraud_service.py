import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from pathlib import Path
import random

from utils.logger import logger
from utils.config import MODEL_DIR
from utils.helpers import calculate_risk_score, generate_explanation, suggested_action
from backend.database.models import Transaction, Prediction, User, ModelHistory, ActivityLog
from ml.inference.predictor import FraudPredictor
from ml.evaluation.evaluator import Evaluator


class FraudService:
    def __init__(self):
        self.predictor = FraudPredictor()
        self.evaluator = Evaluator()
        self._load_best_model()

    def _load_best_model(self):
        model_files = list(MODEL_DIR.glob("*_best.pt"))
        if model_files:
            self.predictor.load_model(model_files[0])
            logger.info(f"Loaded model: {model_files[0]}")
        else:
            logger.warning("No trained model found, using rule-based fallback")

    def predict_transaction(self, transaction_data: Dict[str, Any], db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
        result = self.predictor.predict_single(transaction_data)
        tx = Transaction(
            amount=transaction_data.get("amount", 0),
            merchant=transaction_data.get("merchant", "Unknown"),
            location=transaction_data.get("location", "Unknown"),
            device=transaction_data.get("device", "Unknown"),
            payment_type=transaction_data.get("payment_type", "Credit"),
            customer_age=transaction_data.get("customer_age", 30),
            income=transaction_data.get("income", 50000),
            credit_score=transaction_data.get("credit_score", 700),
            previous_fraud=transaction_data.get("previous_fraud", 0),
            num_transactions=transaction_data.get("num_transactions", 10),
            is_fraud=result["prediction"],
            status="flagged" if result["prediction"] == 1 else "completed",
        )
        db.add(tx)
        db.flush()

        pred = Prediction(
            transaction_id=tx.id,
            user_id=user_id,
            prediction=result["prediction"],
            probability=result["probability"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            explanation=result["explanation"],
            suggested_action=result["suggested_action"],
            confidence=result["confidence"],
            model_used=self.predictor.model_type if self.predictor.model else "rule-based",
        )
        db.add(pred)
        self._log_activity(db, user_id, "prediction", f"Transaction predicted as {result['prediction_label']}", tx.id)
        db.commit()
        db.refresh(tx)
        db.refresh(pred)

        result["transaction_id"] = tx.id
        result["timestamp"] = datetime.utcnow().isoformat()
        return result

    def get_analytics(self, db: Session) -> Dict[str, Any]:
        total = db.query(Transaction).count()
        fraud_count = db.query(Transaction).filter(Transaction.is_fraud == 1).count()
        normal_count = total - fraud_count
        fraud_rate = fraud_count / total if total > 0 else 0
        avg_fraud_amount = (
            db.query(Transaction).filter(Transaction.is_fraud == 1)
            .with_entities(db.query(Transaction.amount).filter(Transaction.is_fraud == 1).subquery())
        )
        avg_amount = (
            db.query(Transaction.amount).filter(Transaction.is_fraud == 1).all()
        )
        avg_fraud_amt = np.mean([a[0] for a in avg_amount]) if avg_amount else 0

        top_locations = self._get_top_locations(db)
        top_merchants = self._get_top_merchants(db)
        fraud_by_payment = self._get_fraud_by_payment(db)
        fraud_over_time = self._get_fraud_over_time(db)

        risk_dist = {"safe": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
        predictions = db.query(Prediction).all()
        for p in predictions:
            level = p.risk_level.lower() if p.risk_level else "safe"
            if level in risk_dist:
                risk_dist[level] += 1

        return {
            "total_transactions": total,
            "fraud_count": fraud_count,
            "normal_count": normal_count,
            "fraud_rate": round(fraud_rate, 4),
            "avg_fraud_amount": round(float(avg_fraud_amt), 2),
            "top_locations": top_locations,
            "top_merchants": top_merchants,
            "fraud_by_payment": fraud_by_payment,
            "fraud_over_time": fraud_over_time,
            "risk_distribution": risk_dist,
        }

    def _get_top_locations(self, db: Session, limit: int = 10) -> List[Dict[str, Any]]:
        from sqlalchemy import func
        results = (
            db.query(Transaction.location, func.count(Transaction.id).label("count"), func.sum(Transaction.is_fraud).label("frauds"))
            .group_by(Transaction.location)
            .order_by(func.count(Transaction.id).desc())
            .limit(limit)
            .all()
        )
        return [{"location": r[0] or "Unknown", "count": r[1], "frauds": r[2] or 0} for r in results]

    def _get_top_merchants(self, db: Session, limit: int = 10) -> List[Dict[str, Any]]:
        from sqlalchemy import func
        results = (
            db.query(Transaction.merchant, func.count(Transaction.id).label("count"), func.sum(Transaction.is_fraud).label("frauds"))
            .group_by(Transaction.merchant)
            .order_by(func.count(Transaction.id).desc())
            .limit(limit)
            .all()
        )
        return [{"merchant": r[0] or "Unknown", "count": r[1], "frauds": r[2] or 0} for r in results]

    def _get_fraud_by_payment(self, db: Session) -> List[Dict[str, Any]]:
        from sqlalchemy import func
        results = (
            db.query(Transaction.payment_type, func.count(Transaction.id).label("count"), func.sum(Transaction.is_fraud).label("frauds"))
            .group_by(Transaction.payment_type)
            .all()
        )
        return [{"payment_type": r[0] or "Unknown", "count": r[1], "frauds": r[2] or 0} for r in results]

    def _get_fraud_over_time(self, db: Session, days: int = 30) -> List[Dict[str, Any]]:
        from sqlalchemy import func, cast, Date
        cutoff = datetime.utcnow() - timedelta(days=days)
        results = (
            db.query(
                cast(Transaction.timestamp, Date).label("date"),
                func.count(Transaction.id).label("total"),
                func.sum(Transaction.is_fraud).label("frauds"),
            )
            .filter(Transaction.timestamp >= cutoff)
            .group_by(cast(Transaction.timestamp, Date))
            .order_by(cast(Transaction.timestamp, Date))
            .all()
        )
        return [{"date": str(r[0]), "total": r[1], "frauds": r[2] or 0} for r in results]

    def search_transactions(
        self, db: Session, query: Optional[str] = None, status: Optional[str] = None,
        min_amount: Optional[float] = None, max_amount: Optional[float] = None,
        limit: int = 100, offset: int = 0
    ) -> Tuple[List[Transaction], int]:
        q = db.query(Transaction)
        if query:
            q = q.filter(
                Transaction.transaction_id.ilike(f"%{query}%")
                | Transaction.merchant.ilike(f"%{query}%")
                | Transaction.location.ilike(f"%{query}%")
            )
        if status:
            q = q.filter(Transaction.status == status)
        if min_amount is not None:
            q = q.filter(Transaction.amount >= min_amount)
        if max_amount is not None:
            q = q.filter(Transaction.amount <= max_amount)
        total = q.count()
        transactions = q.order_by(Transaction.timestamp.desc()).offset(offset).limit(limit).all()
        return transactions, total

    def get_model_metrics(self, db: Session) -> List[Dict[str, Any]]:
        models = db.query(ModelHistory).order_by(ModelHistory.created_at.desc()).all()
        return [
            {
                "id": m.id,
                "model_type": m.model_type,
                "version": m.version,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "roc_auc": m.roc_auc,
                "is_active": m.is_active,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in models
        ]

    def _log_activity(self, db: Session, user_id: Optional[int], action: str, details: str, entity_id: Optional[int] = None):
        log = ActivityLog(
            user_id=user_id,
            action=action,
            details=details,
            entity_type="transaction",
            entity_id=entity_id,
        )
        db.add(log)
