from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from backend.database.connection import get_db
from backend.database.models import User, Transaction, Prediction, Report, ModelHistory, LoginHistory
from backend.auth.jwt_handler import (
    verify_password, get_password_hash, create_access_token, get_current_user,
)
from backend.models.schemas import (
    UserCreate, UserLogin, TokenResponse, TransactionInput, PredictionResponse,
    TransactionResponse, AnalyticsResponse, ReportRequest, PasswordReset,
)
from backend.services.fraud_service import FraudService
from utils.logger import logger
from utils.config import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/api")
fraud_service = FraudService()


@router.post("/auth/register", response_model=TokenResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already exists")

    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(data={"sub": user.username, "user_id": user.id})
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,
    )


@router.post("/auth/login", response_model=TokenResponse)
def login(login_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == login_data.username).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user.last_login = datetime.utcnow()
    db.add(LoginHistory(user_id=user.id, success=True))
    db.commit()

    token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,
    )


@router.post("/auth/logout")
def logout(current_user: User = Depends(get_current_user)):
    return {"message": "Logged out successfully"}


@router.post("/auth/forgot-password")
def forgot_password(email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")
    return {"message": "Password reset instructions sent to email"}


@router.post("/predict", response_model=PredictionResponse)
def predict_transaction(
    transaction: TransactionInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = fraud_service.predict_transaction(transaction.dict(), db, current_user.id)
    return PredictionResponse(**result)


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return fraud_service.get_analytics(db)


@router.get("/transactions", response_model=List[TransactionResponse])
def get_transactions(
    query: Optional[str] = None,
    status: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    transactions, total = fraud_service.search_transactions(
        db, query, status, min_amount, max_amount, limit, offset
    )
    return transactions


@router.get("/transactions/{transaction_id}", response_model=TransactionResponse)
def get_transaction(transaction_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction


@router.get("/predictions")
def get_predictions(
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    predictions = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": p.id,
            "transaction_id": p.transaction_id,
            "prediction": p.prediction,
            "probability": p.probability,
            "risk_level": p.risk_level,
            "explanation": p.explanation,
            "timestamp": p.timestamp.isoformat() if p.timestamp else None,
        }
        for p in predictions
    ]


@router.get("/models/metrics")
def get_model_metrics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return fraud_service.get_model_metrics(db)


@router.post("/models/train")
def train_models(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from ml.pipeline import FraudDetectionPipeline
    pipeline = FraudDetectionPipeline()
    results = pipeline.run_full_pipeline()
    for model_type, metrics in results["all_metrics"].items():
        model_history = ModelHistory(
            model_type=model_type,
            version=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1"),
            roc_auc=metrics.get("roc_auc"),
            is_active=(model_type.lower() == results["best_model"]),
        )
        db.add(model_history)
    db.commit()
    return {"message": "Models trained successfully", "results": results}


@router.get("/reports")
def get_reports(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    reports = db.query(Report).order_by(Report.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "report_type": r.report_type,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reports
    ]


@router.post("/reports/generate")
def generate_report(
    report_request: ReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from backend.services.report_service import ReportService
    report_service = ReportService()
    report_path = report_service.generate_pdf_report(db)
    report = Report(
        user_id=current_user.id,
        title=report_request.title,
        report_type=report_request.report_type,
        file_path=str(report_path),
    )
    db.add(report)
    db.commit()
    return {"message": "Report generated", "file_path": str(report_path)}


@router.get("/graph-data")
def get_graph_data(current_user: User = Depends(get_current_user)):
    from ml.pipeline import FraudDetectionPipeline
    pipeline = FraudDetectionPipeline()
    pipeline.run_full_pipeline()
    graph_data = pipeline.get_graph_data()
    if graph_data is None:
        return {"nodes": [], "edges": []}
    return graph_data


@router.get("/activity")
def get_activity_logs(limit: int = 50, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    logs = db.query(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(limit).all()
    return [
        {
            "id": log.id,
            "action": log.action,
            "details": log.details,
            "entity_type": log.entity_type,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]


@router.get("/dashboard/summary")
def get_dashboard_summary(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    analytics = fraud_service.get_analytics(db)
    recent_predictions = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(10)
        .all()
    )
    return {
        "analytics": analytics,
        "recent_predictions": [
            {
                "id": p.id,
                "prediction": p.prediction,
                "probability": p.probability,
                "risk_level": p.risk_level,
                "timestamp": p.timestamp.isoformat() if p.timestamp else None,
            }
            for p in recent_predictions
        ],
    }
