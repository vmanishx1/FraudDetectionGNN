from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=255)
    full_name: Optional[str] = None

    @validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email address")
        return v


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0)
    merchant: Optional[str] = "Unknown"
    merchant_category: Optional[str] = "General"
    location: Optional[str] = "Unknown"
    device: Optional[str] = "Unknown"
    payment_type: Optional[str] = "Credit"
    customer_age: Optional[int] = 30
    income: Optional[float] = 50000
    credit_score: Optional[int] = 700
    previous_fraud: Optional[int] = 0
    num_transactions: Optional[int] = 10


class PredictionResponse(BaseModel):
    transaction_id: Optional[int]
    prediction: int
    prediction_label: str
    probability: float
    risk_score: float
    risk_level: str
    risk_color: str
    explanation: str
    suggested_action: str
    confidence: float
    timestamp: Optional[datetime] = None


class TransactionResponse(BaseModel):
    id: int
    transaction_id: Optional[str]
    amount: float
    merchant: Optional[str]
    location: Optional[str]
    device: Optional[str]
    payment_type: Optional[str]
    is_fraud: Optional[int]
    timestamp: Optional[datetime]
    status: Optional[str]

    class Config:
        from_attributes = True


class AnalyticsResponse(BaseModel):
    total_transactions: int
    fraud_count: int
    normal_count: int
    fraud_rate: float
    avg_fraud_amount: float
    top_locations: List[Dict[str, Any]]
    top_merchants: List[Dict[str, Any]]
    fraud_by_payment: List[Dict[str, Any]]
    fraud_over_time: List[Dict[str, Any]]
    risk_distribution: Dict[str, int]


class ReportRequest(BaseModel):
    title: str
    report_type: str = "pdf"
    include_graphs: bool = True


class PasswordReset(BaseModel):
    email: str
    new_password: str = Field(..., min_length=6)
    confirm_password: str

    @validator("confirm_password")
    def passwords_match(cls, v, values):
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class ModelMetrics(BaseModel):
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    matthews_corrcoef: Optional[float]
    balanced_accuracy: Optional[float]
