from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, LargeBinary
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), default="analyst")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    last_login = Column(DateTime)

    transactions = relationship("Transaction", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")
    reports = relationship("Report", back_populates="user")
    login_history = relationship("LoginHistory", back_populates="user")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    amount = Column(Float, nullable=False)
    merchant = Column(String(255))
    merchant_category = Column(String(100))
    location = Column(String(255))
    device = Column(String(255))
    payment_type = Column(String(100))
    customer_age = Column(Integer)
    income = Column(Float)
    credit_score = Column(Integer)
    previous_fraud = Column(Integer, default=0)
    num_transactions = Column(Integer, default=0)
    is_fraud = Column(Integer, default=0)
    timestamp = Column(DateTime, server_default=func.now())
    status = Column(String(50), default="completed")

    user = relationship("User", back_populates="transactions")
    predictions = relationship("Prediction", back_populates="transaction")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_score = Column(Float)
    risk_level = Column(String(50))
    explanation = Column(Text)
    suggested_action = Column(String(255))
    confidence = Column(Float)
    model_used = Column(String(100))
    features_used = Column(JSON)
    timestamp = Column(DateTime, server_default=func.now())

    transaction = relationship("Transaction", back_populates="predictions")
    user = relationship("User", back_populates="predictions")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255))
    report_type = Column(String(100))
    content = Column(Text)
    metrics = Column(JSON)
    file_path = Column(String(500))
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="reports")


class ModelHistory(Base):
    __tablename__ = "model_history"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(100), nullable=False)
    version = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    parameters = Column(JSON)
    file_path = Column(String(500))
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())


class LoginHistory(Base):
    __tablename__ = "login_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    login_time = Column(DateTime, server_default=func.now())
    success = Column(Boolean, default=True)

    user = relationship("User", back_populates="login_history")


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(255))
    details = Column(Text)
    entity_type = Column(String(100))
    entity_id = Column(Integer, nullable=True)
    ip_address = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())
