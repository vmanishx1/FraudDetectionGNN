import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")
SECRET_KEY = os.getenv("SECRET_KEY", "fraud-detection-gnn-secret-key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "saved_models"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "datasets"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", BASE_DIR / "reports"))

APP_NAME = os.getenv("APP_NAME", "FraudDetectionGNN")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", "10000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "128"))
DROPOUT = float(os.getenv("DROPOUT", "0.3"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THEME = {
    "primary": "#1E3A8A",
    "secondary": "#2563EB",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "background": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "border_radius": "12px",
}
