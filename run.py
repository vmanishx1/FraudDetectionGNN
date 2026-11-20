"""
FraudDetectionGNN - Application Runner

Run the complete fraud detection application:
  python run.py --backend    # Start FastAPI backend
  python run.py --frontend   # Start Streamlit frontend
  python run.py --all        # Start both (default)
  python run.py --train      # Train models only
  python run.py --test       # Run tests
"""

import subprocess
import sys
import os
import argparse
import time
import webbrowser
from pathlib import Path


def start_backend():
    print("🚀 Starting FastAPI Backend on http://localhost:8000")
    print("📚 API Docs at http://localhost:8000/docs")
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])


def start_frontend():
    print("🚀 Starting Streamlit Frontend on http://localhost:8501")
    time.sleep(2)
    webbrowser.open("http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"])


def run_training():
    print("🧠 Starting model training...")
    from ml.pipeline import FraudDetectionPipeline
    pipeline = FraudDetectionPipeline()
    results = pipeline.run_full_pipeline()
    print(f"\n✅ Training complete!")
    print(f"Best model: {results['best_model']}")
    print(f"Best ROC AUC: {results['best_metrics']['roc_auc']:.4f}")


def run_tests():
    print("🧪 Running tests...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])


def main():
    parser = argparse.ArgumentParser(description="FraudDetectionGNN Runner")
    parser.add_argument("--backend", action="store_true", help="Start backend only")
    parser.add_argument("--frontend", action="store_true", help="Start frontend only")
    parser.add_argument("--all", action="store_true", help="Start both backend and frontend")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--install", action="store_true", help="Install dependencies")

    args = parser.parse_args()

    if args.install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return

    if args.test:
        run_tests()
        return

    if args.train:
        run_training()
        return

    if args.backend:
        start_backend()
        return

    if args.frontend:
        start_frontend()
        return

    if args.all:
        import multiprocessing
        p1 = multiprocessing.Process(target=start_backend)
        p2 = multiprocessing.Process(target=start_frontend)
        p1.start()
        time.sleep(3)
        p2.start()
        p1.join()
        p2.join()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
