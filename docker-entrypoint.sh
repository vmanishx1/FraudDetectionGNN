#!/bin/bash
set -e

echo "================================================"
echo "   🛡️ FraudDetectionGNN - Starting Service"
echo "================================================"

mkdir -p saved_models datasets reports logs

if [ "$1" = "uvicorn" ]; then
    echo "Starting Backend API Server..."
    exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
elif [ "$1" = "streamlit" ]; then
    echo "Starting Streamlit Frontend..."
    exec streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
else
    exec "$@"
fi
