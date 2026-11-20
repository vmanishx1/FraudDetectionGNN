import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

from utils.logger import logger
from utils.config import MODEL_DIR, HIDDEN_DIM, DROPOUT
from ml.preprocessing.data_processor import DataProcessor
from ml.graph_builder.graph_constructor import GraphConstructor
from ml.training.trainer import Trainer
from ml.evaluation.evaluator import Evaluator
from ml.inference.predictor import FraudPredictor


class FraudDetectionPipeline:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path
        self.processor = DataProcessor(data_path)
        self.graph_builder = GraphConstructor()
        self.trainers: Dict[str, Trainer] = {}
        self.best_model_type: Optional[str] = None
        self.best_metrics: Optional[Dict[str, Any]] = None
        self.evaluator = Evaluator()
        self.predictor = FraudPredictor()
        self.graph_data = None

    def run_full_pipeline(self) -> Dict[str, Any]:
        logger.info("Starting full fraud detection pipeline")
        df = self.processor.load_ieee_dataset()
        X, y = self.processor.preprocess(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.train_val_test_split(X, y)
        self.graph_data = self.graph_builder.build_graph(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        results = self.train_and_evaluate(self.graph_data)
        self.predictor = FraudPredictor()
        self.predictor.model = self.trainers[self.best_model_type].model
        self.predictor.model.eval()
        return results

    def train_and_evaluate(self, graph_data) -> Dict[str, Any]:
        in_channels = graph_data.x.shape[1]
        model_types = ["gcn", "graphsage", "gat"]
        all_metrics = {}

        for model_type in model_types:
            logger.info(f"Training {model_type.upper()} model...")
            trainer = Trainer(
                model_type=model_type,
                in_channels=in_channels,
                hidden_dim=HIDDEN_DIM,
                dropout=DROPOUT,
            )
            history = trainer.train(graph_data, graph_data)
            metrics = trainer.evaluate(graph_data)
            trainer.save_model(MODEL_DIR / f"{model_type}_best.pt")
            self.trainers[model_type] = trainer
            all_metrics[model_type.upper()] = {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1": metrics.get("f1", 0),
                "roc_auc": metrics.get("roc_auc", 0),
            }
            logger.info(f"{model_type.upper()} - Acc: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")

        self.best_model_type = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"]).lower()
        self.best_metrics = all_metrics[self.best_model_type.upper()]
        logger.info(f"Best model: {self.best_model_type.upper()} with ROC AUC: {self.best_metrics['roc_auc']:.4f}")

        return {
            "all_metrics": all_metrics,
            "best_model": self.best_model_type,
            "best_metrics": self.best_metrics,
            "history": self.trainers[self.best_model_type].history,
        }

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return self.predictor.predict_single(features)

    def get_graph_data(self) -> Optional[Dict[str, Any]]:
        if self.graph_data is None:
            return None
        return self.graph_builder.get_networkx_data(self.graph_data)
