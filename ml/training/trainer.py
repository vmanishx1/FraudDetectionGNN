import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score
)
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import time

from utils.logger import logger
from utils.config import MODEL_DIR, EPOCHS, LEARNING_RATE, BATCH_SIZE
from ml.models.gnn_models import ModelFactory


class Trainer:
    def __init__(
        self,
        model_type: str = "gcn",
        in_channels: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        learning_rate: float = LEARNING_RATE,
        epochs: int = EPOCHS,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ModelFactory.create_model(
            model_type=model_type,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self._get_class_weights())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        self.epochs = epochs
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                        "precision": [], "recall": [], "f1": [], "roc_auc": []}
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def _get_class_weights(self) -> torch.Tensor:
        return torch.tensor([1.0, 5.0], device=self.device)

    def train(self, train_data: Data, val_data: Data) -> Dict[str, Any]:
        logger.info(f"Training {self.model_type.upper()} on {self.device}")
        self.model.train()

        train_loader = self._create_dataloader(train_data)
        val_loader = self._create_dataloader(val_data)

        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_metrics = self._evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["precision"].append(val_metrics["precision"])
            self.history["recall"].append(val_metrics["recall"])
            self.history["f1"].append(val_metrics["f1"])
            self.history["roc_auc"].append(val_metrics["roc_auc"])

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} | "
                    f"Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f} | "
                    f"ROC AUC: {val_metrics['roc_auc']:.4f} | Time: {time.time()-start_time:.2f}s"
                )

        self.model.load_state_dict(self.best_model_state)
        return self.history

    def _create_dataloader(self, data: Data) -> DataLoader:
        return DataLoader([data], batch_size=1, shuffle=True)

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def _evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                total_loss += loss.item()

                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)

                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        metrics = self._compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        return avg_loss, metrics

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
            "mcc": matthews_corrcoef(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
        return metrics

    def evaluate(self, test_data: Data) -> Dict[str, Any]:
        self.model.eval()
        loader = self._create_dataloader(test_data)
        _, metrics = self._evaluate(loader)
        cm = confusion_matrix(
            test_data.y.cpu().numpy(),
            self.predict(test_data).cpu().numpy()
        )
        metrics["confusion_matrix"] = cm.tolist()
        return metrics

    def predict(self, data: Data) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None)
            return out.argmax(dim=1)

    def predict_proba(self, data: Data) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None)
            return F.softmax(out, dim=1)

    def save_model(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = MODEL_DIR / f"{self.model_type}_best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.best_model_state or self.model.state_dict(),
                "model_type": self.model_type,
                "history": self.history,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logger.info(f"Model saved to {path}")
        return path

    def load_model(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_type = checkpoint.get("model_type", self.model_type)
        self.history = checkpoint.get("history", self.history)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Model loaded from {path}")


