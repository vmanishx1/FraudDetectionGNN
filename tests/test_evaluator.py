import pytest
import numpy as np
from ml.evaluation.evaluator import Evaluator


class TestEvaluator:
    def setup_method(self):
        self.evaluator = Evaluator()
        self.y_true = np.array([0, 0, 0, 1, 1, 1])
        self.y_pred = np.array([0, 0, 1, 1, 1, 1])
        self.y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.9, 0.7])

    def test_compute_all_metrics(self):
        metrics = self.evaluator.compute_all_metrics(self.y_true, self.y_pred, self.y_prob)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_confusion_matrix_shape(self):
        metrics = self.evaluator.compute_all_metrics(self.y_true, self.y_pred, self.y_prob)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (2, 2)

    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = self.evaluator.compute_all_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_plot_methods(self):
        cm = np.array([[50, 10], [5, 35]])
        fig = self.evaluator.plot_confusion_matrix(cm)
        assert fig is not None
        fig = self.evaluator.plot_roc_curve(self.y_true, self.y_prob)
        assert fig is not None
        fig = self.evaluator.plot_precision_recall_curve(self.y_true, self.y_prob)
        assert fig is not None
