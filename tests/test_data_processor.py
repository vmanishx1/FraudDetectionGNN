import pytest
import pandas as pd
import numpy as np
from ml.preprocessing.data_processor import DataProcessor


class TestDataProcessor:
    def setup_method(self):
        self.processor = DataProcessor()

    def test_generate_synthetic_data(self):
        df = self.processor._generate_synthetic_data(1000)
        assert len(df) == 1000
        assert "TransactionAmt" in df.columns
        assert "isFraud" in df.columns
        assert df["isFraud"].dtype in [np.int64, np.int32, int]

    def test_preprocess_pipeline(self):
        df = self.processor._generate_synthetic_data(500)
        X, y = self.processor.preprocess(df)
        assert len(X) == len(y)
        assert len(X.columns) > 10
        assert y.dtype in [np.int64, np.int32, int]

    def test_train_val_test_split(self):
        df = self.processor._generate_synthetic_data(1000)
        X, y = self.processor.preprocess(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.train_val_test_split(X, y)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)

    def test_handle_missing_values(self):
        df = pd.DataFrame({"A": [1, 2, None, 4], "B": ["x", None, "z", "w"]})
        self.processor.numerical_columns = ["A"]
        self.processor.categorical_columns = ["B"]
        result = self.processor._handle_missing_values(df)
        assert result["A"].isnull().sum() == 0
        assert result["B"].isnull().sum() == 0

    def test_handle_outliers(self):
        np.random.seed(42)
        df = pd.DataFrame({"A": np.random.randn(100)})
        df.loc[0, "A"] = 1000
        self.processor._handle_outliers(df)
        assert df["A"].max() < 100
