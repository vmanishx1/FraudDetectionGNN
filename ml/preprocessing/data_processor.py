import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import warnings

from utils.logger import logger

warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = SimpleImputer(strategy="median")
        self.feature_columns: list = []
        self.categorical_columns: list = []
        self.numerical_columns: list = []

    def load_ieee_dataset(self) -> pd.DataFrame:
        if self.data_path is None:
            return self._generate_synthetic_data()
        try:
            train_transaction = pd.read_csv(self.data_path / "train_transaction.csv")
            train_identity = pd.read_csv(self.data_path / "train_identity.csv")
            df = train_transaction.merge(train_identity, on="TransactionID", how="left")
            logger.info(f"Loaded IEEE dataset with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.warning(f"Could not load IEEE dataset: {e}. Generating synthetic data.")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        np.random.seed(42)
        n = n_samples
        df = pd.DataFrame({
            "TransactionID": range(1, n + 1),
            "TransactionDT": np.random.randint(86400, 86400 * 365, n),
            "TransactionAmt": np.abs(np.random.lognormal(4, 1.5, n)),
            "ProductCD": np.random.choice(["W", "H", "C", "R"], n),
            "card1": np.random.randint(10000, 99999, n),
            "card2": np.random.randint(100, 9999, n),
            "card3": np.random.randint(100, 999, n),
            "card4": np.random.choice(["visa", "mastercard", "amex", "discover"], n),
            "card5": np.random.randint(10000, 99999, n),
            "card6": np.random.choice(["debit", "credit"], n),
            "addr1": np.random.uniform(30, 50, n),
            "addr2": np.random.uniform(-120, -70, n),
            "dist1": np.random.exponential(50, n),
            "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "outlook.com", "protonmail.com", None], n),
            "R_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "outlook.com", "protonmail.com", None], n),
            "C1": np.random.exponential(10, n),
            "C2": np.random.exponential(5, n),
            "C3": np.random.exponential(20, n),
            "C4": np.random.exponential(30, n),
            "C5": np.random.exponential(15, n),
            "C6": np.random.exponential(25, n),
            "C7": np.random.exponential(8, n),
            "C8": np.random.exponential(12, n),
            "C9": np.random.exponential(18, n),
            "C10": np.random.exponential(22, n),
            "C11": np.random.exponential(6, n),
            "C12": np.random.exponential(14, n),
            "C13": np.random.exponential(9, n),
            "C14": np.random.exponential(16, n),
            "D1": np.random.exponential(100, n),
            "D2": np.random.exponential(50, n),
            "D3": np.random.exponential(75, n),
            "D4": np.random.exponential(200, n),
            "D5": np.random.exponential(30, n),
            "D6": np.random.exponential(40, n),
            "D7": np.random.exponential(60, n),
            "D8": np.random.exponential(80, n),
            "D9": np.random.exponential(90, n),
            "D10": np.random.exponential(25, n),
            "D11": np.random.exponential(35, n),
            "D12": np.random.exponential(45, n),
            "D13": np.random.exponential(55, n),
            "D14": np.random.exponential(65, n),
            "D15": np.random.exponential(15, n),
            "M1": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "M2": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "M3": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "M4": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "M5": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "M6": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "M7": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "M8": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "M9": np.random.choice(["M0", "M1", "M2", "M3", "M4", "M5"], n),
            "id_01": np.random.randn(n),
            "id_02": np.random.exponential(10, n),
            "id_03": np.random.choice(["M0", "M1"], n),
            "id_04": np.random.choice(["M0", "M1"], n),
            "id_05": np.random.choice(["M0", "M1"], n),
            "id_06": np.random.choice(["M0", "M1"], n),
            "id_07": np.random.choice(["M0", "M1"], n),
            "id_08": np.random.choice(["M0", "M1"], n),
            "id_09": np.random.choice(["M0", "M1"], n),
            "id_10": np.random.choice(["M0", "M1"], n),
            "id_11": np.random.choice(["M0", "M1"], n),
            "id_12": np.random.choice(["M0", "M1", "M2"], n),
            "id_13": np.random.choice(["M0", "M1", "M2"], n),
            "id_14": np.random.choice(["M0", "M1", "M2"], n),
            "id_15": np.random.choice(["M0", "M1", "M2"], n),
            "id_16": np.random.choice(["M0", "M1", "M2"], n),
            "id_17": np.random.choice(["M0", "M1", "M2", "M3"], n),
            "id_18": np.random.choice(["M0", "M1", "M2", "M3"], n),
            "id_19": np.random.choice(["M0", "M1", "M2", "M3"], n),
            "id_20": np.random.choice(["M0", "M1", "M2", "M3"], n),
            "id_21": np.random.uniform(0, 1, n),
            "id_22": np.random.uniform(0, 1, n),
            "id_23": np.random.uniform(0, 1, n),
            "id_24": np.random.uniform(0, 1, n),
            "id_25": np.random.uniform(0, 1, n),
            "id_26": np.random.uniform(0, 1, n),
            "id_27": np.random.uniform(0, 1, n),
            "id_28": np.random.uniform(0, 1, n),
            "id_29": np.random.uniform(0, 1, n),
            "id_30": np.random.uniform(0, 1, n),
            "id_31": np.random.uniform(0, 1, n),
            "id_32": np.random.uniform(0, 1, n),
            "id_33": np.random.uniform(0, 1, n),
            "id_34": np.random.uniform(0, 1, n),
            "id_35": np.random.uniform(0, 1, n),
            "id_36": np.random.uniform(0, 1, n),
            "id_37": np.random.uniform(0, 1, n),
            "id_38": np.random.uniform(0, 1, n),
            "DeviceType": np.random.choice(["mobile", "desktop", "tablet"], n),
            "DeviceInfo": np.random.choice(["iPhone", "Samsung", "Windows", "MacOS", "Linux", "iPad"], n),
            "isFraud": np.random.choice([0, 1], n, p=[0.96, 0.04]),
        })
        logger.info(f"Generated synthetic dataset with {len(df)} rows, fraud rate: {df['isFraud'].mean():.4f}")
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Starting preprocessing pipeline")
        df = df.copy()
        if "isFraud" in df.columns:
            y = df["isFraud"].astype(int)
            df = df.drop(columns=["isFraud"])
        else:
            y = pd.Series([0] * len(df))

        drop_cols = ["TransactionID", "TransactionDT"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        df = self._handle_missing_values(df)
        df = self._encode_categorical(df)
        df = self._create_features(df, y)
        df = self._scale_features(df)
        df = self._handle_outliers(df)

        self.feature_columns = df.columns.tolist()
        logger.info(f"Preprocessing complete. Features: {len(self.feature_columns)}, Samples: {len(df)}")
        return df, y

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df

    def _create_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if "TransactionAmt" in df.columns:
            df["LogTransactionAmt"] = np.log1p(df["TransactionAmt"])
            df["TransactionAmtBin"] = pd.qcut(df["TransactionAmt"], q=10, labels=False, duplicates="drop")
        if "card1" in df.columns and "card2" in df.columns:
            df["CardProduct"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
            df["CardProduct"] = pd.factorize(df["CardProduct"])[0]
        if "dist1" in df.columns:
            df["DistBin"] = pd.qcut(df["dist1"].fillna(0), q=5, labels=False, duplicates="drop")
        for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"]:
            if c in df.columns:
                df[f"{c}_log"] = np.log1p(df[c].clip(lower=0))
        addr_cols = [c for c in df.columns if c.startswith("addr")]
        if len(addr_cols) >= 2:
            df["DistanceEstimate"] = np.sqrt(
                (df[addr_cols[0]].fillna(0) - df[addr_cols[0]].mean()) ** 2 +
                (df[addr_cols[1]].fillna(0) - df[addr_cols[1]].mean()) ** 2
            )
        id_cols = [c for c in df.columns if c.startswith("id_") and df[c].dtype in [np.float64, np.int64]]
        if id_cols:
            df["IDMean"] = df[id_cols].mean(axis=1)
            df["IDStd"] = df[id_cols].std(axis=1).fillna(0)
        if "DeviceType" in df.columns:
            device_map = {"mobile": 1, "desktop": 2, "tablet": 3, "Unknown": 0}
            df["DeviceType"] = df["DeviceType"].map(device_map).fillna(0)
        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(q1, q99)
        return df

    def train_val_test_split(
        self, X: pd.DataFrame, y: pd.Series, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
