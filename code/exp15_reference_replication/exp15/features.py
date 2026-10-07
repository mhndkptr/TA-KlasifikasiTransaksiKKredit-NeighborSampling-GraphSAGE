from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["Use Chip", "MCC", "Errors?", "Merchant City", "Merchant State", "Zip"]
NUMERIC = ["amount", "hour", "day_of_week", "month"]


@dataclass
class FeatureBundle:
    transaction: sparse.csr_matrix
    classifier: sparse.csr_matrix
    transaction_names: list[str]
    classifier_names: list[str]


class IBMFeatureEncoder:
    """Train-only encoder for the IBM adapter; this is not private reference preprocessing."""
    def __init__(self):
        try:
            self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        except TypeError:  # scikit-learn < 1.2
            self.onehot = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
        self.scaler = StandardScaler()
        self.client_map = {}
        self.merchant_map = {}
        self.fitted = False

    @staticmethod
    def _numeric(frame):
        ts = pd.to_datetime(frame["timestamp"])
        return np.column_stack([frame["amount"].fillna(0).to_numpy(np.float64),
                                (ts.dt.hour + ts.dt.minute / 60).to_numpy(np.float64),
                                ts.dt.dayofweek.to_numpy(np.float64),
                                ts.dt.month.to_numpy(np.float64)])

    @staticmethod
    def _categories(frame):
        return frame[CATEGORICAL].astype("string").fillna("<missing>")

    def fit(self, train):
        self.scaler.fit(self._numeric(train))
        self.onehot.fit(self._categories(train))
        self.client_map = {value: i + 1 for i, value in enumerate(pd.unique(train["client_key"]))}
        self.merchant_map = {value: i + 1 for i, value in enumerate(pd.unique(train["merchant_key"]))}
        self.fitted = True
        return self

    def transform(self, frame):
        if not self.fitted:
            raise RuntimeError("Encoder belum fit")
        numeric = sparse.csr_matrix(self.scaler.transform(self._numeric(frame)).astype(np.float32))
        categorical = self.onehot.transform(self._categories(frame)).tocsr()
        transaction = sparse.hstack([numeric, categorical], format="csr", dtype=np.float32)
        client = np.asarray([self.client_map.get(x, 0) for x in frame["client_key"]], dtype=np.float32)[:, None]
        merchant = np.asarray([self.merchant_map.get(x, 0) for x in frame["merchant_key"]], dtype=np.float32)[:, None]
        classifier = sparse.hstack([transaction, sparse.csr_matrix(client), sparse.csr_matrix(merchant)],
                                   format="csr", dtype=np.float32)
        categories = self.onehot.get_feature_names_out(CATEGORICAL).tolist()
        tx_names = NUMERIC + categories
        return FeatureBundle(transaction, classifier, tx_names, tx_names + ["client_id_code", "merchant_id_code"])

    def manifest(self):
        if not self.fitted:
            raise RuntimeError("Encoder belum fit")
        return {"numeric": NUMERIC, "categorical": CATEGORICAL,
                "client_mapping_size": len(self.client_map), "merchant_mapping_size": len(self.merchant_map),
                "policy": "fit_on_full_train_before_graph_undersampling"}

