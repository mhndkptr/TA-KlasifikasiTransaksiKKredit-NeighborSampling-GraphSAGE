from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .artifacts import atomic_pickle


@dataclass
class ClassifierResult:
    scores: np.ndarray
    model: object
    backend: str


def fit_predict(train_x, train_y, holdout_x, *, backend="xgboost", n_estimators=100, seed=42):
    train_y = np.asarray(train_y, dtype=np.int8)
    if len(np.unique(train_y)) != 2:
        raise ValueError("XGBoost memerlukan kedua kelas pada training")
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Install xgboost dari requirements untuk backend referensi") from exc
        model = XGBClassifier(n_estimators=n_estimators, random_state=seed)
        name = "xgboost_reference_defaults"
    elif backend == "sklearn_compat":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        name = "sklearn_random_forest_compat"
    else:
        raise ValueError(f"Backend classifier tidak dikenal: {backend}")
    model.fit(train_x, train_y)
    return ClassifierResult(model.predict_proba(holdout_x)[:, 1], model, name)


def save_model(path, model):
    atomic_pickle(path, model)
