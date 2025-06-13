# core/predictor.py

import numpy as np
import pandas as pd
import pickle

def load_embeddings(embed_path: str, ids_path: str = None):
    """
    Load embeddings and optional sequence IDs from .pkl or .npy
    """
    if embed_path.endswith(".npy"):
        X = np.load(embed_path)
        ids = list(range(X.shape[0])) if not ids_path else pd.read_csv(ids_path)["id"].tolist()
    elif embed_path.endswith(".pkl"):
        with open(embed_path, "rb") as f:
            data = pickle.load(f)
            X = data["embeddings"]
            ids = data.get("ids", list(range(X.shape[0])))
    else:
        raise ValueError("Unsupported embedding format")
    return X, ids

def predict(model, X: np.ndarray):
    """
    Predict outputs (and uncertainty if applicable) for given inputs.

    Returns:
        mean_pred, std_pred (NaN if not available)
    """
    if hasattr(model, "estimators_"):
        all_preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=1)
        mean_pred = all_preds.mean(axis=1)
        std_pred = all_preds.std(axis=1)
    else:
        mean_pred = model.predict(X)
        std_pred = np.full_like(mean_pred, fill_value=np.nan)
    return mean_pred, std_pred

def format_predictions(ids, y_pred, y_std):
    return pd.DataFrame({
        "id": ids,
        "prediction": y_pred,
        "uncertainty": y_std
    })
