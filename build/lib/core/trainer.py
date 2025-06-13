# core/trainer.py

import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    cv: int = 0,
    random_state: int = 42
):
    """
    Train a regression model on embeddings and labels.

    Returns:
        trained model, (optional) CV scores or validation metrics
    """
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_type == "gbr":
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state)
    else:
        raise ValueError("Unsupported model type")

    if cv > 0:
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        return model.fit(X, y), {"cv_r2_scores": scores, "mean_r2": np.mean(scores)}
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        metrics = {
            "val_r2": r2_score(y_val, y_pred),
            "val_rmse": np.sqrt(mean_squared_error(y_val, y_pred))
        }
        return model, metrics

def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
