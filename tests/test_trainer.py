# tests/test_trainer.py

import numpy as np
from esm_mutation_score import train_model

def test_train_rf():
    X = np.random.rand(10, 8)
    y = np.random.rand(10)
    model, metrics = train_model(X, y, model_type="rf", cv=2)
    assert hasattr(model, "predict")
    assert "mean_r2" in metrics
