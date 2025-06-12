#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML model on ESM embeddings")
    parser.add_argument("--embed", required=True, help="Path to .npy or .pkl embedding file")
    parser.add_argument("--labels", required=True, help="CSV file with target values")
    parser.add_argument("--label-col", default="target", help="Column name in CSV for target")
    parser.add_argument("--model", choices=["rf", "gbr"], default="rf", help="Model type: rf or gbr")
    parser.add_argument("--output", required=True, help="Output .pkl file for trained model")
    parser.add_argument("--cv", type=int, default=0, help="# folds for CV (0 = skip)")
    return parser.parse_args()

def load_embeddings(embed_path):
    if embed_path.endswith(".npy"):
        return np.load(embed_path)
    elif embed_path.endswith(".pkl"):
        with open(embed_path, "rb") as f:
            data = pickle.load(f)
            return data["embeddings"]
    else:
        raise ValueError("Unsupported embedding file type. Use .npy or .pkl")

def main():
    args = parse_args()

    print("[*] Loading embeddings and labels...")
    X = load_embeddings(args.embed)
    y = pd.read_csv(args.labels)[args.label_col].values

    if len(y) != X.shape[0]:
        raise ValueError("Mismatch between # of labels and # of embeddings")

    print(f"[*] Training model: {args.model}")
    if args.model == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif args.model == "gbr":
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    if args.cv > 0:
        print(f"[*] Running {args.cv}-fold cross-validation...")
        scores = cross_val_score(model, X, y, cv=args.cv, scoring="r2")
        print(f"[✓] CV R² scores: {scores}")
        print(f"[✓] Mean R²: {np.mean(scores):.4f}")
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f"[✓] R²: {r2_score(y_val, y_pred):.4f}")
        print(f"[✓] RMSE: {np.sqrt(mean_squared_error(y_val, y_pred)):.4f}")

    print(f"[✓] Saving model to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
