#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import pickle
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Predict affinity using trained ML model")
    parser.add_argument("--embed", required=True, help="Path to .npy or .pkl file with embeddings")
    parser.add_argument("--model", required=True, help="Trained model (.pkl)")
    parser.add_argument("--output", required=True, help="Output .csv for predictions")
    parser.add_argument("--ids", help="Optional CSV or .pkl file with sequence IDs")
    return parser.parse_args()

def load_embeddings_and_ids(embed_path, ids_path=None):
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

def predict_with_uncertainty(model, X):
    if hasattr(model, "estimators_"):
        # RandomForest or similar
        print("[*] Estimating uncertainty via tree ensemble variance...")
        all_preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=1)
        mean_pred = all_preds.mean(axis=1)
        std_pred = all_preds.std(axis=1)
    else:
        mean_pred = model.predict(X)
        std_pred = np.full_like(mean_pred, fill_value=np.nan)
    return mean_pred, std_pred

def main():
    args = parse_args()

    print("[*] Loading embeddings...")
    X, ids = load_embeddings_and_ids(args.embed, args.ids)

    print("[*] Loading model...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print("[*] Predicting...")
    y_pred, y_std = predict_with_uncertainty(model, X)

    df_out = pd.DataFrame({"id": ids, "prediction": y_pred, "uncertainty": y_std})
    df_out.to_csv(args.output, index=False)
    print(f"[✓] Predictions saved to: {args.output}")

if __name__ == "__main__":
    main()
