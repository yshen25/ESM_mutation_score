# esm_mutation_score/cli/train.py

import argparse
import pandas as pd
import numpy as np
from esm_mutation_score import train_model, save_model

def run(args):
    X = np.load(args.embed) if args.embed.endswith(".npy") else pickle.load(open(args.embed, "rb"))["embeddings"]
    y = pd.read_csv(args.labels)[args.label_col].values

    model, metrics = train_model(X, y, model_type=args.model, cv=args.cv)
    save_model(model, args.output)

    print(f"[✓] Model saved to {args.output}")
    print("[*] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

def register_cli(subparsers):
    parser = subparsers.add_parser("train", help="Train regression model")
    parser.add_argument("--embed", required=True, help="Path to embeddings (.npy or .pkl)")
    parser.add_argument("--labels", required=True, help="CSV with label column")
    parser.add_argument("--label-col", default="target")
    parser.add_argument("--model", choices=["rf", "gbr"], default="rf")
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=run)
