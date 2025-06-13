# esm_mutation_score/cli/embed.py

import argparse
import pandas as pd
import numpy as np
import pickle
from esm_mutation_score import ESMEmbedder

def run(args):
    df = pd.read_csv(args.input)
    sequences = df["sequence"].tolist()
    ids = df["id"].tolist() if "id" in df.columns else list(range(len(sequences)))

    embedder = ESMEmbedder(args.model)
    embs = embedder.embed_and_stack(sequences, batch_size=args.batch_size, use_cls=args.use_cls)

    if args.output.endswith(".npy"):
        np.save(args.output, embs)
    else:
        with open(args.output, "wb") as f:
            pickle.dump({"ids": ids, "embeddings": embs}, f)
    print(f"[✓] Saved embeddings to {args.output}")

def register_cli(subparsers):
    parser = subparsers.add_parser("embed", help="Embed sequences with ESM")
    parser.add_argument("--input", required=True, help="CSV with 'id' and 'sequence' columns")
    parser.add_argument("--output", required=True, help="Output .npy or .pkl path")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-cls", action="store_true")
    parser.set_defaults(func=run)
