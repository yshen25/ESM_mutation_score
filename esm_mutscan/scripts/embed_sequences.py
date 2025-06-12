#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import os
import pickle
from models.esm_wrapper import ESMEmbedder

def parse_args():
    parser = argparse.ArgumentParser(description="Embed sequences using HuggingFace ESM")
    parser.add_argument("--input", required=True, help="Path to CSV with `id,sequence` columns")
    parser.add_argument("--output", required=True, help="Output path (.npy or .pkl)")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D", help="ESM model name or path")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding")
    parser.add_argument("--use-cls", action="store_true", help="Use CLS token embedding instead of mean pooling")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load input
    df = pd.read_csv(args.input)
    if "sequence" not in df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column")

    sequences = df["sequence"].tolist()
    ids = df["id"].tolist() if "id" in df.columns else list(range(len(sequences)))

    # Run embedding
    embedder = ESMEmbedder(args.model)
    embeddings = embedder.embed_and_stack(sequences, batch_size=args.batch_size, use_cls=args.use_cls)

    # Save
    if args.output.endswith(".npy"):
        np.save(args.output, embeddings)
    elif args.output.endswith(".pkl"):
        with open(args.output, "wb") as f:
            pickle.dump({"ids": ids, "embeddings": embeddings}, f)
    else:
        raise ValueError("Output must end in .npy or .pkl")

    print(f"[✓] Saved embeddings: {args.output}")

if __name__ == "__main__":
    main()
