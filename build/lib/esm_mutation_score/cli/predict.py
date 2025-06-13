# esm_mutation_score/cli/predict.py

import argparse
from esm_mutation_score import load_model, load_embeddings, predict, format_predictions

def run(args):
    model = load_model(args.model)
    X, ids = load_embeddings(args.embed, ids_path=args.ids)
    y_pred, y_std = predict(model, X)
    df = format_predictions(ids, y_pred, y_std)
    df.to_csv(args.output, index=False)
    print(f"[✓] Saved predictions to {args.output}")

def register_cli(subparsers):
    parser = subparsers.add_parser("predict", help="Run prediction with trained model")
    parser.add_argument("--embed", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", help="Optional: CSV with `id` column")
    parser.set_defaults(func=run)
