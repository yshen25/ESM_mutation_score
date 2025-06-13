# esm_mutation_score/cli/mutscan.py

import argparse
import pandas as pd
from tqdm import tqdm
from esm_mutation_score import generate_mutants, score_sequence, load_mlm_model

def run(args):
    wt_seq = args.wt_seq.upper()
    positions = list(map(int, args.positions.split(",")))

    print("[*] Loading model...")
    model, tokenizer = load_mlm_model(args.model)

    wt_score = score_sequence(model, tokenizer, wt_seq)

    mutants = generate_mutants(
        wt_seq,
        positions,
        mut_order=args.mut_order,
        exclude_consecutive=args.no_consec,
        output_format="dict"
    )

    results = []
    for m in tqdm(mutants):
        score = score_sequence(model, tokenizer, m["mutant_seq"])
        if score > wt_score:
            m["score"] = score
            m["delta"] = score - wt_score
            results.append(m)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[✓] Saved {len(results)} mutants to {args.output}")

def register_cli(subparsers):
    parser = subparsers.add_parser("mutscan", help="Run zero-shot mutation scan")
    parser.add_argument("--wt-seq", required=True)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--mut-order", type=int, default=2)
    parser.add_argument("--no-consec", action="store_true")
    parser.set_defaults(func=run)
