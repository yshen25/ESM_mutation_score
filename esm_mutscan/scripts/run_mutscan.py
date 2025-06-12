#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.esm_wrapper import ESMEmbedder
from models.mutation_generator import generate_mutants
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot mutation scan using ESM")
    parser.add_argument("--wt-seq", required=True, help="Wild-type protein sequence")
    parser.add_argument("--positions", required=True, help="Comma-separated 0-based mutable positions")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D", help="HuggingFace ESM model")
    parser.add_argument("--mut-order", type=int, default=2, help="Mutation order (1=single, 2=double, ...)")
    parser.add_argument("--no-consec", action="store_true", help="Exclude consecutive position mutations")
    return parser.parse_args()

def score_sequence(model, tokenizer, seq):
    device = next(model.parameters()).device
    seq = seq.replace(" ", "").upper()
    tokenized = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    true_tokens = input_ids[:, 1:-1]
    token_log_probs = log_probs[:, 1:-1].gather(2, true_tokens.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()

def main():
    args = parse_args()
    positions = list(map(int, args.positions.split(",")))
    wt_seq = args.wt_seq.upper()

    print("[*] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("[*] Scoring WT sequence...")
    wt_score = score_sequence(model, tokenizer, wt_seq)

    print("[*] Generating mutants...")
    mutants = generate_mutants(
        wt_seq,
        positions,
        mut_order=args.mut_order,
        exclude_consecutive=args.no_consec,
        output_format="dict"
    )

    print(f"[*] Scoring {len(mutants)} mutants...")
    results = []
    for m in tqdm(mutants):
        score = score_sequence(model, tokenizer, m["mutant_seq"])
        if score > wt_score:
            m["score"] = score
            m["delta"] = score - wt_score
            results.append(m)

    print(f"[✓] Found {len(results)} improved mutants.")
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[✓] Saved: {args.output}")

if __name__ == "__main__":
    main()
