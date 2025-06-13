# esm_mutation_score/cli/batchtest.py

import argparse
from esm_mutation_score import find_max_batch_size

def run(args):
    result = find_max_batch_size(
        model_name=args.model,
        seq_len=args.seq_len,
        max_batch=args.max_batch,
        gpus=args.gpus,
        fast=args.fast
    )
    for gpu, batch in result.items():
        print(f"[✓] GPU {gpu}: max batch size {batch}")

def register_cli(subparsers):
    parser = subparsers.add_parser("batchtest", help="Probe max batch size per GPU")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--max-batch", type=int, default=512)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--fast", action="store_true")
    parser.set_defaults(func=run)
