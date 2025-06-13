# esm_mutation_score/cli/main.py

import argparse
from . import embed, train, predict, mutscan, batchtest

def main():
    parser = argparse.ArgumentParser(description="ESM Mutation Score CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed.register_cli(subparsers)
    train.register_cli(subparsers)
    predict.register_cli(subparsers)
    mutscan.register_cli(subparsers)
    batchtest.register_cli(subparsers)

    args = parser.parse_args()
    args.func(args)
