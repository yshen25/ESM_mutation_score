import argparse
import subprocess
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="ESM-MutScan command-line entrypoint")
    parser.add_argument("command", help="Which script to run", choices=[
        "embed", "train", "predict", "mutscan", "batchtest"
    ])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script_map = {
        "embed": "scripts/embed_sequences.py",
        "train": "scripts/ml_train.py",
        "predict": "scripts/predict_affinity.py",
        "mutscan": "scripts/run_mutscan.py",
        "batchtest": "scripts/find_max_batch.py"
    }

    script = Path(__file__).resolve().parent.parent / script_map[args.command]
    cmd = ["python", str(script)] + args.args
    os.execvp("python", cmd)  # Replaces process with target script

if __name__ == "__main__":
    main()
