#!/usr/bin/env python3

import torch
import argparse
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description="Find max safe batch size per GPU for a given ESM model")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D", help="ESM model name or path")
    parser.add_argument("--seq-len", type=int, default=100, help="Synthetic sequence length")
    parser.add_argument("--max-batch", type=int, default=512, help="Upper bound for batch size search")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU indices (e.g. 0,1,2)")
    parser.add_argument("--fast", action="store_true", help="If GPUs are identical, only test the first one")
    return parser.parse_args()

def all_same_gpu_model():
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return all(name == names[0] for name in names)

def is_batch_safe(model, tokenizer, batch_size, seq_len, device):
    try:
        seqs = ["A" * seq_len] * batch_size
        tokens = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
            _ = out.hidden_states[-1]
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return False
        raise

def find_max_batch_for_gpu(gpu_idx, model_name, seq_len, max_batch, return_dict):
    torch.cuda.set_device(gpu_idx)
    device = f"cuda:{gpu_idx}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    low, high = 1, max_batch
    while low < high:
        mid = (low + high + 1) // 2
        torch.cuda.empty_cache()
        if is_batch_safe(model, tokenizer, mid, seq_len, device):
            low = mid
        else:
            high = mid - 1

    return_dict[gpu_idx] = low

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("No CUDA devices found.")
        return

    available_gpus = list(range(torch.cuda.device_count()))
    selected_gpus = list(map(int, args.gpus.split(","))) if args.gpus else available_gpus

    if args.fast and all_same_gpu_model():
        print("[*] Identical GPUs detected — using --fast mode to test only GPU 0")
        selected_gpus = [0]

    print(f"[*] Running batch size scan on GPUs: {selected_gpus}")
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for gpu_idx in selected_gpus:
        p = mp.Process(target=find_max_batch_for_gpu, args=(
            gpu_idx, args.model, args.seq_len, args.max_batch, return_dict
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n[✓] Max batch sizes:")
    for gpu_idx in sorted(return_dict.keys()):
        print(f"GPU {gpu_idx}: batch size {return_dict[gpu_idx]}")

if __name__ == "__main__":
    main()
