# core/batch_utils.py

import torch
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp

def all_same_gpu_model():
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return all(name == names[0] for name in names)

def is_batch_safe(model, tokenizer, batch_size, seq_len, device):
    try:
        seqs = ["A" * seq_len] * batch_size
        tokens = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            _ = model(**tokens, output_hidden_states=True).hidden_states[-1]
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return False
        raise

def find_max_batch_size(model_name: str, seq_len: int = 100, max_batch: int = 512, gpus=None, fast=False):
    """
    Returns a dict {gpu_idx: max_batch_size}
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices found.")

    available_gpus = list(range(torch.cuda.device_count()))
    selected_gpus = list(map(int, gpus.split(","))) if gpus else available_gpus

    if fast and all_same_gpu_model():
        selected_gpus = [selected_gpus[0]]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    def worker(gpu_idx, return_dict):
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

    for gpu_idx in selected_gpus:
        p = mp.Process(target=worker, args=(gpu_idx, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return dict(return_dict)
