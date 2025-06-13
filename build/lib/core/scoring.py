## esm_mutation_score/core.py

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_mlm_model(model_name: str, device: str = None):
    """
    Load a masked language model and tokenizer.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def score_sequence(model, tokenizer, seq: str) -> float:
    """
    Compute the log-likelihood score of a full sequence under a masked LM.

    Args:
        model: A HuggingFace masked language model.
        tokenizer: Corresponding tokenizer.
        seq: Protein sequence (AAs only, no whitespace or special tokens)

    Returns:
        Sum log-likelihood of all amino acids.
    """
    device = next(model.parameters()).device
    seq = seq.replace(" ", "").upper()
    tokenized = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized["input_ids"].to(device)

    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    true_tokens = input_ids[:, 1:-1]
    token_log_probs = log_probs[:, 1:-1].gather(2, true_tokens.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()
