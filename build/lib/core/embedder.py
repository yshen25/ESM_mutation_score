## esm_mutation_score/core/embedder.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class ESMEmbedder:
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_sequences(self, sequences, batch_size=8, use_cls=False):
        """
        Embed a list of sequences using the ESM model.

        Args:
            sequences (List[str]): List of protein sequences.
            batch_size (int): Batch size to use for inference.
            use_cls (bool): Whether to use CLS token or mean pooling.

        Returns:
            List[np.ndarray]: List of embedding vectors.
        """
        embeddings = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
            batch = sequences[i:i + batch_size]
            tokenized = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True
            ).to(self.device)

            outputs = self.model(**tokenized, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # last layer

            for j, seq in enumerate(batch):
                length = len(seq)
                rep = hidden_states[j]

                if use_cls:
                    emb = rep[0]  # CLS token
                else:
                    emb = rep[1:1+length].mean(dim=0)  # mean of unpadded tokens

                embeddings.append(emb.cpu().numpy())

            del outputs, hidden_states
            torch.cuda.empty_cache()

        return embeddings

    def embed_and_stack(self, sequences, batch_size=8, use_cls=False):
        """
        Same as embed_sequences but returns a single stacked numpy array.

        Args:
            sequences (List[str])
            batch_size (int)
            use_cls (bool)

        Returns:
            np.ndarray
        """
        embs = self.embed_sequences(sequences, batch_size, use_cls)
        return np.stack(embs)
