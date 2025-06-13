# tests/test_embedder.py

from esm_mutation_score import ESMEmbedder
import torch

def test_embed_shape():
    embedder = ESMEmbedder("facebook/esm2_t6_8M_UR50D")
    out = embedder.embed_and_stack(["ACDEFGHIK"], batch_size=1)
    assert out.shape[0] == 1
