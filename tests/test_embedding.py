from models.esm_wrapper import ESMEmbedder

def test_embedding_dim():
    embedder = ESMEmbedder("facebook/esm2_t33_650M_UR50D")
    embs = embedder.embed_and_stack(["A" * 50], batch_size=1)
    assert embs.shape == (1, 1280)
