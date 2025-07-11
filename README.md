# ESM Mutation Score

ESM-based toolkit for embedding, scoring, and predicting mutation effects on antibody sequences.

## Features

- HuggingFace ESM integration for protein sequence embeddings
- Point mutation generation (single, double, etc)
- Zero-shot mutation scoring via masked language models
- Trainable ML models (RF/GBR) on custom datasets
- Uncertainty-aware prediction
- GPU batch optimization utility
- CLI + Python API support

---

## Installation

```bash
pip install -e .[dev]
```

---

## Command-Line Usage
```bash
esm-mutate embed --input seqs.csv --output embs.pkl
esm-mutate train --embed embs.pkl --labels labels.csv --output model.pkl
esm-mutate predict --embed embs.pkl --model model.pkl --output preds.csv
esm-mutate mutscan --wt-seq SEQ --positions 3,7,10 --output mutants.csv
esm-mutate batchtest --model facebook/esm2_t33_650M_UR50D
```

---

## Python API
```python
from esm_mutation_score import (
    ESMEmbedder, generate_mutants, score_sequence, load_mlm_model,
    train_model, load_model, predict
)

# Embedding
embedder = ESMEmbedder()
X = embedder.embed_and_stack(["ACDEFGHIK"])

# Mutation
mutants = generate_mutants("ACDEFGHIK", [2, 3], mut_order=1)

# Scoring
model, tokenizer = load_mlm_model("facebook/esm2_t33_650M_UR50D")
score = score_sequence(model, tokenizer, "ACDEFGHIK")

# Training
trained_model, metrics = train_model(X, y, model_type="rf")

# Prediction
y_pred, y_std = predict(trained_model, X)
```

---

## Project Structure
- esm_mutation_score/core/: core functionality
- esm_mutation_score/cli/: CLI entrypoints
- tests/: unit tests