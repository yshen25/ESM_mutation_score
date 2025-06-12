#!/bin/bash

# 1. Embed sequences
python scripts/embed_sequences.py \
  --input examples/sample_input.csv \
  --output examples/embeddings.pkl \
  --model facebook/esm2_t33_650M_UR50D \
  --batch-size 4

# 2. Train model
python scripts/ml_train.py \
  --embed examples/embeddings.pkl \
  --labels examples/targets.csv \
  --label-col target \
  --model rf \
  --output examples/rf_model.pkl

# 3. Predict affinity
python scripts/predict_affinity.py \
  --embed examples/embeddings.pkl \
  --model examples/rf_model.pkl \
  --output examples/predictions.csv
