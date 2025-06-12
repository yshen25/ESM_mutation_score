# ESM-MutScan Toolkit
Score protein sequences using ESM based methods, including zero shot and few shot modes with multiple methods

A modular pipeline for:
- Embedding sequences with HuggingFace ESM models
- Generating single/double mutants
- Performing zero-shot mutation scoring
- Training and applying ML models on ESM embeddings

---

## 🧱 Components

- `embed_sequences.py`: embed sequences from CSV
- `ml_train.py`: train regressors on ESM embeddings
- `predict_affinity.py`: score new sequences
- `run_mutscan.py`: scan mutants and filter for improved ESM scores
- `find_max_batch.py`: detect max GPU-safe batch size

---

## 📦 Example Pipeline

```bash
bash examples/run_pipeline.sh
```

## 📥 Input Format

sample_input.csv
```
id,sequence
mut1,EVQLVESG...
```
```
targets.csv
id,target
mut1,0.5
```

## ✅ Install Dependencies
```
pip install -r requirements.txt
```
💡 For GPU support, install PyTorch with CUDA matching your system:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🧪 Run Tests
```
pytest tests/
```
## 🔮 Coming Soon
- CLI-based end-to-end mutation design

- Ensemble scoring

- ESMFold structure-aware filtering