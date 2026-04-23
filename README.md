# Decision Intelligence Assistant

RAG + ML + LLM pipeline for customer support ticket triage (TWCS-based).

## Project Structure

- `src/rag/` - RAG indexing, retrieval, and triage flow
- `src/ML/` - ML training + ML/LLM prediction scripts
- `src/LLM/` - LLM client/provider handling
- `src/data/` - data loading/cleaning utilities
- `src/features/` - shared feature engineering helpers
- `src/prepare_datasets.py` - unified dataset preparation for RAG + ML
- `data/processed/` - prepared datasets and vector index artifacts
- `data/artifacts/` - trained ML model + evaluation report

## Two Execution Modes

### Training mode (local, no Docker)

Use this mode to prepare data, build embeddings, and train ML artifacts.

- Installs from `requirements/train.txt`
- Runs on your host machine
- Produces artifacts in `data/processed/` and `data/artifacts/`

### Serving mode (Docker)

Use this mode for fast API responses with prebuilt artifacts.

- Installs from `requirements/serve.txt`
- Runs `backend`, `frontend`, and `chroma` in Docker Compose
- Optimized for RAG/API response path (not training)

## Requirements

- Python `>=3.10`
- `uv` installed

Install local training dependencies:

```bash
pip install -r requirements/train.txt
```

## Environment Setup

Copy and edit environment values:

```bash
copy .env.example .env
```

Use one hosted LLM provider key in `.env`:

- `GROQ_API_KEY` (recommended current setup), or
- `OPENROUTER_API_KEY`, or
- `OPENAI_API_KEY`

## Training Mode Commands (Local)

Always run from repo root with module mode (`-m`).

### 1) Prepare datasets

```bash
uv run python -m src.prepare_datasets
```

Outputs include:

- `data/processed/retrieval_corpus.csv`
- `data/processed/ml_train.csv`
- `data/processed/ml_val.csv`
- `data/processed/ml_test.csv`

### 2) Build RAG index (Chroma + OpenAI embeddings)

```bash
uv run python -m src.rag.index_rag --backend openai --storage chroma --chroma-collection rag_tickets --embed-model text-embedding-3-small
```

### 3) Query Retrieval Only

```bash
uv run python -m src.rag.retrieve_rag --query "internet down cannot login" --backend chroma --chroma-collection rag_tickets --k 5 --json
```

### 4) Full RAG Triage (RAG answer + non-RAG answer)

```bash
uv run python -m src.rag.triage_with_rag --ticket "internet down cannot login" --retrieval-backend chroma --chroma-collection rag_tickets --embed-model text-embedding-3-small --k 5 --json
```

### 5) Train ML Priority Baseline

```bash
uv run python -m src.ML.train_priority_baseline
```

Outputs:

- `data/artifacts/priority_baseline.joblib`
- `data/artifacts/priority_baseline_report.json`

### 6) ML Priority Prediction (single ticket)

```bash
uv run python -m src.ML.predict_priority --ticket "URGENT: internet down cannot login now" --json
```

### 7) LLM Zero-Shot Priority Prediction

```bash
uv run python -m src.ML.predict_zero_shot --ticket "internet down cannot login now" --json
```

### 8) Compare ML vs Zero-Shot in one output

```bash
uv run python -m src.ML.predict_compare --ticket "internet down cannot login now" --json
```

### 9) One-shot local training script (Windows PowerShell)

```powershell
.\scripts\run_training_local.ps1
```

## Serving Mode Commands (Docker)

### 1) Bring up serving stack

```powershell
$env:DOCKER_DEFAULT_PLATFORM='linux/arm64'
docker compose up -d --build
docker compose ps
```

### 2) Or use the helper script

```powershell
.\scripts\run_serving_docker.ps1
```

### 3) API defaults for fast serving

- `/rag/ask` defaults to `backend=chroma` (faster than local TF-IDF path).
- ML endpoints remain available, but they require training dependencies/artifacts.

## Expected JSON Outputs

- RAG triage:
  - `rag_answer`
  - `non_rag_answer`
  - `top_answer_tweet_id`
- ML prediction:
  - `predicted_priority_label`
  - `predicted_priority`
  - `probabilities`
- Zero-shot prediction:
  - `priority`
  - `confidence`
  - `rationale`
  - `next_action`

## Troubleshooting

- If you see import errors, run commands from project root and use `python -m ...`.
- If LLM key errors appear, check `.env` values and placeholders.
- If Chroma retrieval returns empty/noisy results, rebuild index with `src.rag.index_rag`.
