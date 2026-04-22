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

## Requirements

- Python `>=3.10`
- `uv` installed

Install dependencies:

```bash
uv sync
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

## Run Commands

Always run from repo root with module mode (`-m`).

### 1) Prepare Datasets (shared for RAG + ML)

```bash
uv run python -m src.prepare_datasets
```

Outputs include:

- `data/processed/retrieval_corpus.csv`
- `data/processed/ml_train.csv`
- `data/processed/ml_val.csv`
- `data/processed/ml_test.csv`

### 2) Build RAG Index (Chroma + sentence-transformers embeddings)

```bash
uv run python -m src.rag.index_rag --backend sbert --storage chroma --chroma-collection rag_tickets --embed-model sentence-transformers/all-MiniLM-L6-v2
```

### 3) Query Retrieval Only

```bash
uv run python -m src.rag.retrieve_rag --query "internet down cannot login" --backend chroma --chroma-collection rag_tickets --k 5 --json
```

### 4) Full RAG Triage (RAG answer + non-RAG answer)

```bash
uv run python -m src.rag.triage_with_rag --ticket "internet down cannot login" --retrieval-backend chroma --chroma-collection rag_tickets --embed-model sentence-transformers/all-MiniLM-L6-v2 --k 5 --json
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
