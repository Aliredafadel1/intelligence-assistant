param(
    [string]$EmbedModel = "text-embedding-3-small",
    [string]$ChromaCollection = "rag_tickets"
)

$ErrorActionPreference = "Stop"

Write-Host "== Training mode (local, no Docker) =="
Write-Host "Installing training dependencies..."
python -m pip install --upgrade pip
pip install -r "requirements/train.txt"

Write-Host "Preparing datasets..."
python -m src.prepare_datasets

Write-Host "Building OpenAI embedding index into Chroma..."
python -m src.rag.index_rag --backend openai --storage chroma --chroma-collection $ChromaCollection --embed-model $EmbedModel

Write-Host "Training ML baseline artifact..."
python -m src.ML.train_priority_baseline

Write-Host "Training mode completed."
