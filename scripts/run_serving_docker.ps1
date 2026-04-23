param(
    [string]$Platform = "linux/arm64"
)

$ErrorActionPreference = "Stop"

Write-Host "== Serving mode (Docker) =="
$env:DOCKER_DEFAULT_PLATFORM = $Platform

docker compose config --quiet
docker compose build backend frontend
docker compose up -d
docker compose ps

Write-Host "Serving mode is up."
