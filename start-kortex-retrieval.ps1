<#
.SYNOPSIS
    Build, index, and serve Kortex retrieval for the vscodium-rust IDE.

.DESCRIPTION
    Brings up the retrieval stack in one command:

      1. builds aim-index and aim-proxy in release mode
      2. builds the .aim catalog over the workspace, if one is missing
      3. starts aim-proxy on 127.0.0.1:1536 with the filesystem watcher

    Once running, point the IDE's AI provider at http://127.0.0.1:1536.
    Requests are augmented with retrieved workspace context on the way
    through; the watcher keeps that context current as files change, so
    the catalog only needs a full rebuild when you want to re-tune
    chunking or swap embedders.

    Retrieval degrades safely. If the catalog is missing, the query is
    conversational, nothing clears the relevance gate, or retrieval
    exceeds its latency budget, the request is forwarded unchanged.

.PARAMETER Workspace
    Workspace root to index and watch. Defaults to this script's parent
    directory, which is the vscodium-rust root.

.PARAMETER Rebuild
    Rebuild the catalog even if one already exists.

.PARAMETER Upstream
    Where to forward Ollama-dialect traffic.

.PARAMETER OpenAiUpstream
    Where to forward OpenAI-dialect traffic. Defaults to Lemonade.

.PARAMETER Model
    Embedding model for a dense catalog. Omit to use the built-in
    lexical embedder, which needs no server and no model.

.EXAMPLE
    .\start-kortex-retrieval.ps1

.EXAMPLE
    .\start-kortex-retrieval.ps1 -Rebuild -Model nomic-embed-text
#>
[CmdletBinding()]
param(
    [string]$Workspace,
    [switch]$Rebuild,
    [string]$Upstream = "http://127.0.0.1:11434",
    [string]$OpenAiUpstream = "http://localhost:13305",
    [string]$Model
)

$ErrorActionPreference = "Stop"

$KortexRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Workspace) {
    $Workspace = Split-Path -Parent $KortexRoot
}
$Workspace = (Resolve-Path $Workspace).Path
$CatalogDir = Join-Path $Workspace ".aim"

Write-Host "Kortex retrieval" -ForegroundColor Cyan
Write-Host "  workspace  $Workspace"
Write-Host "  catalog    $CatalogDir"

# --- 1. Build -----------------------------------------------------------
# `--features libaim/http-embed` is only needed for dense embeddings, but
# it costs nothing to always compile it in; the backend is chosen at run
# time by whether -Model was passed.
Write-Host "`nBuilding (release)..." -ForegroundColor Cyan
Push-Location $KortexRoot
try {
    cargo build --release -p aim-proxy
    if ($LASTEXITCODE -ne 0) { throw "aim-proxy build failed" }

    cargo build --release -p libaim --bin aim-index --features libaim/http-embed
    if ($LASTEXITCODE -ne 0) { throw "aim-index build failed" }
}
finally {
    Pop-Location
}

$AimIndex = Join-Path $KortexRoot "target\release\aim-index.exe"
$AimProxy = Join-Path $KortexRoot "target\release\aim-proxy.exe"

# --- 2. Index -----------------------------------------------------------
$containerPath = Join-Path $CatalogDir "catalog.aim"
$needsIndex = $Rebuild -or -not (Test-Path $containerPath)

if ($needsIndex) {
    Write-Host "`nBuilding catalog (this walks the whole tree)..." -ForegroundColor Cyan
    $indexArgs = @("build", $Workspace, "--out", $CatalogDir)
    if ($Model) {
        # `auto` probes Lemonade, then OpenAI-compatible, then Ollama.
        $indexArgs += @("--model", $Model, "--backend", "auto")
    }
    & $AimIndex @indexArgs
    if ($LASTEXITCODE -ne 0) { throw "catalog build failed" }
}
else {
    Write-Host "`nCatalog already present; pass -Rebuild to rebuild it." -ForegroundColor DarkGray
    & $AimIndex stats $CatalogDir
}

# --- 3. Serve -----------------------------------------------------------
$env:KORTEX_AIM_CATALOG = $CatalogDir
$env:KORTEX_WORKSPACE = $Workspace
$env:KORTEX_UPSTREAM_OLLAMA = $Upstream
$env:KORTEX_UPSTREAM_OPENAI = $OpenAiUpstream

Write-Host "`nPoint the IDE's AI provider at http://127.0.0.1:1536" -ForegroundColor Green
Write-Host "  Ollama dialect    /api/chat, /api/generate"
Write-Host "  OpenAI dialect    /v1/chat/completions"
Write-Host "  Anthropic dialect /v1/messages"
Write-Host "Ctrl-C to stop.`n" -ForegroundColor DarkGray

& $AimProxy
