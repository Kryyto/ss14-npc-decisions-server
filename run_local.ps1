# Runs the THX-01 typed-decision API next to an SS14 server (Windows).
# First run creates .venv and installs torch (CPU wheel, or CUDA with -Cuda) + requirements; later runs
# reuse them and reinstall only when requirements.txt or the torch flavour changes. Variables already set
# in the environment (or in .env) win over the defaults below.
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8095,
    [int]$Threads = 2,
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$marker = Join-Path $PSScriptRoot ".venv\.requirements.sha256"
$torchIndex = if ($Cuda) { "https://download.pytorch.org/whl/cu126" } else { "https://download.pytorch.org/whl/cpu" }

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed (Python 3.10+ required)" }
}

$hash = (Get-FileHash requirements.txt -Algorithm SHA256).Hash + "|" + $torchIndex
$installed = if (Test-Path $marker) { (Get-Content $marker -Raw).Trim() } else { "" }

if ($hash -ne $installed) {
    Write-Host "Installing dependencies from $torchIndex (this can take a while)..."
    & $python -m pip install --upgrade pip
    & $python -m pip install torch --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
    Set-Content -Path $marker -Value $hash -Encoding ascii
}

if ((Test-Path .env.example) -and -not (Test-Path .env)) {
    Copy-Item .env.example .env
}

function Set-Default([string]$Name, [string]$Value) {
    if (-not [Environment]::GetEnvironmentVariable($Name)) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

Set-Default "OMP_NUM_THREADS" "$Threads"
Set-Default "MKL_NUM_THREADS" "$Threads"

Write-Host "Starting on http://${ListenHost}:${Port} (first start downloads the ~650 MB model)..."
& $python -m app.main --host $ListenHost --port $Port
exit $LASTEXITCODE
