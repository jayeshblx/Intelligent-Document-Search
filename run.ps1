# Launch ExampleApp (hybrid search + credit meter) on http://127.0.0.1:8004
#
# Use THIS instead of a bare `uvicorn ...` — your terminal often has the old
# `Documents\POC\.venv` virtualenv active, whose code has no credit meter. This
# script pins the correct interpreter (this folder's .venv) and source path, and
# clears any stale server already on 8004, so the credit meter always shows up.
#
#   .\run.ps1
#
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "venv python not found at $py" }

# Force this project's source onto the path (independent of any active venv).
$srcDirs = 'services\search\src', 'packages\core\src', 'services\processing\src', 'services\ingestion\src'
$env:PYTHONPATH = ($srcDirs | ForEach-Object { Join-Path $root $_ }) -join ';'

# Free port 8004 if a stale server (e.g. one started from the wrong venv) holds it.
$stale = Get-NetTCPConnection -LocalPort 8004 -State Listen -ErrorAction SilentlyContinue
if ($stale) {
    $stale | Select-Object -Unique OwningProcess | ForEach-Object {
        Write-Host "Stopping stale server on 8004 (PID $($_.OwningProcess))..." -ForegroundColor Yellow
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 1000
}

Write-Host "Starting ExampleApp on http://127.0.0.1:8004  (Ctrl+C to stop)" -ForegroundColor Green
& $py -m uvicorn rag_search.app:app --port 8004
