$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appUrl = "http://127.0.0.1:8501"
$ollamaUrl = "http://127.0.0.1:11434"
$ollamaTagsUrl = "$ollamaUrl/api/tags"

function Test-ServiceReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    try {
        Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found at $venvPython"
}

$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCommand) {
    throw "Ollama is not installed or not available on PATH."
}

$ollamaReady = Test-ServiceReady -Url $ollamaTagsUrl
if (-not $ollamaReady) {
    $existingOllama = Get-CimInstance Win32_Process -Filter "Name = 'ollama.exe'" |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*serve*"
        } |
        Select-Object -First 1

    if (-not $existingOllama) {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve" | Out-Null
    }

    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        if (Test-ServiceReady -Url $ollamaTagsUrl) {
            $ollamaReady = $true
            break
        }
    }
}

if (-not $ollamaReady) {
    throw "Ollama did not become ready at $ollamaUrl. Start it manually with 'ollama serve' and try again."
}

$existingProcess = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
    Where-Object {
        $_.CommandLine -and
        $_.CommandLine -like "*streamlit*" -and
        $_.CommandLine -like "*app.py*" -and
        $_.CommandLine -like "*blackbox_explainer*"
    } |
    Select-Object -First 1

if (-not $existingProcess) {
    $launchCommand = @(
        "Set-Location '$projectRoot'"
        "`$env:BBE_BASE_URL = '$ollamaUrl'"
        "& '$venvPython' -c ""import streamlit""" 
        "if (`$LASTEXITCODE -ne 0) { & '$venvPython' -m pip install -r requirements.txt }"
        "& '$venvPython' -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501"
    ) -join "; "

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $launchCommand | Out-Null

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            Invoke-WebRequest -Uri $appUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
            break
        }
        catch {
        }
    }
}

Start-Process $appUrl
