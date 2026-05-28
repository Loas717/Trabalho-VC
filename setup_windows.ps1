$ErrorActionPreference = "Stop"

$localPython = Join-Path $PSScriptRoot ".python311\python.exe"
$python311 = $null

if (Test-Path $localPython) {
    $python311 = $localPython
} else {
    $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $python311 = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        $ErrorActionPreference = $previousErrorActionPreference
        if ($LASTEXITCODE -ne 0) {
            $python311 = $null
        }
    }
}

if (-not $python311) {
    Write-Host "Python 3.11 nao encontrado."
    Write-Host "Instale o Python 3.11 em: https://www.python.org/downloads/release/python-3119/"
    Write-Host "Ou instale localmente em .python311 usando o instalador oficial."
    Write-Host "Depois rode novamente: .\setup_windows.ps1"
    exit 1
}

if (-not (Test-Path ".venv")) {
    & $python311 -m venv .venv
}

Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue
Remove-Item Env:\REQUESTS_CA_BUNDLE -ErrorAction SilentlyContinue
Remove-Item Env:\CURL_CA_BUNDLE -ErrorAction SilentlyContinue

.\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Ambiente pronto."
Write-Host "Para ativar:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Para marcar vagas:"
Write-Host "  python selector.py"
Write-Host ""
Write-Host "Para rodar a deteccao:"
Write-Host "  python main.py"
