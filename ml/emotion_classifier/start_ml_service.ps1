param(
    [switch]$Detach,
    [string]$Host = '127.0.0.1',
    [int]$Port = 8001,
    [string]$LogLevel = 'info'
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = $env:EMOTION_ML_PYTHON
if (-not $venvPython) {
    $venvPython = Join-Path $scriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path $venvPython)) {
    $resolvedPython = Get-Command $venvPython -ErrorAction SilentlyContinue
    if (-not $resolvedPython) {
        Write-Error "Python interpreter '$venvPython' was not found. Set EMOTION_ML_PYTHON or create the classifier .venv."
        exit 1
    }
    $venvPython = $resolvedPython.Source
}

$uvicornArgs = @('-m', 'uvicorn', 'api_service:app', '--host', $Host, '--port', $Port, '--log-level', $LogLevel)

if ($Detach) {
    # Start detached background process using the venv python
    Start-Process -FilePath $venvPython -ArgumentList $uvicornArgs -WorkingDirectory $scriptRoot -WindowStyle Hidden -ErrorAction Stop
    Write-Output "Started detached ML service (python: $venvPython) listening on $Host`:$Port"
} else {
    # Run in the foreground so logs appear in the current console
    & $venvPython @uvicornArgs
}
