$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$artifactDirectory = Join-Path $projectDirectory 'artifacts'
New-Item -ItemType Directory -Path $artifactDirectory -Force | Out-Null
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -like '*-m fashion_ai.full_fashionpedia*'
}
if ($active) {
    Write-Output 'Fashionpedia pipeline is already running.'
    exit 0
}
$worker = Start-Process -FilePath $pythonExecutable `
    -ArgumentList @('-u', '-X', 'utf8', '-m', 'fashion_ai.full_fashionpedia') `
    -WorkingDirectory $projectDirectory -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $artifactDirectory 'fashionpedia-training.log') `
    -RedirectStandardError (Join-Path $artifactDirectory 'fashionpedia-training-errors.log') `
    -PassThru
Write-Output "Started full Fashionpedia pipeline. PID: $($worker.Id)"
Write-Output (Join-Path $artifactDirectory 'fashionpedia-training.log')
