$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$stateRoot=Join-Path $projectRoot 'datasets/Stylique-Polyvore/.download'
$pidFile=Join-Path $stateRoot 'background-pid.txt'
if(Test-Path -LiteralPath $pidFile){
    $downloadProcessId=[int](Get-Content -LiteralPath $pidFile)
    $existing=Get-CimInstance Win32_Process -Filter "ProcessId=$downloadProcessId" -ErrorAction SilentlyContinue
    if($existing -and $existing.CommandLine -like '*download-stylique-ranges.mjs*'){Write-Output "Tiến trình tải đang chạy: PID $downloadProcessId";exit 0}
}
$nodeBinary=(Get-Command node -ErrorAction Stop).Source
$downloadScript=Join-Path $projectRoot 'scripts/download-stylique-ranges.mjs'
$process=Start-Process -FilePath $nodeBinary -ArgumentList @('"'+$downloadScript+'"') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $stateRoot 'download.log') -RedirectStandardError (Join-Path $stateRoot 'download-error.log') -PassThru
$process.Id | Set-Content -LiteralPath $pidFile
Write-Output "Đã chạy tải nền. PID: $($process.Id)"
Write-Output "Trạng thái: $(Join-Path $stateRoot 'download-report.json')"
