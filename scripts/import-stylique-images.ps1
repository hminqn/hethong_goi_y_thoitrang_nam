$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$archive=Join-Path $projectRoot 'datasets/Stylique-Polyvore/images.zip'
if(!(Test-Path -LiteralPath $archive)){throw 'images.zip chưa tải xong. Không sử dụng ảnh từ dataset cũ.'}
$repository=Get-Content -LiteralPath 'datasets/Stylique-Polyvore/.download/repository.json' -Raw | ConvertFrom-Json
$expected=$repository.siblings | Where-Object rfilename -eq 'images.zip'
if((Get-Item -LiteralPath $archive).Length -ne $expected.size){throw 'images.zip chưa đủ byte.'}
if((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.lfs.sha256){throw 'images.zip không khớp SHA-256.'}
$text=[IO.File]::ReadAllText((Join-Path $projectRoot 'web/catalog.js'))
$catalog=$text.Substring('window.CATALOG = '.Length).Trim().TrimEnd(';') | ConvertFrom-Json
if(!$catalog.ready -or $catalog.source -ne 'https://huggingface.co/datasets/Stylique/Polyvore'){throw 'Cần học mô hình Stylique trước khi nhập ảnh.'}
$ids=[Collections.Generic.HashSet[string]]::new()
foreach($item in $catalog.items){[void]$ids.Add([string]$item.id)}
Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force 'web/images' | Out-Null
$zip=[IO.Compression.ZipFile]::OpenRead($archive)
$count=0
try{foreach($entry in $zip.Entries){$basename=[IO.Path]::GetFileName($entry.FullName);if($basename -match '^(\d+)\.(jpg|png|webp)$' -and $entry.Length -lt 20MB){$id=$Matches[1];if($ids.Contains($id)){$destination=Join-Path $projectRoot "web/images/$basename";[IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$destination,$true);$count++}}}}finally{$zip.Dispose()}
& node scripts/index-images.mjs
if($LASTEXITCODE -ne 0){throw 'Không tạo được chỉ mục ảnh.'}
Write-Output "Đã nhập $count ảnh thật từ Stylique/Polyvore."
