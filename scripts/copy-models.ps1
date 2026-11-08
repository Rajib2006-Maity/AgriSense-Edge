param(
    [Parameter(Mandatory = $true)]
    [string]$PteDir
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$targetApp = Join-Path $projectRoot 'mobile\AgriSenseMobileApp'
$destination = Join-Path $targetApp 'android\app\src\main\assets\models'

if (-not (Test-Path $destination)) {
    throw "Android project not found. Run scripts/setup-android.ps1 first. Expected: $destination"
}
if (-not (Test-Path $PteDir)) { throw "PTE directory not found: $PteDir" }

$required = @('cnn_classifier_fp32.pte', 'unet_segmentation_fp32.pte')
$missing = @($required | Where-Object { -not (Test-Path (Join-Path $PteDir $_)) })
if ($missing.Count -gt 0) {
    throw "FP32 model files are required but missing: $($missing -join ', '). Export the models before copying."
}

$files = @(
    'cnn_classifier_fp32.pte',
    'unet_segmentation_fp32.pte',
    'cnn_classifier_int8.pte',
    'unet_segmentation_int8.pte'
)
foreach ($file in $files) {
    $source = Join-Path $PteDir $file
    if (Test-Path $source) {
        Copy-Item $source $destination -Force
        Write-Host "Copied $file"
    } else {
        Write-Host "Skipped optional file: $file" -ForegroundColor Yellow
    }
}
Write-Host "Model files copied to: $destination" -ForegroundColor Green
Write-Host 'FP32 mode is the default. Select INT8 only after both INT8 files are present.'
