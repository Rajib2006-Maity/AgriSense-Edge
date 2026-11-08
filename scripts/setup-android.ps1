# Creates the missing React Native Android project scaffold, then installs the
# AgriSense camera UI and ExecuTorch native bridge from mobile/AgriSenseMobile.
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceApp = Join-Path $projectRoot 'mobile\AgriSenseMobile'
$targetApp = Join-Path $projectRoot 'mobile\AgriSenseMobileApp'

function Require-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. $InstallHint"
    }
}

Require-Command 'node' 'Install Node.js 22 LTS, then reopen VS Code.'
Require-Command 'npm' 'Install Node.js 22 LTS, then reopen VS Code.'
Require-Command 'java' 'Install JDK 17 and configure JAVA_HOME.'

$nodeMajor = [int]((node -p "process.versions.node.split('.')[0]").Trim())
if ($nodeMajor -lt 18) { throw "Node.js 18 or newer is required. Detected: $(node --version)" }

if (-not (Test-Path (Join-Path $sourceApp 'App.tsx'))) {
    throw "Source app files were not found at $sourceApp"
}
if (Test-Path (Join-Path $targetApp 'android\gradlew.bat')) {
    throw "A generated app already exists at $targetApp. Delete/rename it first if you want to regenerate it."
}
if (Test-Path $targetApp) {
    $existing = Get-ChildItem -Force $targetApp -ErrorAction SilentlyContinue
    if ($existing) { throw "Target folder $targetApp is not empty. Move it aside before running setup." }
}

Write-Host "`n[1/7] Creating React Native 0.75.4 Android/iOS scaffold..." -ForegroundColor Cyan
Push-Location $projectRoot
try {
    npx --yes @react-native-community/cli@14.1.0 init AgriSenseMobile --version 0.75.4 --directory $targetApp --pm npm
    if ($LASTEXITCODE -ne 0) { throw 'React Native scaffold generation failed.' }
} finally {
    Pop-Location
}

Write-Host "`n[2/7] Copying AgriSense app source..." -ForegroundColor Cyan
Copy-Item (Join-Path $sourceApp 'App.tsx') (Join-Path $targetApp 'App.tsx') -Force
$sourceSrc = Join-Path $sourceApp 'src'
$targetSrc = Join-Path $targetApp 'src'
if (Test-Path $targetSrc) { Remove-Item $targetSrc -Recurse -Force }
Copy-Item $sourceSrc $targetSrc -Recurse -Force

Write-Host "`n[3/7] Installing camera and file-system packages..." -ForegroundColor Cyan
Push-Location $targetApp
try {
    npm install react-native-vision-camera@4.5.3 react-native-fs@2.20.0 react-native-worklets-core@1.3.3
    if ($LASTEXITCODE -ne 0) { throw 'npm dependency installation failed.' }
} finally {
    Pop-Location
}
$babelPath = Join-Path $targetApp 'babel.config.js'
$babelText = Get-Content $babelPath -Raw
if ($babelText -notmatch 'react-native-worklets-core/plugin') {
    $babelText = $babelText -replace 'presets:\s*(\[[^\]]*\])\s*,?', "presets: `$1,`r`n  plugins: ['react-native-worklets-core/plugin'],"
    Set-Content -Path $babelPath -Value $babelText -Encoding UTF8
}

Write-Host "`n[4/7] Registering the Android ExecuTorch bridge..." -ForegroundColor Cyan
$mainApplication = Get-ChildItem (Join-Path $targetApp 'android\app\src\main\java') -Filter 'MainApplication.kt' -Recurse | Select-Object -First 1
if (-not $mainApplication) { throw 'Could not find MainApplication.kt in the generated Android project.' }
$nativeDir = $mainApplication.Directory.FullName
Copy-Item (Join-Path $sourceApp 'android\app\src\main\java\com\agrisensemobile\ExecuTorchModule.kt') (Join-Path $nativeDir 'ExecuTorchModule.kt') -Force
Copy-Item (Join-Path $sourceApp 'android\app\src\main\java\com\agrisensemobile\ExecuTorchPackage.kt') (Join-Path $nativeDir 'ExecuTorchPackage.kt') -Force

$appPackage = (Get-Content $mainApplication.FullName | Select-String '^package\s+(.+)$').Matches.Groups[1].Value.Trim()
foreach ($kotlinFile in @('ExecuTorchModule.kt', 'ExecuTorchPackage.kt')) {
    $kotlinPath = Join-Path $nativeDir $kotlinFile
    $kotlinText = Get-Content $kotlinPath -Raw
    $kotlinText = [regex]::Replace($kotlinText, '(?m)^package\s+[^\r\n]+', "package $appPackage", 1)
    Set-Content -Path $kotlinPath -Value $kotlinText -Encoding UTF8
}
$mainText = Get-Content $mainApplication.FullName -Raw
if ($mainText -notmatch 'ExecuTorchPackage\(') {
    $pattern = 'PackageList\(this\)\.packages\.apply\s*\{'
    if ($mainText -notmatch $pattern) { throw 'Could not locate the React Native package registration block in MainApplication.kt.' }
    $mainText = [regex]::Replace($mainText, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $m.Value + "`r`n      add(ExecuTorchPackage())" }, 1)
    Set-Content -Path $mainApplication.FullName -Value $mainText -Encoding UTF8
}

Write-Host "`n[5/7] Updating Android permissions and Gradle configuration..." -ForegroundColor Cyan
$manifestPath = Join-Path $targetApp 'android\app\src\main\AndroidManifest.xml'
$manifest = Get-Content $manifestPath -Raw
if ($manifest -notmatch 'android.permission.CAMERA') {
    $manifest = [regex]::Replace($manifest, '(<manifest[^>]*>)', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $m.Value + "`r`n    <uses-permission android:name=`"android.permission.CAMERA`" />" }, 1)
    Set-Content -Path $manifestPath -Value $manifest -Encoding UTF8
}

$gradlePath = Join-Path $targetApp 'android\app\build.gradle'
$gradle = Get-Content $gradlePath -Raw
if ($gradle -notmatch 'executorch-android') {
    $gradle = [regex]::Replace($gradle, '(?m)^dependencies\s*\{', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $m.Value + "`r`n    implementation('org.pytorch:executorch-android:0.4.0')" }, 1)
}
if ($gradle -notmatch 'noCompress.*pte') {
    $gradle = [regex]::Replace($gradle, '(?m)^android\s*\{', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $m.Value + "`r`n    androidResources { noCompress += ['pte'] }" }, 1)
}
Set-Content -Path $gradlePath -Value $gradle -Encoding UTF8

$propertiesPath = Join-Path $targetApp 'android\gradle.properties'
$properties = Get-Content $propertiesPath -Raw
if ($properties -match '(?m)^newArchEnabled=') {
    $properties = [regex]::Replace($properties, '(?m)^newArchEnabled=.*$', 'newArchEnabled=true')
} else {
    $properties += "`r`nnewArchEnabled=true`r`n"
}
Set-Content -Path $propertiesPath -Value $properties -Encoding UTF8

Write-Host "`n[6/7] Creating the model asset folder..." -ForegroundColor Cyan
$modelsPath = Join-Path $targetApp 'android\app\src\main\assets\models'
New-Item -ItemType Directory -Force -Path $modelsPath | Out-Null
@'
Copy the exported .pte files into this folder before running real AI inference.
Required for FP32 mode:
- cnn_classifier_fp32.pte
- unet_segmentation_fp32.pte
Optional for INT8 mode:
- cnn_classifier_int8.pte
- unet_segmentation_int8.pte
No model files are included in the source ZIP.
'@ | Set-Content -Path (Join-Path $modelsPath 'README.txt') -Encoding UTF8

Write-Host "`n[7/7] Android project scaffold is ready." -ForegroundColor Green
Write-Host "Project: $targetApp"
Write-Host 'Next: add .pte models with scripts/copy-models.ps1, then run npm run android from the generated app folder.'
Write-Host 'You can also start without models: the app offers a clearly labelled camera preview-only mode.'
