Write-Host '=== AgriSense development environment ===' -ForegroundColor Cyan
foreach ($command in @('node', 'npm', 'java', 'adb')) {
    $found = Get-Command $command -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "$command : $($found.Source)" -ForegroundColor Green
    } else {
        Write-Host "$command : NOT FOUND" -ForegroundColor Yellow
    }
}
Write-Host "`nNode version:"; node --version 2>$null
Write-Host "npm version:"; npm --version 2>$null
Write-Host "`nJava version:"; java -version 2>&1
Write-Host "`nANDROID_HOME: $env:ANDROID_HOME"
Write-Host "ANDROID_SDK_ROOT: $env:ANDROID_SDK_ROOT"
if (Get-Command adb -ErrorAction SilentlyContinue) { adb version }
Write-Host "`nFor Android, install JDK 17 and Android Studio (SDK Platform + Build-Tools + Platform-Tools)."
Write-Host 'For iOS builds, a Mac with Xcode is required.'
