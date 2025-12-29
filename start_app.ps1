# Mandala Laser Engraving App - PowerShell Launcher
Write-Host "Starting Mandala Laser Engraving App..." -ForegroundColor Cyan
Write-Host ""

# Find Python 3.12 with streamlit
$pythonPaths = @(
    "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12*\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Microsoft\WindowsApps\python.exe",
    "C:\Python312\python.exe"
)

$python = $null
foreach ($path in $pythonPaths) {
    $resolved = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($resolved) {
        $python = $resolved.FullName
        break
    }
}

if (-not $python) {
    # Try default python and check for streamlit
    $python = "python"
}

Write-Host "Using Python: $python" -ForegroundColor Green
Write-Host "Starting Enhanced Version (20 layers)..." -ForegroundColor Yellow
Write-Host ""
Write-Host "The app will open in your default browser at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

& $python -m streamlit run app_enhanced.py

pause
