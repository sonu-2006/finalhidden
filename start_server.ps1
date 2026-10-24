Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting Medical Hallucination Detection Platform (Group 76)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = "python"
if (Test-Path "C:\Users\gnyan\Downloads\RTHD\.venv\Scripts\python.exe") {
    $pythonCmd = "C:\Users\gnyan\Downloads\RTHD\.venv\Scripts\python.exe"
}

Write-Host "Using Python: $pythonCmd" -ForegroundColor Green
Write-Host "Opening platform at: http://127.0.0.1:8000" -ForegroundColor Yellow
Start-Process "http://127.0.0.1:8000"

& $pythonCmd -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
