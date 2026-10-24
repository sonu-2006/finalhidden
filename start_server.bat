@echo off
echo ======================================================================
echo Starting Medical Hallucination Detection Platform (Group 76)
echo ======================================================================
echo.

set PYTHON_CMD=python
if exist "C:\Users\gnyan\Downloads\RTHD\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=C:\Users\gnyan\Downloads\RTHD\.venv\Scripts\python.exe"
)

echo Using Python: %PYTHON_CMD%
echo Opening platform in default browser: http://127.0.0.1:8000
echo.
start http://127.0.0.1:8000
"%PYTHON_CMD%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
