@echo off
REM Camelyon17 Research Platform - Windows Startup Script
echo =====================================================
echo  Camelyon17 Research Demonstration Platform
echo  Federated Learning Research Environment
echo =====================================================
echo.

echo Starting FastAPI inference server...
echo Platform will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.

REM Use system Python (venv may be blocked by Application Control policies in OneDrive locations)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
