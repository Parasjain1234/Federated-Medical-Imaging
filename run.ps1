# Camelyon17 Research Platform - PowerShell Startup Script
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Camelyon17 Research Demonstration Platform" -ForegroundColor White
Write-Host " Federated Learning Research Environment" -ForegroundColor Gray
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Platform available at: http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

# Use system Python uvicorn (venv may be blocked by Application Control in OneDrive)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
