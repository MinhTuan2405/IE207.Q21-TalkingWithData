# Script to setup custom Open WebUI
# Run this to clone and prepare Open WebUI for customization

Write-Host "Setting up custom Open WebUI..." -ForegroundColor Green

# Create ui directory if not exists
if (-not (Test-Path "ui")) {
    New-Item -ItemType Directory -Path "ui"
}

Set-Location "ui"

# Clone Open WebUI if not already cloned
if (-not (Test-Path "open-webui")) {
    Write-Host "Cloning Open WebUI repository..." -ForegroundColor Yellow
    git clone https://github.com/open-webui/open-webui.git
} else {
    Write-Host "Open WebUI already exists, pulling latest changes..." -ForegroundColor Yellow
    Set-Location "open-webui"
    git pull
    Set-Location ..
}

Set-Location "open-webui"

Write-Host "`nOpen WebUI source code ready!" -ForegroundColor Green
Write-Host "`nCustomization tips:" -ForegroundColor Cyan
Write-Host "1. Frontend files: ./src/" -ForegroundColor White
Write-Host "2. Backend files: ./backend/" -ForegroundColor White
Write-Host "3. Styles: ./src/lib/styles/" -ForegroundColor White
Write-Host "4. Components: ./src/lib/components/" -ForegroundColor White
Write-Host "`nAfter customizing, run: docker-compose build open-webui" -ForegroundColor Cyan

Set-Location ../..
