# Ollama Model Downloader Script for Windows
# Chạy script này sau khi đã khởi động services

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Ollama Model Downloader" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Ollama service có chạy không
Write-Host "🔍 Checking Ollama service status..." -ForegroundColor Yellow
$ollamaRunning = docker ps --filter "name=talkwdata_ollama" --filter "status=running" --format "{{.Names}}"

if (-not $ollamaRunning) {
    Write-Host "❌ Ollama service is not running!" -ForegroundColor Red
    Write-Host "   Please start services first: docker-compose up -d" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Ollama service is running!" -ForegroundColor Green
Write-Host ""

# Chờ service sẵn sàng
Write-Host "⏳ Waiting for Ollama to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Kiểm tra models hiện có
Write-Host "📋 Checking existing models..." -ForegroundColor Yellow
docker exec talkwdata_ollama ollama list
Write-Host ""

# Pull llama3.2 model
$checkLlama = docker exec talkwdata_ollama ollama list | Select-String "llama3.2"
if (-not $checkLlama) {
    Write-Host "📥 Downloading llama3.2 model..." -ForegroundColor Yellow
    Write-Host "   (This may take several minutes depending on your internet speed)" -ForegroundColor Gray
    docker exec talkwdata_ollama ollama pull llama3.2
    Write-Host "✅ llama3.2 model downloaded successfully!" -ForegroundColor Green
} else {
    Write-Host "✅ llama3.2 model already exists (skipping download)" -ForegroundColor Green
}

Write-Host ""

# Pull nomic-embed-text model
$checkNomic = docker exec talkwdata_ollama ollama list | Select-String "nomic-embed-text"
if (-not $checkNomic) {
    Write-Host "📥 Downloading nomic-embed-text model for embeddings..." -ForegroundColor Yellow
    docker exec talkwdata_ollama ollama pull nomic-embed-text
    Write-Host "✅ nomic-embed-text model downloaded successfully!" -ForegroundColor Green
} else {
    Write-Host "✅ nomic-embed-text model already exists (skipping download)" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ All required models are ready!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available models:" -ForegroundColor Cyan
docker exec talkwdata_ollama ollama list
Write-Host ""
Write-Host "Models are saved in: ./volumes/ollama" -ForegroundColor Gray
Write-Host "You don't need to download them again after restart!" -ForegroundColor Gray
