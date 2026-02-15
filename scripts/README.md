# Scripts Directory

Thư mục chứa các scripts tiện ích cho dự án TalkWithData.

## 📁 Danh sách Scripts

### 1. **pull-ollama-models.ps1** (Windows)
Script PowerShell để pull Ollama models sau khi services đã khởi động.

**Sử dụng:**
```powershell
# Từ thư mục gốc của project
.\scripts\pull-ollama-models.ps1
```

### 2. **pull-ollama-models.sh** (Linux/Mac)
Script Bash để pull Ollama models sau khi services đã khởi động.

**Sử dụng:**
```bash
# Từ thư mục gốc của project
bash scripts/pull-ollama-models.sh

# Hoặc chạy trong container
docker exec talkwdata_ollama /bin/bash /root/pull-models.sh
```

### 3. **init-databases.sh** (Linux - Docker)
Script tự động tạo database cho TalkWithData server khi PostgreSQL khởi động.

**Tự động chạy:** Script này được mount vào PostgreSQL container và tự động chạy khi container khởi động lần đầu.

### 4. **setup.ps1** (Windows)
Script thiết lập môi trường ban đầu cho dự án.

## 🚀 Quick Start

### Windows
```powershell
# 1. Khởi động services
docker-compose up -d

# 2. Pull Ollama models
.\scripts\pull-ollama-models.ps1

# 3. Kiểm tra
docker exec talkwdata_ollama ollama list
```

### Linux/Mac
```bash
# 1. Khởi động services
docker-compose up -d

# 2. Pull Ollama models (chạy trong container)
docker exec talkwdata_ollama /bin/bash /root/pull-models.sh

# 3. Kiểm tra
docker exec talkwdata_ollama ollama list
```

## 📝 Lưu ý

- Models được lưu trong `../volumes/ollama/` nên chỉ cần pull 1 lần
- Mỗi model có thể nặng vài GB, cần đủ dung lượng đĩa
- Database tự động được tạo khi khởi động PostgreSQL lần đầu
