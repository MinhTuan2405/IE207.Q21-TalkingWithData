# TalkingWithData - Tài liệu Mô tả Hệ thống

> **Phiên bản:** 1.0 (Demo)  
> **Ngày cập nhật:** 01/03/2026  
> **Mô tả:** Nền tảng cho phép người dùng giao tiếp với dữ liệu bằng ngôn ngữ tự nhiên

---

## 1. Tổng quan Dự án

### 1.1. Giới thiệu

**TalkingWithData** là một nền tảng cho phép người dùng **truy vấn dữ liệu bằng ngôn ngữ tự nhiên** (Natural Language to SQL/Data). Thay vì phải biết SQL hay các ngôn ngữ truy vấn, người dùng chỉ cần đặt câu hỏi bằng tiếng Việt/Anh, hệ thống sẽ tự động:

1. Hiểu ý định của người dùng
2. Chuyển đổi câu hỏi thành truy vấn SQL phù hợp
3. Thực thi truy vấn trên cơ sở dữ liệu
4. Trả về kết quả dưới dạng dễ hiểu

### 1.2. Mục tiêu

| Mục tiêu | Mô tả |
|-----------|--------|
| **Text-to-SQL** | Chuyển đổi câu hỏi ngôn ngữ tự nhiên thành truy vấn SQL |
| **Quản lý hội thoại** | Lưu trữ lịch sử trò chuyện, hỗ trợ ngữ cảnh đa lượt |
| **Tìm kiếm ngữ nghĩa** | Tìm kiếm dựa trên ý nghĩa (semantic search) qua vector database |
| **AI cục bộ** | Sử dụng LLM chạy local (Ollama) — không phụ thuộc API bên ngoài |
| **Giao diện thân thiện** | UI dạng chat (Open WebUI) để tương tác tự nhiên |

### 1.3. Kiến trúc tổng quan

![Kiến trúc hệ thống](images/architecture.png)

Hệ thống gồm **6 thành phần chính** (microservices), được container hóa bằng Docker:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network                           │
│   (talkwdata_network)                                           │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ Open     │   │ FastAPI  │   │  Ollama  │   │   Qdrant    │  │
│  │ WebUI    │──▶│ Server   │──▶│  (LLM)   │   │  (Vector    │  │
│  │ :8080    │   │ :8000    │   │ :11434   │   │   DB) :6333  │  │
│  └──────────┘   └────┬─────┘   └──────────┘   └─────────────┘  │
│                      │                                          │
│                      ▼                                          │
│               ┌──────────┐         ┌───────────────────────┐    │
│               │PostgreSQL│         │   Dagster             │    │
│               │  :5432   │◀────────│   Orchestration       │    │
│               │          │         │   :3000 (UI)          │    │
│               └──────────┘         │   :4000 (gRPC)        │    │
│                                    └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Kiến trúc Chi tiết

### 2.1. Các Thành phần (Services)

| Service | Công nghệ | Port | Chức năng |
|---------|-----------|------|-----------|
| **server** | FastAPI (Python 3.11) | 8000 | API backend chính — xác thực, text-to-SQL, quản lý hội thoại |
| **postgres** | PostgreSQL 16 | 5432 | Cơ sở dữ liệu quan hệ — lưu trữ users, conversations, messages |
| **ollama** | Ollama (LLM Runtime) | 11434 | Chạy mô hình AI local (llama3.2, nomic-embed-text) |
| **qdrant** | Qdrant | 6333, 6334 | Vector database — lưu trữ embeddings cho semantic search |
| **dagster** | Dagster (Python) | 3000, 4000 | Orchestration — điều phối data pipeline (ETL, indexing) |
| **open-webui** | Open WebUI (SvelteKit) | 8080 | Giao diện người dùng dạng chat |

### 2.2. Luồng xử lý truy vấn người dùng

![Luồng xử lý truy vấn](images/user_query_flow.png)

```
Người dùng đặt câu hỏi
        │
        ▼
  ┌─────────────┐
  │  Open WebUI  │  (1) Gửi câu hỏi qua giao diện chat
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  FastAPI     │  (2) Nhận request, xác thực JWT
  │  Server      │  (3) Phân tích intent (ý định)
  │              │  (4) Lấy schema database liên quan
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Qdrant │ │Ollama │  (5) Tìm kiếm ngữ nghĩa schema phù hợp
│       │ │(LLM)  │  (6) LLM sinh câu SQL từ câu hỏi + schema
└───────┘ └───┬───┘
               │
               ▼
        ┌─────────────┐
        │  PostgreSQL  │  (7) Thực thi câu SQL
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  FastAPI     │  (8) Format kết quả
        │  Server      │  (9) Trả response cho UI
        └─────────────┘
```

### 2.3. Luồng Data Pipeline (Dagster)

![Dagster Flow](images/dagster_flow.png)

Dagster đảm nhiệm việc **điều phối các tác vụ dữ liệu**:
- Crawl/Import schema từ các database nguồn
- Tạo embeddings cho schema (bảng, cột, mô tả) 
- Đẩy embeddings vào Qdrant vector database
- Lên lịch cập nhật định kỳ

---

## 3. Server (FastAPI Backend)

### 3.1. Tổng quan

Server là **trung tâm xử lý logic** của toàn bộ hệ thống, được xây dựng bằng **FastAPI** với kiến trúc modular.

**Tech Stack:**
- **Framework:** FastAPI 0.129.0
- **ORM:** SQLAlchemy 2.0.46
- **Database Migration:** Alembic 1.18.4
- **Authentication:** JWT (python-jose) + bcrypt
- **AI Client:** ollama 0.6.1 (Python SDK)
- **HTTP Client:** httpx 0.28.1
- **Validation:** Pydantic 2.12.5

### 3.2. Cấu trúc thư mục

```
server/
├── main.py                    # Entry point — khởi tạo FastAPI app
├── requirements.txt           # Dependencies
├── .server.env                # Biến môi trường (không commit)
├── .server.env.example        # Template biến môi trường
├── alembic.ini                # Config cho database migration
│
├── alembic/                   # Database migrations
│   ├── env.py                 # Alembic environment config
│   └── versions/              # Migration files
│       └── 1c13b92c3bd5_del_is_superuser_col.py
│
├── core/                      # Lõi hệ thống (dùng chung)
│   ├── database.py            # Kết nối DB, SessionLocal, Base
│   ├── dependencies.py        # FastAPI Dependencies (auth middleware)
│   └── sercurity.py           # JWT, password hashing
│
├── module/                    # Các module nghiệp vụ
│   ├── auth/                  # ✅ Đã hoàn thành
│   ├── conversation/          # 🔲 Chưa triển khai
│   ├── message/               # 🔲 Chưa triển khai
│   ├── search/                # 🔲 Chưa triển khai
│   ├── text_to_data/          # 🔲 Chưa triển khai (core feature)
│   └── user/                  # 🔲 Chưa triển khai
│
└── shared/                    # Utilities dùng chung (trống)
```

### 3.3. Kiến trúc Module

Mỗi module tuân thủ cấu trúc **4 tầng** (Layered Architecture):

```
module/<tên_module>/
├── __init__.py       # Đăng ký router cho module
├── endpoint/         # 🌐 API Layer — Định nghĩa HTTP endpoints
│   └── *.py          #    Nhận request, gọi service, trả response
├── schema/           # 📋 Schema Layer — Pydantic models (DTO)
│   └── *.py          #    Validate input/output data
├── model/            # 🗄️ Model Layer — SQLAlchemy ORM models
│   └── *.py          #    Mapping với bảng database
└── service/          # ⚙️ Service Layer — Business logic
    └── *.py          #    Xử lý nghiệp vụ chính
```

### 3.4. Core Layer Chi tiết

#### 3.4.1. database.py — Kết nối Database

```python
# Đọc DATABASE_URL từ .server.env
# Tạo engine với connection pooling:
#   - pool_size=5, max_overflow=10
#   - pool_pre_ping=True (auto reconnect)
#   - SSL mode: require
# Cung cấp get_db() generator cho dependency injection
```

| Config | Giá trị | Mô tả |
|--------|---------|--------|
| `pool_size` | 5 | Số connection tối thiểu trong pool |
| `max_overflow` | 10 | Số connection tối đa vượt pool |
| `pool_recycle` | 3600s | Thời gian tái tạo connection |
| `pool_pre_ping` | True | Kiểm tra connection trước khi dùng |

#### 3.4.2. sercurity.py — Bảo mật & JWT

| Hàm | Mô tả |
|-----|--------|
| `hash_password(password)` | Hash password bằng bcrypt |
| `verify_password(plain, hashed)` | So sánh password |
| `create_access_token(data)` | Tạo JWT access token (mặc định 30 phút) |
| `create_refresh_token(data)` | Tạo JWT refresh token (mặc định 7 ngày) |
| `decode_token(token)` | Giải mã và xác thực JWT token |

**Cấu hình JWT:**
| Biến | Mặc định | Mô tả |
|------|----------|--------|
| `SECRET_KEY` | (bắt buộc thay đổi) | Khóa bí mật ký JWT |
| `JWT_ALGORITHM` | HS256 | Thuật toán mã hóa |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 30 | Thời hạn access token |
| `REFRESH_TOKEN_EXPIRE_DAYS` | 7 | Thời hạn refresh token |

#### 3.4.3. dependencies.py — Middleware Xác thực

Cung cấp 2 dependency cho FastAPI:

| Dependency | Phương thức | Dùng cho |
|------------|-------------|----------|
| `get_current_user` | HTTPBearer (Header) | API endpoints thông thường |
| `get_current_user_oauth2` | OAuth2PasswordBearer | Tương thích Swagger UI / OAuth2 flow |

Cả hai đều thực hiện:
1. Trích xuất token từ request
2. Decode và validate JWT
3. Kiểm tra token type = "access"
4. Tìm user trong database
5. Kiểm tra user `is_active`
6. Trả về User object

---

### 3.5. Module Auth (✅ Đã hoàn thành)

Module xác thực đầy đủ với register/login/logout.

#### 3.5.1. Data Model — User

| Cột | Kiểu | Ràng buộc | Mô tả |
|-----|------|-----------|--------|
| `id` | String (UUID) | PK | ID tự tạo UUID v4 |
| `email` | String | UNIQUE, NOT NULL, INDEX | Email đăng nhập |
| `username` | String | UNIQUE, NOT NULL, INDEX | Tên hiển thị |
| `hashed_password` | String | NOT NULL | Password đã hash bcrypt |
| `full_name` | String | nullable | Họ tên đầy đủ |
| `is_active` | Boolean | default=True | Trạng thái tài khoản |
| `created_at` | DateTime(tz) | server_default=now() | Ngày tạo |
| `updated_at` | DateTime(tz) | onupdate=now() | Ngày cập nhật |

#### 3.5.2. API Endpoints

| Method | Path | Auth | Mô tả |
|--------|------|------|--------|
| `POST` | `/auth/register` | ❌ | Đăng ký tài khoản mới |
| `POST` | `/auth/signin` | ❌ | Đăng nhập bằng email/password |
| `POST` | `/auth/signout` | ✅ | Đăng xuất (invalidate phía client) |
| `GET` | `/auth/me` | ✅ | Lấy thông tin user hiện tại |
| `POST` | `/auth/token` | ❌ | OAuth2 token endpoint (cho Swagger UI) |

#### 3.5.3. Schema (Request/Response)

**Request:**
```
UserRegister {
    email: EmailStr          # Email hợp lệ
    username: str            # 3-50 ký tự
    password: str            # 6-100 ký tự
    full_name?: str          # Tùy chọn
}

UserSignIn {
    email: EmailStr
    password: str
}
```

**Response:**
```
TokenResponse {
    access_token: str
    refresh_token: str
    token_type: "bearer"
    user: UserResponse {
        id, email, username, full_name, is_active, created_at
    }
}
```

#### 3.5.4. Business Logic (Service)

| Hàm | Luồng xử lý |
|-----|-------------|
| `register()` | Kiểm tra email/username trùng → Hash password → Tạo user → Tạo JWT tokens → Trả TokenResponse |
| `signin()` | Tìm user theo email → Verify password → Kiểm tra is_active → Tạo JWT tokens → Trả TokenResponse |
| `login_oauth2()` | Tương tự signin nhưng trả format OAuth2 (`{access_token, token_type}`) |

---

### 3.6. Module Conversation (🔲 Chưa triển khai)

**Mục đích:** Quản lý các cuộc hội thoại của người dùng.

**Chức năng dự kiến:**
- Tạo cuộc hội thoại mới
- Liệt kê cuộc hội thoại của user
- Lấy chi tiết cuộc hội thoại
- Xóa cuộc hội thoại
- Đổi tên cuộc hội thoại

**Data Model dự kiến — Conversation:**

| Cột | Kiểu | Mô tả |
|-----|------|--------|
| `id` | UUID (PK) | ID cuộc hội thoại |
| `user_id` | UUID (FK → users) | Người sở hữu |
| `title` | String | Tiêu đề hội thoại |
| `created_at` | DateTime | Ngày tạo |
| `updated_at` | DateTime | Lần cập nhật cuối |

### 3.7. Module Message (🔲 Chưa triển khai)

**Mục đích:** Quản lý tin nhắn trong hội thoại.

**Chức năng dự kiến:**
- Gửi tin nhắn (user message)
- Lưu phản hồi AI (assistant message)
- Lấy lịch sử tin nhắn theo conversation
- Lưu câu SQL đã sinh và kết quả

**Data Model dự kiến — Message:**

| Cột | Kiểu | Mô tả |
|-----|------|--------|
| `id` | UUID (PK) | ID tin nhắn |
| `conversation_id` | UUID (FK → conversations) | Thuộc cuộc hội thoại nào |
| `role` | Enum (user/assistant) | Vai trò người gửi |
| `content` | Text | Nội dung tin nhắn |
| `sql_query` | Text (nullable) | Câu SQL đã sinh (nếu có) |
| `query_result` | JSON (nullable) | Kết quả truy vấn |
| `created_at` | DateTime | Thời điểm gửi |

### 3.8. Module Text-to-Data (🔲 Chưa triển khai — Core Feature)

**Mục đích:** Chuyển đổi câu hỏi ngôn ngữ tự nhiên thành truy vấn SQL.

**Chức năng dự kiến:**
- Nhận câu hỏi từ người dùng
- Lấy schema database liên quan (từ Qdrant)
- Gọi LLM (Ollama) sinh câu SQL
- Thực thi SQL trên database nguồn
- Trả kết quả đã format

**Data Model dự kiến — Schema (lưu metadata database nguồn):**

| Cột | Kiểu | Mô tả |
|-----|------|--------|
| `id` | UUID (PK) | ID |
| `database_name` | String | Tên database nguồn |
| `table_name` | String | Tên bảng |
| `column_name` | String | Tên cột |
| `data_type` | String | Kiểu dữ liệu |
| `description` | Text | Mô tả ý nghĩa |
| `embedding_id` | String | ID vector trong Qdrant |

### 3.9. Module Search (🔲 Chưa triển khai)

**Mục đích:** Tìm kiếm ngữ nghĩa (semantic search) qua vector database.

**Chức năng dự kiến:**
- Tìm kiếm schema phù hợp với câu hỏi
- Query Qdrant bằng embedding của câu hỏi
- Trả về top-K kết quả liên quan nhất

### 3.10. Module User (🔲 Chưa triển khai)

**Mục đích:** Quản lý thông tin người dùng (mở rộng từ auth).

**Chức năng dự kiến:**
- Cập nhật profile
- Đổi mật khẩu
- Quản lý cài đặt cá nhân

---

## 4. Database (PostgreSQL)

### 4.1. Tổng quan

Hệ thống sử dụng **PostgreSQL 16** với **2 database riêng biệt**:

| Database | User | Mục đích |
|----------|------|----------|
| `dagster` | `dagster` | Metadata cho Dagster orchestration |
| `talkwdata_db` | `talkwdata_user` | Dữ liệu ứng dụng (users, conversations, messages) |

### 4.2. Schema hiện tại (talkwdata_db)

```sql
CREATE TABLE users (
    id          VARCHAR PRIMARY KEY,         -- UUID v4
    email       VARCHAR UNIQUE NOT NULL,     -- Indexed
    username    VARCHAR UNIQUE NOT NULL,     -- Indexed
    hashed_password VARCHAR NOT NULL,
    full_name   VARCHAR,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ
);

CREATE UNIQUE INDEX ix_users_email ON users(email);
CREATE UNIQUE INDEX ix_users_username ON users(username);
```

### 4.3. Database Migration (Alembic)

Migration được quản lý bằng Alembic, cấu hình đọc `DATABASE_URL` từ `.server.env`.

```bash
# Tạo migration mới
cd server
alembic revision --autogenerate -m "mô tả thay đổi"

# Chạy migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 5. AI Service (Ollama)

### 5.1. Tổng quan

Ollama chạy các mô hình AI **hoàn toàn local**, không cần gọi API bên ngoài.

### 5.2. Mô hình sử dụng

| Model | Kích thước | Mục đích |
|-------|-----------|----------|
| **llama3.2** | ~2GB | LLM chính — hiểu ngôn ngữ, sinh SQL |
| **nomic-embed-text** | ~274MB | Tạo text embeddings cho semantic search |

### 5.3. Tích hợp

- **Python SDK:** `ollama==0.6.1` trong server
- **API Base URL:** `http://ollama:11434` (trong Docker network)
- Server gọi Ollama để:
  - Sinh câu SQL từ câu hỏi + schema context
  - Tạo embedding cho câu hỏi (tìm kiếm schema)

---

## 6. Vector Database (Qdrant)

### 6.1. Tổng quan

Qdrant lưu trữ **vector embeddings** để thực hiện **tìm kiếm ngữ nghĩa** (semantic search).

### 6.2. Vai trò trong hệ thống

```
Schema Database           Qdrant                    User Query
(bảng, cột, mô tả)       (vector store)            (câu hỏi)
       │                       │                        │
       ▼                       │                        ▼
  nomic-embed-text ──▶ Lưu embeddings          nomic-embed-text
                               │                        │
                               ▼                        ▼
                         So sánh cosine similarity ◀────┘
                               │
                               ▼
                    Top-K schema liên quan nhất
                               │
                               ▼
                    Đưa vào prompt cho LLM sinh SQL
```

### 6.3. Cấu hình

| Config | Giá trị |
|--------|---------|
| REST API | `:6333` |
| gRPC | `:6334` |
| Storage | `./volumes/qdrant_storage` |

---

## 7. Orchestration (Dagster)

### 7.1. Tổng quan

Dagster điều phối các **data pipeline** — tự động hóa việc xử lý và chuẩn bị dữ liệu.

### 7.2. Kiến trúc Dagster

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│ dagster_webserver│     │  dagster_daemon   │     │dagster_        │
│ :3000 (UI)      │     │  (scheduler,      │     │orchestration   │
│                 │     │   sensor, runs)   │     │:4000 (gRPC)    │
└────────┬────────┘     └────────┬─────────┘     └───────┬────────┘
         │                       │                        │
         └───────────┬───────────┘                        │
                     ▼                                    │
              ┌─────────────┐                             │
              │  PostgreSQL  │◀────────────────────────────┘
              │  (dagster)   │
              └─────────────┘
```

| Component | Chức năng |
|-----------|----------|
| **webserver** | Giao diện quản lý pipeline (port 3000) |
| **daemon** | Chạy schedules, sensors, queued runs |
| **orchestration** | Code server — chứa định nghĩa assets/jobs (gRPC port 4000) |

### 7.3. Pipeline dự kiến

- **Schema Crawler:** Quét metadata từ database nguồn
- **Embedding Generator:** Tạo vector embeddings từ schema
- **Qdrant Indexer:** Đẩy embeddings vào Qdrant
- **Scheduled Refresh:** Cập nhật định kỳ khi schema thay đổi

---

## 8. UI (Open WebUI)

### 8.1. Tổng quan

Giao diện người dùng sử dụng **Open WebUI** — một dự án mã nguồn mở dạng chat UI, tùy chỉnh để phù hợp với TalkingWithData.

> ⚠️ **Trạng thái:** Đang trong quá trình tùy chỉnh (commented out trong docker-compose)

### 8.2. Tùy chỉnh

| Config | Giá trị |
|--------|---------|
| Tên ứng dụng | "Talking with Data" |
| Ollama Backend | `http://ollama:11434` |
| Custom API | `http://server:8000` |
| Port | 8080 |

---

## 9. Cấu hình & Triển khai

### 9.1. Biến môi trường

#### Server (.server.env)

| Biến | Mô tả | Ví dụ |
|------|--------|-------|
| `DATABASE_URL` | Connection string PostgreSQL | `postgresql://user:pass@host:5432/db` |
| `SECRET_KEY` | Khóa ký JWT | (random string, bắt buộc đổi) |
| `JWT_ALGORITHM` | Thuật toán JWT | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Hạn access token | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Hạn refresh token | `7` |
| `OLLAMA_BASE_URL` | URL đến Ollama service | `http://ollama:11434` |
| `OLLAMA_DEFAULT_MODEL` | Model LLM mặc định | `llama3.2` |
| `QDRANT_HOST` | Host Qdrant | `qdrant` |
| `QDRANT_PORT` | Port Qdrant REST | `6333` |

#### Orchestration (.env)

| Biến | Mô tả | Ví dụ |
|------|--------|-------|
| `DAGSTER_PG_USERNAME` | User PostgreSQL cho Dagster | `dagster` |
| `DAGSTER_PG_PASSWORD` | Password | `dagster_password` |
| `DAGSTER_PG_DB` | Database name | `dagster` |
| `DAGSTER_OVERALL_CONCURRENCY_LIMIT` | Max concurrent runs | `10` |

### 9.2. Docker Compose — Tổng quan Services

```yaml
services:
  postgres:          # PostgreSQL 16         → port 5432
  dagster_webserver: # Dagster UI            → port 3000
  dagster_daemon:    # Dagster Daemon        → (internal)
  dagster_orchestration: # Dagster Code      → port 4000
  ollama:            # Ollama LLM            → port 11434
  qdrant:            # Qdrant Vector DB      → port 6333, 6334
  server:            # FastAPI Backend       → port 8000
  # open-webui:      # Chat UI (chưa bật)   → port 8080
```

### 9.3. Hướng dẫn Khởi chạy

```bash
# 1. Clone repository
git clone <repo-url>
cd talkingwithdata

# 2. Tạo file environment
cp server/.server.env.example server/.server.env
cp orchestration/orchestration.env.example orchestration/.env
# → Chỉnh sửa các giá trị trong file .env

# 3. Khởi chạy toàn bộ services
docker-compose up -d

# 4. Pull mô hình AI (chạy sau khi Ollama đã sẵn sàng)
# Windows:
powershell scripts/pull-ollama-models.ps1
# Linux/Mac:
bash scripts/pull-ollama-models.sh

# 5. Chạy database migration
cd server
alembic upgrade head

# 6. Kiểm tra các services
curl http://localhost:8000/health    # Server
curl http://localhost:11434          # Ollama
# Dagster UI: http://localhost:3000
# Qdrant Dashboard: http://localhost:6333/dashboard
```

### 9.4. Chạy Development (không Docker)

```bash
# 1. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\Activate.ps1  # Windows

# 2. Cài đặt dependencies
pip install -r server/requirements.txt

# 3. Chạy server (dev mode với hot reload)
cd server
python main.py
# Server chạy tại http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

---

## 10. Trạng thái Hiện tại & Roadmap

### 10.1. Tiến độ

| Module/Component | Trạng thái | Ghi chú |
|-----------------|-----------|---------|
| Docker Compose setup | ✅ Hoàn thành | 7 services đã cấu hình |
| PostgreSQL + Alembic | ✅ Hoàn thành | Migration cho bảng users |
| Core (database, security, dependencies) | ✅ Hoàn thành | JWT, password hashing, middleware |
| Auth module | ✅ Hoàn thành | Register, signin, signout, me, token |
| Conversation module | 🔲 Chưa bắt đầu | Cấu trúc thư mục đã tạo |
| Message module | 🔲 Chưa bắt đầu | Cấu trúc thư mục đã tạo |
| Text-to-Data module | 🔲 Chưa bắt đầu | **Core feature** — ưu tiên cao |
| Search module | 🔲 Chưa bắt đầu | Semantic search với Qdrant |
| User module | 🔲 Chưa bắt đầu | Quản lý profile |
| Dagster pipelines | 🔲 Chưa bắt đầu | Schema crawler, embedding indexer |
| Open WebUI customization | 🔲 Chưa bắt đầu | Đã có source, chưa tùy chỉnh |

### 10.2. Thứ tự triển khai đề xuất

```
Phase 1: Foundation (✅ Hoàn thành)
├── Docker infrastructure
├── Database setup + migration
└── Authentication system

Phase 2: Core Features (🔲 Tiếp theo)
├── Conversation module       ← Quản lý phiên chat
├── Message module             ← Lưu lịch sử tin nhắn
├── Text-to-Data module        ← ⭐ Tính năng chính
│   ├── Schema storage
│   ├── Ollama integration (text → SQL)
│   └── Query execution
└── Search module              ← Semantic search schema

Phase 3: Data Pipeline (🔲)
├── Dagster schema crawler
├── Embedding generator
└── Qdrant indexing pipeline

Phase 4: UI & Polish (🔲)
├── Open WebUI customization
├── User module (profile)
└── Error handling & logging
```

---

## 11. Tổng kết Công nghệ

| Tầng | Công nghệ | Phiên bản |
|------|-----------|-----------|
| **Frontend** | Open WebUI (SvelteKit + Python) | Custom fork |
| **Backend API** | FastAPI | 0.129.0 |
| **ORM** | SQLAlchemy | 2.0.46 |
| **Migration** | Alembic | 1.18.4 |
| **Validation** | Pydantic | 2.12.5 |
| **Auth** | JWT (python-jose) + bcrypt | — |
| **Database** | PostgreSQL | 16 |
| **Vector DB** | Qdrant | latest |
| **LLM Runtime** | Ollama | latest |
| **LLM Model** | llama3.2 | — |
| **Embedding Model** | nomic-embed-text | — |
| **Orchestration** | Dagster | 1.12.13 |
| **Container** | Docker + Docker Compose | — |
| **Language** | Python | 3.11 |
