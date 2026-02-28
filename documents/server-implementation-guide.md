# TalkingWithData Server — Hướng dẫn Triển khai Chi tiết

> **Tài liệu dành cho:** Developer  
> **Cập nhật:** 01/03/2026  
> **Mục tiêu:** Hướng dẫn từng bước triển khai toàn bộ server backend

---

## Mục lục

- [Phase 1: Foundation (✅ Đã hoàn thành)](#phase-1-foundation--đã-hoàn-thành)
- [Phase 2: Conversation Module](#phase-2-conversation-module)
- [Phase 3: Message Module](#phase-3-message-module)
- [Phase 4: Shared — Ollama & Qdrant Client](#phase-4-shared--ollama--qdrant-client)
- [Phase 5: Search Module](#phase-5-search-module)
- [Phase 6: Text-to-Data Module (⭐ Core Feature)](#phase-6-text-to-data-module--core-feature)
- [Phase 7: User Module](#phase-7-user-module)
- [Phase 8: Đăng ký Routers & Migration](#phase-8-đăng-ký-routers--migration)
- [Phase 9: Testing & Chạy thử](#phase-9-testing--chạy-thử)

---

## Quy ước chung

### Cấu trúc mỗi Module

```
module/<tên>/
├── __init__.py          # Tạo APIRouter, include sub-routers
├── endpoint/
│   ├── __init__.py      # (trống)
│   └── <tên_endpoint>.py
├── model/
│   ├── __init__.py      # (trống)
│   └── <tên_model>.py
├── schema/
│   ├── __init__.py      # (trống)
│   └── <tên_schema>.py
└── service/
    ├── __init__.py      # (trống)
    └── <tên_service>.py
```

### Thứ tự code trong mỗi module

```
1. Model (ORM)      — Định nghĩa bảng database
2. Schema (Pydantic) — Định nghĩa request/response
3. Service           — Business logic
4. Endpoint          — HTTP route handlers
5. __init__.py       — Đăng ký router
```

### Import pattern (đã có sẵn)

```python
# Trong endpoint → gọi service
from module.<tên>.service import <tên>_service

# Trong service → gọi model
from module.<tên>.model.<tên_model> import <ModelClass>

# Auth middleware
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
```

---

## Phase 1: Foundation (✅ Đã hoàn thành)

Các phần đã hoàn thành, **không cần code thêm**:

| File | Nội dung |
|------|---------|
| `main.py` | FastAPI app, CORS, include auth_router |
| `core/database.py` | Engine, SessionLocal, Base, get_db() |
| `core/sercurity.py` | JWT create/decode, password hash/verify |
| `core/dependencies.py` | get_current_user(), get_current_user_oauth2() |
| `module/auth/` | Register, signin, signout, me, token — đầy đủ |

**Cách module auth hoạt động (tham khảo khi làm module mới):**

```
module/auth/__init__.py
    → Tạo router = APIRouter(prefix="/auth", tags=["Authentication"])
    → include_router(register.router, signin.router, ...)

module/auth/endpoint/register.py
    → router = APIRouter()
    → @router.post("/register") → gọi auth_service.register(db, data)

module/auth/service/auth_service.py  
    → def register(db, data) → validate → tạo User → tạo JWT → return

module/auth/model/user.py
    → class User(Base) → __tablename__ = "users"

module/auth/schema/auth_schema.py
    → UserRegister, UserSignIn, UserResponse, TokenResponse
```

---

## Phase 2: Conversation Module

### Bước 2.1 — Model: `module/conversation/model/conversation.py`

```python
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database import Base
import uuid


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=False, default="New Conversation")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
```

**Giải thích:**
- `user_id` — FK đến bảng `users`, CASCADE khi xóa user → xóa hết conversation
- `messages` — relationship 1-N với Message (sẽ tạo ở Phase 3)
- `default="New Conversation"` — tiêu đề mặc định khi tạo mới

### Bước 2.2 — Schema: `module/conversation/schema/conversation_schema.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ============ Request Schemas ============

class ConversationCreate(BaseModel):
    """Tạo conversation mới"""
    title: Optional[str] = Field(default="New Conversation", max_length=200)


class ConversationUpdate(BaseModel):
    """Cập nhật conversation (đổi tên)"""
    title: str = Field(min_length=1, max_length=200)


# ============ Response Schemas ============

class ConversationResponse(BaseModel):
    """Response cho 1 conversation"""
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Response cho danh sách conversations"""
    conversations: List[ConversationResponse]
    total: int
```

### Bước 2.3 — Service: `module/conversation/service/conversation_service.py`

```python
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from module.conversation.model.conversation import Conversation
from module.conversation.schema.conversation_schema import (
    ConversationCreate, ConversationUpdate, ConversationResponse, ConversationListResponse
)


def create_conversation(db: Session, user_id: str, data: ConversationCreate) -> ConversationResponse:
    """Tạo conversation mới"""
    conversation = Conversation(
        user_id=user_id,
        title=data.title or "New Conversation"
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return ConversationResponse.model_validate(conversation)


def get_conversations(db: Session, user_id: str, skip: int = 0, limit: int = 50) -> ConversationListResponse:
    """Lấy danh sách conversations của user (mới nhất trước)"""
    query = db.query(Conversation).filter(Conversation.user_id == user_id)
    total = query.count()
    conversations = query.order_by(Conversation.updated_at.desc()).offset(skip).limit(limit).all()
    
    return ConversationListResponse(
        conversations=[ConversationResponse.model_validate(c) for c in conversations],
        total=total
    )


def get_conversation_by_id(db: Session, conversation_id: str, user_id: str) -> ConversationResponse:
    """Lấy chi tiết 1 conversation (phải thuộc về user)"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return ConversationResponse.model_validate(conversation)


def update_conversation(db: Session, conversation_id: str, user_id: str, data: ConversationUpdate) -> ConversationResponse:
    """Đổi tên conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    conversation.title = data.title
    db.commit()
    db.refresh(conversation)
    return ConversationResponse.model_validate(conversation)


def delete_conversation(db: Session, conversation_id: str, user_id: str) -> dict:
    """Xóa conversation (cascade xóa messages)"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted successfully"}
```

**Giải thích pattern:**
- Mọi query đều filter theo `user_id` → user chỉ thấy conversation của mình  
- `skip/limit` → phân trang  
- `order_by(updated_at.desc())` → conversation mới tương tác hiện lên đầu
- `delete` → cascade xóa tất cả messages trong conversation

### Bước 2.4 — Endpoint: `module/conversation/endpoint/conversation_endpoint.py`

```python
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from module.conversation.schema.conversation_schema import (
    ConversationCreate, ConversationUpdate, ConversationResponse, ConversationListResponse
)
from module.conversation.service import conversation_service

router = APIRouter()


@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    data: ConversationCreate,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Tạo cuộc hội thoại mới"""
    return conversation_service.create_conversation(db, current_user.id, data)


@router.get("/", response_model=ConversationListResponse)
def get_conversations(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Lấy danh sách cuộc hội thoại"""
    return conversation_service.get_conversations(db, current_user.id, skip, limit)


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Lấy chi tiết 1 cuộc hội thoại"""
    return conversation_service.get_conversation_by_id(db, conversation_id, current_user.id)


@router.put("/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: str,
    data: ConversationUpdate,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Đổi tên cuộc hội thoại"""
    return conversation_service.update_conversation(db, conversation_id, current_user.id, data)


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Xóa cuộc hội thoại"""
    return conversation_service.delete_conversation(db, conversation_id, current_user.id)
```

### Bước 2.5 — Đăng ký Router: `module/conversation/__init__.py`

```python
from fastapi import APIRouter
from module.conversation.endpoint import conversation_endpoint

router = APIRouter(prefix="/conversations", tags=["Conversations"])

router.include_router(conversation_endpoint.router)
```

### Bước 2.6 — Tạo các `__init__.py` trống

Tạo file trống cho:
- `module/conversation/endpoint/__init__.py`
- `module/conversation/model/__init__.py`
- `module/conversation/schema/__init__.py`
- `module/conversation/service/__init__.py`

---

## Phase 3: Message Module

### Bước 3.1 — Model: `module/message/model/message.py`

```python
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database import Base
import uuid
import enum


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(
        String, 
        ForeignKey("conversations.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    role = Column(SAEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    
    # Dành cho assistant response — lưu SQL đã sinh và kết quả
    sql_query = Column(Text, nullable=True)
    query_result = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
```

**Giải thích:**
- `role` — enum: `user` (câu hỏi) hoặc `assistant` (phản hồi AI)
- `sql_query` — câu SQL mà LLM sinh ra (chỉ có ở assistant message)
- `query_result` — kết quả JSON sau khi chạy SQL (chỉ có ở assistant message)
- `conversation_id` — FK cascade, xóa conversation → xóa hết messages

### Bước 3.2 — Schema: `module/message/schema/message_schema.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


class MessageRoleEnum(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


# ============ Request Schemas ============

class MessageCreate(BaseModel):
    """User gửi tin nhắn (câu hỏi)"""
    content: str = Field(min_length=1, max_length=5000)


# ============ Response Schemas ============

class MessageResponse(BaseModel):
    """Response cho 1 message"""
    id: str
    conversation_id: str
    role: MessageRoleEnum
    content: str
    sql_query: Optional[str] = None
    query_result: Optional[Any] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    """Danh sách messages trong conversation"""
    messages: List[MessageResponse]
    total: int


class ChatResponse(BaseModel):
    """Response khi user gửi tin nhắn — trả về cả user msg + assistant msg"""
    user_message: MessageResponse
    assistant_message: MessageResponse
```

### Bước 3.3 — Service: `module/message/service/message_service.py`

```python
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from module.message.model.message import Message, MessageRole
from module.message.schema.message_schema import (
    MessageCreate, MessageResponse, MessageListResponse, ChatResponse
)
from module.conversation.model.conversation import Conversation


def _verify_conversation_ownership(db: Session, conversation_id: str, user_id: str) -> Conversation:
    """Kiểm tra conversation thuộc về user"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conversation


def get_messages(db: Session, conversation_id: str, user_id: str, skip: int = 0, limit: int = 100) -> MessageListResponse:
    """Lấy lịch sử messages (cũ nhất trước — dạng chat)"""
    _verify_conversation_ownership(db, conversation_id, user_id)
    
    query = db.query(Message).filter(Message.conversation_id == conversation_id)
    total = query.count()
    messages = query.order_by(Message.created_at.asc()).offset(skip).limit(limit).all()
    
    return MessageListResponse(
        messages=[MessageResponse.model_validate(m) for m in messages],
        total=total
    )


def send_message(db: Session, conversation_id: str, user_id: str, data: MessageCreate) -> ChatResponse:
    """
    User gửi câu hỏi → Hệ thống:
    1. Lưu user message
    2. Gọi text_to_data service để xử lý câu hỏi
    3. Lưu assistant message (kèm SQL + kết quả)
    4. Cập nhật conversation.updated_at
    5. Trả về cả 2 messages
    """
    conversation = _verify_conversation_ownership(db, conversation_id, user_id)
    
    # 1. Lưu user message
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=data.content
    )
    db.add(user_message)
    
    # 2. Gọi text_to_data service xử lý
    #    (Phase 6 sẽ triển khai chi tiết, tạm thời trả placeholder)
    try:
        from module.text_to_data.service import text_to_data_service
        result = text_to_data_service.process_question(db, data.content)
        assistant_content = result.get("answer", "Tôi không thể trả lời câu hỏi này.")
        sql_query = result.get("sql_query")
        query_result = result.get("query_result")
    except Exception as e:
        assistant_content = f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"
        sql_query = None
        query_result = None
    
    # 3. Lưu assistant message
    assistant_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=assistant_content,
        sql_query=sql_query,
        query_result=query_result
    )
    db.add(assistant_message)
    
    # 4. Cập nhật conversation title (nếu vẫn là default)
    if conversation.title == "New Conversation":
        # Lấy 50 ký tự đầu của câu hỏi làm title
        conversation.title = data.content[:50] + ("..." if len(data.content) > 50 else "")
    
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)
    
    return ChatResponse(
        user_message=MessageResponse.model_validate(user_message),
        assistant_message=MessageResponse.model_validate(assistant_message)
    )


def delete_message(db: Session, message_id: str, conversation_id: str, user_id: str) -> dict:
    """Xóa 1 message (ít dùng nhưng có sẵn)"""
    _verify_conversation_ownership(db, conversation_id, user_id)
    
    message = db.query(Message).filter(
        Message.id == message_id,
        Message.conversation_id == conversation_id
    ).first()
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    db.delete(message)
    db.commit()
    return {"message": "Message deleted successfully"}
```

**Giải thích `send_message()` — đây là hàm quan trọng nhất:**
1. Lưu câu hỏi của user vào DB
2. Gọi `text_to_data_service.process_question()` — trung tâm AI  
3. Lưu câu trả lời AI + SQL + kết quả vào DB  
4. Tự động đặt tên conversation nếu chưa có  
5. Trả cả 2 messages cho frontend hiển thị

### Bước 3.4 — Endpoint: `module/message/endpoint/message_endpoint.py`

```python
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from module.message.schema.message_schema import (
    MessageCreate, MessageListResponse, ChatResponse
)
from module.message.service import message_service

router = APIRouter()


@router.get("/{conversation_id}/messages", response_model=MessageListResponse)
def get_messages(
    conversation_id: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Lấy lịch sử tin nhắn của 1 cuộc hội thoại"""
    return message_service.get_messages(db, conversation_id, current_user.id, skip, limit)


@router.post("/{conversation_id}/messages", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def send_message(
    conversation_id: str,
    data: MessageCreate,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Gửi câu hỏi → nhận phản hồi AI"""
    return message_service.send_message(db, conversation_id, current_user.id, data)


@router.delete("/{conversation_id}/messages/{message_id}")
def delete_message(
    conversation_id: str,
    message_id: str,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Xóa 1 tin nhắn"""
    return message_service.delete_message(db, message_id, conversation_id, current_user.id)
```

**Lưu ý:** Message endpoints được đặt nested dưới conversation:
- `GET /conversations/{id}/messages` — lấy lịch sử
- `POST /conversations/{id}/messages` — gửi tin nhắn
- `DELETE /conversations/{id}/messages/{msg_id}` — xóa

### Bước 3.5 — Đăng ký Router: `module/message/__init__.py`

```python
from fastapi import APIRouter
from module.message.endpoint import message_endpoint

router = APIRouter(prefix="/conversations", tags=["Messages"])

router.include_router(message_endpoint.router)
```

**Lưu ý:** prefix là `/conversations` vì message nằm nested trong conversation URL.

### Bước 3.6 — Tạo các `__init__.py` trống

- `module/message/endpoint/__init__.py`
- `module/message/model/__init__.py`
- `module/message/schema/__init__.py`
- `module/message/service/__init__.py`

---

## Phase 4: Shared — Ollama & Qdrant Client

Trước khi làm Text-to-Data và Search, cần tạo **client dùng chung** để gọi Ollama và Qdrant.

### Bước 4.1 — Cài thêm dependencies

Thêm vào `requirements.txt`:

```
qdrant-client==1.12.1
python-jose[cryptography]==3.4.0
passlib[bcrypt]==1.7.4
pydantic[email]==2.12.5
```

Chạy:
```bash
pip install qdrant-client python-jose[cryptography] passlib[bcrypt]
```

### Bước 4.2 — Ollama Client: `shared/ollama_client.py`

```python
import ollama
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / ".server.env"
load_dotenv(dotenv_path=env_path)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Tạo Ollama client
client = ollama.Client(host=OLLAMA_BASE_URL)


def chat(prompt: str, system_prompt: str = "", model: str = None) -> str:
    """
    Gọi LLM để chat — dùng cho text-to-SQL
    
    Args:
        prompt: Câu hỏi / prompt chính
        system_prompt: System instruction (vai trò, context)
        model: Model name (mặc định llama3.2)
    
    Returns:
        str: Câu trả lời từ LLM
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat(
        model=model or OLLAMA_DEFAULT_MODEL,
        messages=messages
    )
    
    return response["message"]["content"]


def generate_embedding(text: str, model: str = None) -> list[float]:
    """
    Tạo embedding vector từ text — dùng cho semantic search
    
    Args:
        text: Đoạn text cần tạo embedding
        model: Embedding model (mặc định nomic-embed-text)
    
    Returns:
        list[float]: Vector embedding (768 dimensions cho nomic-embed-text)
    """
    response = client.embed(
        model=model or OLLAMA_EMBED_MODEL,
        input=text
    )
    
    return response["embeddings"][0]


def generate_embeddings_batch(texts: list[str], model: str = None) -> list[list[float]]:
    """
    Tạo embedding cho nhiều texts cùng lúc (batch)
    
    Args:
        texts: Danh sách texts
        model: Embedding model
    
    Returns:
        list[list[float]]: Danh sách vectors
    """
    response = client.embed(
        model=model or OLLAMA_EMBED_MODEL,
        input=texts
    )
    
    return response["embeddings"]
```

**Giải thích:**
- `chat()` — gọi LLM sinh text (dùng cho text-to-SQL)
- `generate_embedding()` — tạo vector cho 1 đoạn text (dùng cho search)
- `generate_embeddings_batch()` — tạo vector cho nhiều texts (dùng cho indexing)
- Ollama Python SDK (`ollama==0.6.1`) đã có sẵn trong requirements

### Bước 4.3 — Qdrant Client: `shared/qdrant_client.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
)
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import uuid

env_path = Path(__file__).parent.parent / ".server.env"
load_dotenv(dotenv_path=env_path)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Tạo Qdrant client
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Tên collection chứa schema embeddings
SCHEMA_COLLECTION = "database_schemas"

# Dimension phụ thuộc embedding model (nomic-embed-text = 768)
EMBEDDING_DIMENSION = 768


def ensure_collection_exists(collection_name: str = SCHEMA_COLLECTION):
    """Tạo collection nếu chưa có"""
    collections = [c.name for c in qdrant.get_collections().collections]
    
    if collection_name not in collections:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        print(f"Created Qdrant collection: {collection_name}")


def upsert_schema_vectors(
    vectors: list[list[float]], 
    payloads: list[dict],
    collection_name: str = SCHEMA_COLLECTION
):
    """
    Lưu/cập nhật schema embeddings vào Qdrant
    
    Args:
        vectors: Danh sách embedding vectors
        payloads: Danh sách metadata (database_name, table_name, column_info, description)
        collection_name: Tên collection
    """
    ensure_collection_exists(collection_name)
    
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        )
        for vector, payload in zip(vectors, payloads)
    ]
    
    qdrant.upsert(collection_name=collection_name, points=points)


def search_similar_schemas(
    query_vector: list[float],
    limit: int = 5,
    database_name: Optional[str] = None,
    collection_name: str = SCHEMA_COLLECTION
) -> list[dict]:
    """
    Tìm kiếm schema phù hợp nhất với câu hỏi
    
    Args:
        query_vector: Embedding vector của câu hỏi
        limit: Số kết quả trả về (top-K)
        database_name: Lọc theo database cụ thể (optional)
        collection_name: Tên collection
    
    Returns:
        list[dict]: Danh sách schema matches, mỗi item gồm:
            - score: Cosine similarity (0-1)
            - payload: Metadata (table_name, columns, description, ...)
    """
    query_filter = None
    if database_name:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="database_name",
                    match=MatchValue(value=database_name)
                )
            ]
        )
    
    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    )
    
    return [
        {
            "score": point.score,
            "payload": point.payload
        }
        for point in results.points
    ]


def delete_schema_vectors(
    database_name: str,
    collection_name: str = SCHEMA_COLLECTION
):
    """Xóa tất cả vectors của 1 database (trước khi re-index)"""
    qdrant.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="database_name",
                    match=MatchValue(value=database_name)
                )
            ]
        )
    )
```

**Giải thích:**
- `ensure_collection_exists()` — auto tạo collection khi chạy lần đầu
- `upsert_schema_vectors()` — lưu embeddings + metadata vào Qdrant
- `search_similar_schemas()` — tìm top-K schema phù hợp nhất (cosine similarity)
- `delete_schema_vectors()` — xóa hết vectors của 1 database (khi cần re-index)

### Bước 4.4 — Export shared modules: `shared/__init__.py`

```python
# Shared utilities
```

---

## Phase 5: Search Module

### Bước 5.1 — Schema: `module/search/schema/search_schema.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Any


class SearchRequest(BaseModel):
    """Request tìm kiếm schema"""
    query: str = Field(min_length=1, max_length=2000, description="Câu hỏi tìm kiếm")
    database_name: Optional[str] = Field(default=None, description="Lọc theo database cụ thể")
    top_k: int = Field(default=5, ge=1, le=20, description="Số kết quả trả về")


class SchemaMatch(BaseModel):
    """1 kết quả khớp schema"""
    score: float
    database_name: str
    table_name: str
    columns: str
    description: Optional[str] = None


class SearchResponse(BaseModel):
    """Response tìm kiếm"""
    query: str
    results: List[SchemaMatch]
    total: int
```

### Bước 5.2 — Service: `module/search/service/search_service.py`

```python
from module.search.schema.search_schema import SearchRequest, SearchResponse, SchemaMatch
from shared.ollama_client import generate_embedding
from shared.qdrant_client import search_similar_schemas


def search_schemas(data: SearchRequest) -> SearchResponse:
    """
    Tìm kiếm schema phù hợp với câu hỏi
    
    Flow:
    1. Tạo embedding vector từ câu hỏi (Ollama nomic-embed-text)
    2. Query Qdrant tìm top-K vectors gần nhất
    3. Trả kết quả đã format
    """
    # 1. Tạo embedding cho câu hỏi
    query_vector = generate_embedding(data.query)
    
    # 2. Tìm kiếm trong Qdrant
    raw_results = search_similar_schemas(
        query_vector=query_vector,
        limit=data.top_k,
        database_name=data.database_name
    )
    
    # 3. Format kết quả
    results = []
    for item in raw_results:
        payload = item["payload"]
        results.append(SchemaMatch(
            score=item["score"],
            database_name=payload.get("database_name", ""),
            table_name=payload.get("table_name", ""),
            columns=payload.get("columns", ""),
            description=payload.get("description", "")
        ))
    
    return SearchResponse(
        query=data.query,
        results=results,
        total=len(results)
    )


def get_schema_context(query: str, database_name: str = None, top_k: int = 5) -> str:
    """
    Tìm schema và format thành context string cho LLM prompt
    
    Hàm này được gọi bởi text_to_data_service (không phải endpoint)
    
    Returns:
        str: Schema context dạng text, ví dụ:
            "Table: orders (id INT PK, customer_id INT FK, total DECIMAL, created_at TIMESTAMP)
             Table: customers (id INT PK, name VARCHAR, email VARCHAR)"
    """
    query_vector = generate_embedding(query)
    
    raw_results = search_similar_schemas(
        query_vector=query_vector,
        limit=top_k,
        database_name=database_name
    )
    
    if not raw_results:
        return "No relevant schema found."
    
    context_parts = []
    for item in raw_results:
        payload = item["payload"]
        table = payload.get("table_name", "unknown")
        columns = payload.get("columns", "")
        desc = payload.get("description", "")
        
        line = f"Table: {table}"
        if columns:
            line += f" ({columns})"
        if desc:
            line += f" -- {desc}"
        context_parts.append(line)
    
    return "\n".join(context_parts)
```

**Giải thích 2 hàm:**
- `search_schemas()` — public API: user tìm kiếm schema → trả SearchResponse
- `get_schema_context()` — internal: text_to_data gọi → trả string context cho LLM prompt

### Bước 5.3 — Endpoint: `module/search/endpoint/search_endpoint.py`

```python
from fastapi import APIRouter, Depends
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from module.search.schema.search_schema import SearchRequest, SearchResponse
from module.search.service import search_service

router = APIRouter()


@router.post("/", response_model=SearchResponse)
def search_schemas(
    data: SearchRequest,
    current_user: User = Depends(get_current_user_oauth2)
):
    """Tìm kiếm schema database phù hợp với câu hỏi"""
    return search_service.search_schemas(data)
```

### Bước 5.4 — Đăng ký Router: `module/search/__init__.py`

```python
from fastapi import APIRouter
from module.search.endpoint import search_endpoint

router = APIRouter(prefix="/search", tags=["Search"])

router.include_router(search_endpoint.router)
```

### Bước 5.5 — Tạo các `__init__.py` trống

- `module/search/endpoint/__init__.py`
- `module/search/model/__init__.py`
- `module/search/schema/__init__.py`
- `module/search/service/__init__.py`

---

## Phase 6: Text-to-Data Module (⭐ Core Feature)

Đây là **tính năng cốt lõi** — chuyển câu hỏi ngôn ngữ tự nhiên thành SQL và thực thi.

### Bước 6.1 — Model: `module/text_to_data/model/schema.py` (đã có file, cần thêm code)

```python
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func
from core.database import Base
import uuid


class DatabaseSchema(Base):
    """Metadata của database nguồn (bảng, cột, mô tả)"""
    __tablename__ = "database_schemas"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    database_name = Column(String, nullable=False, index=True)
    connection_string = Column(String, nullable=False)
    table_name = Column(String, nullable=False, index=True)
    columns = Column(Text, nullable=False)       # VD: "id INT PK, name VARCHAR(100), email VARCHAR(255)"
    description = Column(Text, nullable=True)     # Mô tả bảng/ngữ nghĩa
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

**Giải thích:**
- Mỗi row = **1 bảng** trong database nguồn
- `connection_string` — connection đến database nguồn (để thực thi SQL)
- `columns` — danh sách cột dạng text (sẽ embedding vào Qdrant)
- `description` — mô tả ngữ nghĩa bảng (giúp LLM hiểu context)

### Bước 6.2 — Model lưu Database Connection: `module/text_to_data/model/database_connection.py`

```python
from sqlalchemy import Column, String, Text, Boolean, DateTime
from sqlalchemy.sql import func
from core.database import Base
import uuid


class DatabaseConnection(Base):
    """Thông tin kết nối đến database nguồn"""
    __tablename__ = "database_connections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    connection_string = Column(Text, nullable=False)  # postgresql://user:pass@host:port/dbname
    db_type = Column(String, nullable=False, default="postgresql")  # postgresql, mysql, ...
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

### Bước 6.3 — Schema: `module/text_to_data/schema/text_to_data_schema.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# ============ Database Connection Schemas ============

class DatabaseConnectionCreate(BaseModel):
    """Đăng ký database nguồn"""
    name: str = Field(min_length=1, max_length=100)
    connection_string: str = Field(min_length=10)
    db_type: str = Field(default="postgresql")
    description: Optional[str] = None


class DatabaseConnectionResponse(BaseModel):
    id: str
    name: str
    db_type: str
    description: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ============ Schema Management ============

class SchemaImportRequest(BaseModel):
    """Request import schema từ database nguồn"""
    connection_id: str


class SchemaResponse(BaseModel):
    id: str
    database_name: str
    table_name: str
    columns: str
    description: Optional[str]

    class Config:
        from_attributes = True


class SchemaListResponse(BaseModel):
    schemas: List[SchemaResponse]
    total: int


# ============ Text-to-SQL Core ============

class QuestionRequest(BaseModel):
    """Request hỏi dữ liệu"""
    question: str = Field(min_length=1, max_length=5000, description="Câu hỏi bằng ngôn ngữ tự nhiên")
    database_name: Optional[str] = Field(default=None, description="Chỉ định database (nếu không → tự tìm)")


class QueryResultResponse(BaseModel):
    """Response kết quả truy vấn"""
    question: str
    sql_query: str
    answer: str
    query_result: Optional[Any] = None
    schema_context: Optional[str] = None
```

### Bước 6.4 — Service: `module/text_to_data/service/text_to_data_service.py`

```python
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text, inspect
from fastapi import HTTPException, status
from module.text_to_data.model.schema import DatabaseSchema
from module.text_to_data.model.database_connection import DatabaseConnection
from module.text_to_data.schema.text_to_data_schema import (
    DatabaseConnectionCreate, DatabaseConnectionResponse,
    SchemaImportRequest, SchemaResponse, SchemaListResponse,
    QuestionRequest, QueryResultResponse
)
from shared.ollama_client import chat, generate_embedding
from shared.qdrant_client import (
    search_similar_schemas, upsert_schema_vectors, 
    delete_schema_vectors, ensure_collection_exists
)
from shared.ollama_client import generate_embeddings_batch


# ================================================================
# 1. QUẢN LÝ DATABASE CONNECTION
# ================================================================

def add_connection(db: Session, data: DatabaseConnectionCreate) -> DatabaseConnectionResponse:
    """Đăng ký database nguồn mới"""
    # Kiểm tra trùng tên
    existing = db.query(DatabaseConnection).filter(DatabaseConnection.name == data.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database connection '{data.name}' already exists"
        )
    
    # Test kết nối
    try:
        test_engine = create_engine(data.connection_string)
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot connect to database: {str(e)}"
        )
    
    connection = DatabaseConnection(
        name=data.name,
        connection_string=data.connection_string,
        db_type=data.db_type,
        description=data.description
    )
    db.add(connection)
    db.commit()
    db.refresh(connection)
    return DatabaseConnectionResponse.model_validate(connection)


def get_connections(db: Session) -> list[DatabaseConnectionResponse]:
    """Lấy danh sách database đã đăng ký"""
    connections = db.query(DatabaseConnection).filter(DatabaseConnection.is_active == True).all()
    return [DatabaseConnectionResponse.model_validate(c) for c in connections]


# ================================================================
# 2. IMPORT SCHEMA TỪ DATABASE NGUỒN
# ================================================================

def import_schema(db: Session, data: SchemaImportRequest) -> SchemaListResponse:
    """
    Import schema từ database nguồn:
    1. Kết nối đến database nguồn
    2. Đọc metadata (tên bảng, cột, kiểu dữ liệu)
    3. Lưu vào bảng database_schemas
    4. Tạo embeddings và lưu vào Qdrant
    """
    # Lấy connection info
    connection = db.query(DatabaseConnection).filter(DatabaseConnection.id == data.connection_id).first()
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found"
        )
    
    # Kết nối đến database nguồn
    try:
        source_engine = create_engine(connection.connection_string)
        inspector = inspect(source_engine)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to source database: {str(e)}"
        )
    
    # Xóa schema cũ của database này (nếu có)
    db.query(DatabaseSchema).filter(DatabaseSchema.database_name == connection.name).delete()
    
    # Đọc schema từ database nguồn
    schemas = []
    table_names = inspector.get_table_names()
    
    for table_name in table_names:
        columns_info = inspector.get_columns(table_name)
        pk_columns = [pk["name"] for pk in (inspector.get_pk_constraint(table_name).get("constrained_columns", []) if isinstance(inspector.get_pk_constraint(table_name), dict) else [])]
        
        # Format columns string
        col_parts = []
        for col in columns_info:
            col_str = f"{col['name']} {str(col['type'])}"
            if col['name'] in pk_columns:
                col_str += " PK"
            if not col.get('nullable', True):
                col_str += " NOT NULL"
            col_parts.append(col_str)
        
        columns_str = ", ".join(col_parts)
        
        # Lưu vào database
        schema_record = DatabaseSchema(
            database_name=connection.name,
            connection_string=connection.connection_string,
            table_name=table_name,
            columns=columns_str,
            description=f"Table {table_name} in {connection.name}"
        )
        db.add(schema_record)
        schemas.append(schema_record)
    
    db.commit()
    
    # Tạo embeddings và lưu vào Qdrant
    _index_schemas_to_qdrant(schemas)
    
    # Refresh để lấy id
    for s in schemas:
        db.refresh(s)
    
    return SchemaListResponse(
        schemas=[SchemaResponse.model_validate(s) for s in schemas],
        total=len(schemas)
    )


def _index_schemas_to_qdrant(schemas: list[DatabaseSchema]):
    """Tạo embeddings cho schemas và lưu vào Qdrant"""
    if not schemas:
        return
    
    # Xóa vectors cũ
    database_name = schemas[0].database_name
    try:
        delete_schema_vectors(database_name)
    except Exception:
        pass  # Collection có thể chưa tồn tại
    
    ensure_collection_exists()
    
    # Tạo texts để embedding
    texts = []
    payloads = []
    for schema in schemas:
        # Text mô tả bảng — sẽ được embedding
        embed_text = f"Table: {schema.table_name}. Columns: {schema.columns}."
        if schema.description:
            embed_text += f" Description: {schema.description}"
        
        texts.append(embed_text)
        payloads.append({
            "database_name": schema.database_name,
            "table_name": schema.table_name,
            "columns": schema.columns,
            "description": schema.description or "",
            "connection_string": schema.connection_string
        })
    
    # Batch embedding
    vectors = generate_embeddings_batch(texts)
    
    # Lưu vào Qdrant
    upsert_schema_vectors(vectors, payloads)


def get_schemas(db: Session, database_name: str = None) -> SchemaListResponse:
    """Lấy danh sách schemas đã import"""
    query = db.query(DatabaseSchema)
    if database_name:
        query = query.filter(DatabaseSchema.database_name == database_name)
    
    schemas = query.all()
    return SchemaListResponse(
        schemas=[SchemaResponse.model_validate(s) for s in schemas],
        total=len(schemas)
    )


# ================================================================
# 3. TEXT-TO-SQL — CORE LOGIC  
# ================================================================

SYSTEM_PROMPT = """You are a SQL expert assistant. Your job is to convert natural language questions into SQL queries.

RULES:
1. Only generate SELECT queries (never INSERT, UPDATE, DELETE, DROP, etc.)
2. Use the provided database schema to write accurate queries
3. Return ONLY the SQL query, no explanations
4. If the question cannot be answered with the given schema, respond with: CANNOT_ANSWER
5. Use standard SQL syntax compatible with PostgreSQL
6. Always use table and column names exactly as provided in the schema
"""

USER_PROMPT_TEMPLATE = """Given the following database schema:

{schema_context}

Convert this question to a SQL query:
"{question}"

Return ONLY the SQL query, nothing else."""

ANSWER_PROMPT_TEMPLATE = """Given the following:

Question: "{question}"
SQL Query: {sql_query}
Query Result: {query_result}

Provide a clear, concise answer to the question in natural language (use Vietnamese if the question is in Vietnamese).
Format numbers nicely. If the result is a table, describe it briefly.
"""


def process_question(db: Session, question: str, database_name: str = None) -> dict:
    """
    Core function: Chuyển câu hỏi → SQL → Kết quả → Câu trả lời
    
    Flow:
    1. Tìm schema phù hợp trong Qdrant (semantic search)
    2. Build prompt với schema context
    3. Gọi LLM sinh câu SQL
    4. Thực thi SQL trên database nguồn
    5. Gọi LLM tạo câu trả lời tự nhiên
    6. Trả về kết quả hoàn chỉnh
    
    Returns:
        dict: {
            "question": str,
            "sql_query": str,
            "answer": str,
            "query_result": Any,
            "schema_context": str
        }
    """
    # 1. Tìm schema phù hợp
    query_vector = generate_embedding(question)
    schema_results = search_similar_schemas(
        query_vector=query_vector,
        limit=5,
        database_name=database_name
    )
    
    if not schema_results:
        return {
            "question": question,
            "sql_query": None,
            "answer": "Không tìm thấy schema database phù hợp. Vui lòng import schema trước.",
            "query_result": None,
            "schema_context": None
        }
    
    # 2. Build schema context
    context_parts = []
    connection_string = None
    for item in schema_results:
        payload = item["payload"]
        table = payload.get("table_name", "")
        columns = payload.get("columns", "")
        desc = payload.get("description", "")
        
        line = f"Table: {table} ({columns})"
        if desc:
            line += f" -- {desc}"
        context_parts.append(line)
        
        # Lấy connection string từ kết quả đầu tiên
        if not connection_string:
            connection_string = payload.get("connection_string")
    
    schema_context = "\n".join(context_parts)
    
    # 3. Gọi LLM sinh SQL
    user_prompt = USER_PROMPT_TEMPLATE.format(
        schema_context=schema_context,
        question=question
    )
    
    sql_query = chat(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
    sql_query = sql_query.strip().strip("```sql").strip("```").strip()
    
    # Kiểm tra nếu LLM không thể trả lời
    if "CANNOT_ANSWER" in sql_query:
        return {
            "question": question,
            "sql_query": None,
            "answer": "Không thể trả lời câu hỏi này với schema hiện tại.",
            "query_result": None,
            "schema_context": schema_context
        }
    
    # 4. Validate SQL (chỉ cho phép SELECT)
    sql_upper = sql_query.upper().strip()
    if not sql_upper.startswith("SELECT"):
        return {
            "question": question,
            "sql_query": sql_query,
            "answer": "LLM sinh câu truy vấn không hợp lệ (chỉ cho phép SELECT).",
            "query_result": None,
            "schema_context": schema_context
        }
    
    # 5. Thực thi SQL trên database nguồn
    query_result = None
    try:
        source_engine = create_engine(connection_string)
        with source_engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = list(result.keys())
            
            # Convert thành list of dicts
            query_result = [dict(zip(columns, row)) for row in rows]
            
            # Giới hạn kết quả (tránh quá lớn)
            if len(query_result) > 100:
                query_result = query_result[:100]
    except Exception as e:
        return {
            "question": question,
            "sql_query": sql_query,
            "answer": f"Lỗi khi thực thi truy vấn: {str(e)}",
            "query_result": None,
            "schema_context": schema_context
        }
    
    # 6. Gọi LLM tạo câu trả lời tự nhiên
    answer_prompt = ANSWER_PROMPT_TEMPLATE.format(
        question=question,
        sql_query=sql_query,
        query_result=str(query_result[:20])  # Giới hạn kết quả gửi cho LLM
    )
    
    answer = chat(prompt=answer_prompt)
    
    return {
        "question": question,
        "sql_query": sql_query,
        "answer": answer,
        "query_result": query_result,
        "schema_context": schema_context
    }
```

**Giải thích chi tiết luồng `process_question()`:**

```
User: "Có bao nhiêu đơn hàng trong tháng 1?"
 │
 ├─ (1) Embedding câu hỏi → vector 768 chiều
 ├─ (2) Qdrant tìm top-5 bảng liên quan:
 │       → orders (id, customer_id, total, created_at)
 │       → order_items (id, order_id, product_id, quantity)
 │       → ...
 ├─ (3) Build prompt:
 │       System: "You are a SQL expert..."
 │       User: "Schema: Table orders (...)\n Question: Có bao nhiêu..."
 ├─ (4) LLM sinh: "SELECT COUNT(*) FROM orders WHERE created_at >= '2026-01-01'"
 ├─ (5) Validate: ✅ bắt đầu bằng SELECT
 ├─ (6) Thực thi trên database nguồn → [{count: 150}]
 ├─ (7) LLM format: "Trong tháng 1 có 150 đơn hàng."
 └─ Return: {sql_query, answer, query_result, schema_context}
```

### Bước 6.5 — Endpoint: `module/text_to_data/endpoint/get_schema.py` (đã có file, thêm code)

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from module.text_to_data.schema.text_to_data_schema import (
    SchemaListResponse, SchemaImportRequest
)
from module.text_to_data.service import text_to_data_service

router = APIRouter()


@router.get("/schemas", response_model=SchemaListResponse)
def get_schemas(
    database_name: str = Query(default=None),
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Lấy danh sách schemas đã import"""
    return text_to_data_service.get_schemas(db, database_name)


@router.post("/schemas/import", response_model=SchemaListResponse)
def import_schema(
    data: SchemaImportRequest,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Import schema từ database nguồn"""
    return text_to_data_service.import_schema(db, data)
```

### Bước 6.6 — Endpoint quản lý connection: `module/text_to_data/endpoint/connection_endpoint.py`

```python
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from module.text_to_data.schema.text_to_data_schema import (
    DatabaseConnectionCreate, DatabaseConnectionResponse
)
from module.text_to_data.service import text_to_data_service

router = APIRouter()


@router.post("/connections", response_model=DatabaseConnectionResponse, status_code=status.HTTP_201_CREATED)
def add_connection(
    data: DatabaseConnectionCreate,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Đăng ký database nguồn mới"""
    return text_to_data_service.add_connection(db, data)


@router.get("/connections", response_model=list[DatabaseConnectionResponse])
def get_connections(
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Lấy danh sách database nguồn"""
    return text_to_data_service.get_connections(db)
```

### Bước 6.7 — Endpoint hỏi dữ liệu: `module/text_to_data/endpoint/query_endpoint.py`

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from module.text_to_data.schema.text_to_data_schema import (
    QuestionRequest, QueryResultResponse
)
from module.text_to_data.service import text_to_data_service

router = APIRouter()


@router.post("/query", response_model=QueryResultResponse)
def query_data(
    data: QuestionRequest,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """
    ⭐ Core API: Hỏi dữ liệu bằng ngôn ngữ tự nhiên
    
    Flow: Câu hỏi → Tìm schema → Sinh SQL → Thực thi → Trả kết quả
    """
    result = text_to_data_service.process_question(db, data.question, data.database_name)
    return QueryResultResponse(**result)
```

### Bước 6.8 — Đăng ký Router: `module/text_to_data/__init__.py`

```python
from fastapi import APIRouter
from module.text_to_data.endpoint import get_schema, connection_endpoint, query_endpoint

router = APIRouter(prefix="/text-to-data", tags=["Text to Data"])

router.include_router(connection_endpoint.router)
router.include_router(get_schema.router)
router.include_router(query_endpoint.router)
```

### Bước 6.9 — Tạo các `__init__.py` trống

- `module/text_to_data/endpoint/__init__.py`  ← đã có
- `module/text_to_data/model/__init__.py`
- `module/text_to_data/schema/__init__.py`
- `module/text_to_data/service/__init__.py`

---

## Phase 7: User Module

### Bước 7.1 — Schema: `module/user/schema/user_schema.py`

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UserProfile(BaseModel):
    """Thông tin profile user"""
    id: str
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class UserUpdateProfile(BaseModel):
    """Cập nhật profile"""
    full_name: Optional[str] = Field(default=None, max_length=100)
    username: Optional[str] = Field(default=None, min_length=3, max_length=50)


class ChangePasswordRequest(BaseModel):
    """Đổi mật khẩu"""
    current_password: str
    new_password: str = Field(min_length=6, max_length=100)
```

### Bước 7.2 — Service: `module/user/service/user_service.py`

```python
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from module.auth.model.user import User
from module.user.schema.user_schema import UserUpdateProfile, UserProfile, ChangePasswordRequest
from core.sercurity import verify_password, hash_password


def get_profile(user: User) -> UserProfile:
    """Lấy profile user"""
    return UserProfile.model_validate(user)


def update_profile(db: Session, user: User, data: UserUpdateProfile) -> UserProfile:
    """Cập nhật profile"""
    if data.username and data.username != user.username:
        # Kiểm tra username trùng
        existing = db.query(User).filter(User.username == data.username, User.id != user.id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        user.username = data.username
    
    if data.full_name is not None:
        user.full_name = data.full_name
    
    db.commit()
    db.refresh(user)
    return UserProfile.model_validate(user)


def change_password(db: Session, user: User, data: ChangePasswordRequest) -> dict:
    """Đổi mật khẩu"""
    if not verify_password(data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    user.hashed_password = hash_password(data.new_password)
    db.commit()
    return {"message": "Password changed successfully"}
```

### Bước 7.3 — Endpoint: `module/user/endpoint/user_endpoint.py`

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from module.user.schema.user_schema import UserProfile, UserUpdateProfile, ChangePasswordRequest
from module.user.service import user_service

router = APIRouter()


@router.get("/profile", response_model=UserProfile)
def get_profile(current_user: User = Depends(get_current_user_oauth2)):
    """Xem profile"""
    return user_service.get_profile(current_user)


@router.put("/profile", response_model=UserProfile)
def update_profile(
    data: UserUpdateProfile,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Cập nhật profile"""
    return user_service.update_profile(db, current_user, data)


@router.post("/change-password")
def change_password(
    data: ChangePasswordRequest,
    current_user: User = Depends(get_current_user_oauth2),
    db: Session = Depends(get_db)
):
    """Đổi mật khẩu"""
    return user_service.change_password(db, current_user, data)
```

### Bước 7.4 — Đăng ký Router: `module/user/__init__.py`

```python
from fastapi import APIRouter
from module.user.endpoint import user_endpoint

router = APIRouter(prefix="/user", tags=["User"])

router.include_router(user_endpoint.router)
```

### Bước 7.5 — Tạo các `__init__.py` trống

- `module/user/endpoint/__init__.py`
- `module/user/model/__init__.py`
- `module/user/schema/__init__.py`
- `module/user/service/__init__.py`

---

## Phase 8: Đăng ký Routers & Migration

### Bước 8.1 — Cập nhật `main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from module.auth import router as auth_router
from module.conversation import router as conversation_router
from module.message import router as message_router
from module.text_to_data import router as text_to_data_router
from module.search import router as search_router
from module.user import router as user_router

app = FastAPI(title="Server TalkWithData API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký tất cả routers
app.include_router(auth_router)
app.include_router(conversation_router)
app.include_router(message_router)
app.include_router(text_to_data_router)
app.include_router(search_router)
app.include_router(user_router)


@app.get("/")
def root():
    return {"message": "TalkWithData API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# Only use in the dev environment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### Bước 8.2 — Cập nhật `alembic/env.py` (import tất cả models)

Tìm đoạn import models trong `alembic/env.py` và thêm:

```python
# Import tất cả models để Alembic nhận diện
try:
    from core.database import Base
    from module.auth.model.user import User
    from module.conversation.model.conversation import Conversation
    from module.message.model.message import Message
    from module.text_to_data.model.schema import DatabaseSchema
    from module.text_to_data.model.database_connection import DatabaseConnection
    target_metadata = Base.metadata
except ImportError:
    target_metadata = None
```

### Bước 8.3 — Chạy Migration

```bash
cd server

# Tạo migration cho các bảng mới
alembic revision --autogenerate -m "add conversations messages schemas connections"

# Kiểm tra file migration sinh ra (trong alembic/versions/)
# Xem nội dung đã đúng chưa trước khi chạy

# Chạy migration
alembic upgrade head
```

### Bước 8.4 — Cập nhật `requirements.txt`

Thêm các packages còn thiếu:

```
qdrant-client==1.12.1
python-jose[cryptography]==3.4.0
passlib[bcrypt]==1.7.4
pydantic[email]==2.12.5
```

Cài đặt:
```bash
pip install qdrant-client python-jose[cryptography] passlib[bcrypt] pydantic[email]
pip freeze > requirements.txt
```

---

## Phase 9: Testing & Chạy thử

### Bước 9.1 — Khởi động services (Docker)

```bash
# Từ thư mục gốc project
docker-compose up -d postgres ollama qdrant

# Chờ postgres healthy
docker-compose logs -f postgres
```

### Bước 9.2 — Chạy server (dev mode)

```bash
cd server
python main.py
```

Mở Swagger UI: **http://localhost:8000/docs**

### Bước 9.3 — Test theo thứ tự

#### 1) Auth — Đăng ký & Đăng nhập

```bash
# Đăng ký
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","username":"testuser","password":"123456"}'

# Lưu lại access_token từ response
# VD: TOKEN="eyJhbGciOiJIUzI1NiIs..."
```

#### 2) Conversation — Tạo & Liệt kê

```bash
# Tạo conversation
curl -X POST http://localhost:8000/conversations/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Test conversation"}'

# Liệt kê
curl http://localhost:8000/conversations/ \
  -H "Authorization: Bearer $TOKEN"
```

#### 3) Text-to-Data — Đăng ký database & Import schema

```bash
# Đăng ký database nguồn
curl -X POST http://localhost:8000/text-to-data/connections \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sample_db",
    "connection_string": "postgresql://user:pass@localhost:5432/sample_db",
    "db_type": "postgresql"
  }'

# Import schema (dùng connection_id từ response trên)
curl -X POST http://localhost:8000/text-to-data/schemas/import \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"connection_id": "xxx-xxx-xxx"}'
```

#### 4) Text-to-Data — Hỏi dữ liệu (⭐ Core test)

```bash
# Hỏi dữ liệu bằng ngôn ngữ tự nhiên
curl -X POST http://localhost:8000/text-to-data/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "Có bao nhiêu bản ghi trong bảng users?"}'
```

#### 5) Message — Chat flow hoàn chỉnh

```bash
# Gửi tin nhắn trong conversation
curl -X POST http://localhost:8000/conversations/{conversation_id}/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Có bao nhiêu đơn hàng trong tháng 1?"}'

# Xem lịch sử
curl http://localhost:8000/conversations/{conversation_id}/messages \
  -H "Authorization: Bearer $TOKEN"
```

---

## Tổng kết — Danh sách Files cần tạo

### Files mới (tạo từ đầu)

| # | File | Phase |
|---|------|-------|
| 1 | `module/conversation/model/conversation.py` | 2 |
| 2 | `module/conversation/schema/conversation_schema.py` | 2 |
| 3 | `module/conversation/service/conversation_service.py` | 2 |
| 4 | `module/conversation/endpoint/conversation_endpoint.py` | 2 |
| 5 | `module/conversation/__init__.py` | 2 |
| 6 | `module/message/model/message.py` | 3 |
| 7 | `module/message/schema/message_schema.py` | 3 |
| 8 | `module/message/service/message_service.py` | 3 |
| 9 | `module/message/endpoint/message_endpoint.py` | 3 |
| 10 | `module/message/__init__.py` | 3 |
| 11 | `shared/ollama_client.py` | 4 |
| 12 | `shared/qdrant_client.py` | 4 |
| 13 | `shared/__init__.py` | 4 |
| 14 | `module/search/schema/search_schema.py` | 5 |
| 15 | `module/search/service/search_service.py` | 5 |
| 16 | `module/search/endpoint/search_endpoint.py` | 5 |
| 17 | `module/search/__init__.py` | 5 |
| 18 | `module/text_to_data/model/database_connection.py` | 6 |
| 19 | `module/text_to_data/schema/text_to_data_schema.py` | 6 |
| 20 | `module/text_to_data/service/text_to_data_service.py` | 6 |
| 21 | `module/text_to_data/endpoint/connection_endpoint.py` | 6 |
| 22 | `module/text_to_data/endpoint/query_endpoint.py` | 6 |
| 23 | `module/user/schema/user_schema.py` | 7 |
| 24 | `module/user/service/user_service.py` | 7 |
| 25 | `module/user/endpoint/user_endpoint.py` | 7 |
| 26 | `module/user/__init__.py` | 7 |

### Files cần sửa

| # | File | Thay đổi |
|---|------|---------|
| 1 | `module/text_to_data/__init__.py` | Thêm router đăng ký |
| 2 | `module/text_to_data/model/schema.py` | Thêm DatabaseSchema model |
| 3 | `module/text_to_data/endpoint/get_schema.py` | Thêm schema endpoints |
| 4 | `main.py` | Import & include tất cả routers |
| 5 | `alembic/env.py` | Import tất cả models |
| 6 | `requirements.txt` | Thêm qdrant-client, passlib, etc. |

### `__init__.py` trống cần tạo (nếu chưa có)

```
module/conversation/endpoint/__init__.py
module/conversation/model/__init__.py
module/conversation/schema/__init__.py
module/conversation/service/__init__.py
module/message/endpoint/__init__.py
module/message/model/__init__.py
module/message/schema/__init__.py
module/message/service/__init__.py
module/search/endpoint/__init__.py
module/search/model/__init__.py
module/search/schema/__init__.py
module/search/service/__init__.py
module/text_to_data/schema/__init__.py
module/text_to_data/service/__init__.py
module/user/endpoint/__init__.py
module/user/model/__init__.py
module/user/schema/__init__.py
module/user/service/__init__.py
```

---

## API Tổng kết

| Method | Path | Module | Mô tả |
|--------|------|--------|--------|
| `POST` | `/auth/register` | Auth | Đăng ký |
| `POST` | `/auth/signin` | Auth | Đăng nhập |
| `POST` | `/auth/signout` | Auth | Đăng xuất |
| `GET` | `/auth/me` | Auth | Thông tin tôi |
| `POST` | `/auth/token` | Auth | OAuth2 token |
| `POST` | `/conversations/` | Conversation | Tạo hội thoại |
| `GET` | `/conversations/` | Conversation | Danh sách |
| `GET` | `/conversations/{id}` | Conversation | Chi tiết |
| `PUT` | `/conversations/{id}` | Conversation | Đổi tên |
| `DELETE` | `/conversations/{id}` | Conversation | Xóa |
| `GET` | `/conversations/{id}/messages` | Message | Lịch sử tin nhắn |
| `POST` | `/conversations/{id}/messages` | Message | ⭐ Gửi câu hỏi |
| `DELETE` | `/conversations/{id}/messages/{mid}` | Message | Xóa tin nhắn |
| `POST` | `/text-to-data/connections` | Text-to-Data | Đăng ký DB nguồn |
| `GET` | `/text-to-data/connections` | Text-to-Data | Danh sách DB |
| `GET` | `/text-to-data/schemas` | Text-to-Data | Xem schemas |
| `POST` | `/text-to-data/schemas/import` | Text-to-Data | Import schema |
| `POST` | `/text-to-data/query` | Text-to-Data | ⭐ Hỏi dữ liệu |
| `POST` | `/search/` | Search | Tìm kiếm schema |
| `GET` | `/user/profile` | User | Xem profile |
| `PUT` | `/user/profile` | User | Sửa profile |
| `POST` | `/user/change-password` | User | Đổi mật khẩu |
