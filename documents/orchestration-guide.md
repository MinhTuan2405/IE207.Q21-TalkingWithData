# Dagster Orchestration — Quản lý Luồng Data & Tracking Flow

> **Dự án:** TalkingWithData  
> **Engine:** Dagster 1.12+  
> **Cập nhật:** 01/03/2026

---

## Mục lục

- [Phần 1: Dagster — Tổng quan](#phần-1-dagster--tổng-quan)
- [Phần 2: Các khái niệm cốt lõi](#phần-2-các-khái-niệm-cốt-lõi)
- [Phần 3: Kiến trúc Dagster trong TalkingWithData](#phần-3-kiến-trúc-dagster-trong-talkingwithdata)
- [Phần 4: Triển khai Data Pipelines](#phần-4-triển-khai-data-pipelines)
- [Phần 5: Sensors & Schedules — Automation](#phần-5-sensors--schedules--automation)
- [Phần 6: Tracking & Observability](#phần-6-tracking--observability)
- [Phần 7: Cấu hình & Deployment](#phần-7-cấu-hình--deployment)
- [Phần 8: Best Practices & Troubleshooting](#phần-8-best-practices--troubleshooting)

---

# Phần 1: Dagster — Tổng quan

## 1.1. Dagster là gì?

**Dagster** là một **data orchestrator** — phần mềm quản lý, lên lịch, và giám sát các pipeline xử lý dữ liệu. Khác với Airflow (task-centric), Dagster là **asset-centric**: focus vào **dữ liệu được tạo ra** thay vì **task cần chạy**.

```
Airflow:  Task A → Task B → Task C     (quan tâm "cần làm gì?")
Dagster:  Asset X → Asset Y → Asset Z  (quan tâm "cần tạo data gì?")
```

## 1.2. Tại sao dùng Dagster trong TalkingWithData?

TalkingWithData có nhiều luồng data cần được **quản lý tự động** và **tracking**:

| Luồng Data | Mô tả | Tần suất |
|-----------|--------|---------|
| **Schema Import** | Kết nối DB nguồn → extract DDL → lưu metadata | Khi user thêm DB mới |
| **Schema Embedding** | DDL text → Ollama embedding → lưu Qdrant | Sau khi import schema |
| **Training Pipeline** | DDL + Docs + Q&A → Vanna/LangChain training | Khi có dữ liệu mới |
| **Schema Sync** | Phát hiện thay đổi schema ở DB nguồn → cập nhật | Định kỳ (hàng giờ/ngày) |
| **Analytics** | Thống kê queries, accuracy, usage | Cuối ngày |

**Dagster giải quyết:**
- ✅ **Orchestrate** — Điều phối thứ tự các bước đúng dependency
- ✅ **Track** — Ghi log, visualize mọi pipeline run
- ✅ **Schedule** — Tự động chạy theo lịch hoặc sự kiện
- ✅ **Retry** — Tự chạy lại khi lỗi
- ✅ **Observe** — Giám sát qua UI (localhost:3000)

## 1.3. Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────┐
│                     Dagster System                        │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Webserver   │  │   Daemon     │  │  Code Server   │ │
│  │  (UI + API)  │  │  (scheduler, │  │  (gRPC, chứa   │ │
│  │  Port: 3000  │  │   sensors,   │  │   code pipeline│ │
│  │              │  │   run queue) │  │   Port: 4000)  │ │
│  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘ │
│         │                 │                   │          │
│         └─────────┬───────┘───────────────────┘          │
│                   │                                      │
│         ┌─────────▼─────────┐                            │
│         │   PostgreSQL      │                            │
│         │   (run storage,   │                            │
│         │    event log,     │                            │
│         │    schedule state)│                            │
│         └───────────────────┘                            │
└──────────────────────────────────────────────────────────┘

         │                     │                   │
         ▼                     ▼                   ▼
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │   Ollama    │    │   Qdrant     │    │   Source DB  │
  │   (LLM +   │    │   (Vector    │    │  (PostgreSQL │
  │  embedding) │    │    Store)    │    │   của user)  │
  └─────────────┘    └──────────────┘    └──────────────┘
```

---

# Phần 2: Các khái niệm cốt lõi

## 2.1. Assets — Đơn vị dữ liệu

**Asset** = một phần dữ liệu được pipeline tạo ra hoặc cập nhật. Thay vì nghĩ "cần chạy task gì", ta nghĩ "cần tạo/cập nhật data gì".

```python
from dagster import asset

@asset
def raw_schema_metadata():
    """Asset: metadata schema từ database nguồn"""
    # Extract DDL từ information_schema
    schemas = extract_schemas_from_source_db()
    return schemas

@asset(deps=[raw_schema_metadata])
def schema_embeddings(raw_schema_metadata):
    """Asset: embeddings của schema (phụ thuộc raw_schema_metadata)"""
    embeddings = generate_embeddings(raw_schema_metadata)
    store_in_qdrant(embeddings)
    return embeddings
```

**Dependency graph tự động:**
```
raw_schema_metadata → schema_embeddings → trained_vanna_model
                   ↘ schema_documentation ↗
```

## 2.2. Ops & Jobs — Đơn vị thực thi

**Op** = một function thực thi (compute unit), **Job** = tổ hợp các ops.

```python
from dagster import op, job, In, Out

@op(out=Out(list))
def extract_tables(context):
    """Op: Lấy danh sách bảng từ DB nguồn"""
    context.log.info("Extracting tables...")
    tables = get_tables_from_source()
    return tables

@op(ins={"tables": In(list)}, out=Out(dict))
def generate_ddl(context, tables):
    """Op: Sinh DDL cho từng bảng"""
    context.log.info(f"Generating DDL for {len(tables)} tables")
    ddl_map = {}
    for table in tables:
        ddl_map[table] = get_ddl(table)
    return ddl_map

@op(ins={"ddl_map": In(dict)})
def create_embeddings(context, ddl_map):
    """Op: Tạo embeddings và lưu Qdrant"""
    for table, ddl in ddl_map.items():
        embedding = ollama_embed(ddl)
        qdrant_upsert(table, embedding, ddl)
        context.log.info(f"Embedded: {table}")

@job
def schema_import_job():
    """Job: Full pipeline import schema"""
    tables = extract_tables()
    ddl = generate_ddl(tables)
    create_embeddings(ddl)
```

## 2.3. Resources — Kết nối services

**Resource** = kết nối đến service bên ngoài (DB, API, etc.), inject vào ops/assets.

```python
from dagster import resource, ConfigurableResource
from sqlalchemy import create_engine
from qdrant_client import QdrantClient
import ollama as ollama_sdk

class SourceDatabaseResource(ConfigurableResource):
    """Resource: Kết nối database nguồn của user"""
    connection_string: str
    
    def get_engine(self):
        return create_engine(self.connection_string)
    
    def execute_query(self, sql: str):
        engine = self.get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(sql))
            return result.fetchall()

class OllamaResource(ConfigurableResource):
    """Resource: Ollama LLM + Embedding"""
    base_url: str = "http://ollama:11434"
    model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    
    def get_client(self):
        return ollama_sdk.Client(host=self.base_url)
    
    def embed(self, text: str) -> list[float]:
        client = self.get_client()
        response = client.embed(model=self.embed_model, input=text)
        return response["embeddings"][0]
    
    def generate(self, prompt: str) -> str:
        client = self.get_client()
        response = client.generate(model=self.model, prompt=prompt)
        return response["response"]

class QdrantResource(ConfigurableResource):
    """Resource: Qdrant Vector Store"""
    host: str = "qdrant"
    port: int = 6333
    collection_name: str = "talkwdata_schemas"
    
    def get_client(self):
        return QdrantClient(host=self.host, port=self.port)
    
    def upsert(self, id: str, vector: list[float], payload: dict):
        client = self.get_client()
        from qdrant_client.models import PointStruct
        client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=id, vector=vector, payload=payload)]
        )
    
    def search(self, vector: list[float], limit: int = 5):
        client = self.get_client()
        return client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit
        )
```

## 2.4. Sensors — Trigger tự động

**Sensor** = function chạy liên tục, phát hiện sự kiện → trigger job/asset.

```python
from dagster import sensor, RunRequest, SensorEvaluationContext

@sensor(job=schema_import_job, minimum_interval_seconds=60)
def new_database_sensor(context: SensorEvaluationContext):
    """Sensor: Phát hiện khi có database mới được thêm → trigger import"""
    # Kiểm tra bảng database_connections xem có row mới không
    last_cursor = context.cursor or "0"
    new_connections = check_new_connections(since_id=int(last_cursor))
    
    for conn in new_connections:
        yield RunRequest(
            run_key=f"import_{conn.id}",
            run_config={
                "resources": {
                    "source_db": {
                        "config": {
                            "connection_string": conn.connection_string
                        }
                    }
                }
            }
        )
    
    if new_connections:
        context.update_cursor(str(new_connections[-1].id))
```

## 2.5. Schedules — Chạy định kỳ

```python
from dagster import schedule, ScheduleEvaluationContext

@schedule(cron_schedule="0 */6 * * *", job=schema_sync_job)
def schema_sync_schedule(context: ScheduleEvaluationContext):
    """Schedule: Đồng bộ schema mỗi 6 giờ"""
    return RunRequest()

@schedule(cron_schedule="0 0 * * *", job=analytics_job)
def daily_analytics_schedule(context: ScheduleEvaluationContext):
    """Schedule: Thống kê hàng ngày lúc 00:00"""
    return RunRequest()
```

## 2.6. Partitions — Xử lý data theo phân vùng

```python
from dagster import DailyPartitionsDefinition, asset

daily_partitions = DailyPartitionsDefinition(start_date="2026-01-01")

@asset(partitions_def=daily_partitions)
def daily_query_stats(context):
    """Asset: Thống kê query theo ngày"""
    date = context.partition_key  # "2026-03-01"
    stats = aggregate_query_stats(date)
    return stats
```

---

# Phần 3: Kiến trúc Dagster trong TalkingWithData

## 3.1. Cấu trúc Docker hiện tại

TalkingWithData chạy **3 container** Dagster:

| Container | Vai trò | Port | Entrypoint |
|-----------|---------|------|------------|
| `talkwdata_dagster_webserver` | UI + REST API | 3000 | `dagster-webserver` |
| `talkwdata_dagster_daemon` | Scheduler, Sensors, Run Queue | — | `dagster-daemon run` |
| `talkwdata_dagster_orchestration` | Code Server (chứa pipeline code) | 4000 (gRPC) | `dagster api grpc` |

```
dagster_webserver ←→ postgres ←→ dagster_daemon
        ↕ (gRPC)                      ↕ (gRPC)
    dagster_orchestration         dagster_orchestration
    (code server :4000)           (code server :4000)
```

**Tại sao tách 3 container?**
- **Webserver**: UI luôn sẵn sàng, không bị ảnh hưởng khi deploy code mới
- **Daemon**: Chạy background, quản lý schedules/sensors, queue runs
- **Code Server**: Chứa code pipeline, có thể restart/deploy lại mà không ảnh hưởng UI

## 3.2. Cấu trúc source code

```
orchestration/
├── dagster.yaml                 ← Cấu hình instance (storage, launcher, ...)
├── workspace.yaml               ← Khai báo code locations (gRPC server)
├── orchestration.env.example    ← Biến môi trường mẫu
├── pyproject.toml               ← Dependencies (dagster, dagster-postgres, ...)
└── src/
    └── orchestration/
        ├── __init__.py
        ├── definitions.py       ← Entry point: load tất cả defs
        └── defs/                ← Thư mục chứa definitions
            ├── __init__.py
            ├── assets/          ← (tạo mới) Data assets
            │   ├── __init__.py
            │   ├── schema_assets.py
            │   ├── embedding_assets.py
            │   ├── training_assets.py
            │   └── analytics_assets.py
            ├── jobs/            ← (tạo mới) Jobs
            │   ├── __init__.py
            │   ├── schema_import_job.py
            │   ├── schema_sync_job.py
            │   └── analytics_job.py
            ├── resources/       ← (tạo mới) External connections
            │   ├── __init__.py
            │   ├── source_db.py
            │   ├── ollama_resource.py
            │   └── qdrant_resource.py
            ├── sensors/         ← (tạo mới) Event-driven triggers
            │   ├── __init__.py
            │   ├── new_connection_sensor.py
            │   └── schema_change_sensor.py
            └── schedules/       ← (tạo mới) Time-based triggers
                ├── __init__.py
                ├── sync_schedule.py
                └── analytics_schedule.py
```

## 3.3. definitions.py — Entry point

File hiện tại dùng `load_from_defs_folder` — Dagster tự scan thư mục `defs/` và load tất cả assets, jobs, sensors, schedules, resources:

```python
# orchestration/src/orchestration/definitions.py (hiện tại)
from pathlib import Path
from dagster import definitions, load_from_defs_folder

@definitions
def defs():
    return load_from_defs_folder(path_within_project=Path(__file__).parent)
```

Cách này **tự động phát hiện** mọi definition trong `defs/` — không cần import thủ công.

> **Lưu ý:** `load_from_defs_folder` yêu cầu Dagster 1.10+ và mỗi file trong `defs/` phải export Dagster objects (assets, jobs, etc.) ở top-level.

---

# Phần 4: Triển khai Data Pipelines

## 4.1. Pipeline 1: Schema Import (core)

Luồng chính khi user kết nối database mới:

```
User thêm DB connection
       │
       ▼
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Extract Schema  │────▶│  Generate         │────▶│  Store in        │
│  (information_   │     │  Embeddings       │     │  Qdrant          │
│   schema)        │     │  (Ollama nomic)   │     │  (vector DB)     │
└──────────────────┘     └───────────────────┘     └──────────────────┘
       │                                                    │
       ▼                                                    ▼
┌──────────────────┐                              ┌──────────────────┐
│  Save metadata   │                              │  Train Vanna     │
│  (PostgreSQL)    │                              │  (DDL + docs)    │
└──────────────────┘                              └──────────────────┘
```

### Assets Implementation

```python
# defs/assets/schema_assets.py

from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from dagster import Config
from sqlalchemy import create_engine, text, inspect
from typing import Optional
import json


class SchemaImportConfig(Config):
    """Config cho schema import"""
    connection_string: str
    database_name: str
    schema_name: str = "public"


@asset(
    group_name="schema",
    description="Extract raw schema metadata từ database nguồn",
    kinds={"postgres"},
)
def raw_schema_metadata(
    context: AssetExecutionContext,
    config: SchemaImportConfig
) -> dict:
    """
    Bước 1: Kết nối database nguồn, extract DDL cho tất cả bảng
    
    Output: {
        "database_name": str,
        "tables": [
            {
                "table_name": str,
                "columns": [{"name": str, "type": str, "nullable": bool, "primary_key": bool}],
                "ddl": str,
                "foreign_keys": [...],
                "indexes": [...]
            }
        ]
    }
    """
    engine = create_engine(config.connection_string)
    inspector = inspect(engine)
    
    tables_data = []
    table_names = inspector.get_table_names(schema=config.schema_name)
    
    context.log.info(f"Found {len(table_names)} tables in {config.database_name}")
    
    for table_name in table_names:
        # Columns
        columns = []
        pk_columns = [col for col in inspector.get_pk_constraint(table_name, schema=config.schema_name).get("constrained_columns", [])]
        
        for col in inspector.get_columns(table_name, schema=config.schema_name):
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "primary_key": col["name"] in pk_columns,
                "default": str(col.get("default", "")) if col.get("default") else None
            })
        
        # Foreign keys
        fks = []
        for fk in inspector.get_foreign_keys(table_name, schema=config.schema_name):
            fks.append({
                "constrained_columns": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"]
            })
        
        # Indexes
        indexes = []
        for idx in inspector.get_indexes(table_name, schema=config.schema_name):
            indexes.append({
                "name": idx["name"],
                "columns": idx["column_names"],
                "unique": idx.get("unique", False)
            })
        
        # Generate DDL string
        cols_ddl = []
        for col in columns:
            col_def = f"  {col['name']} {col['type']}"
            if col['primary_key']:
                col_def += " PRIMARY KEY"
            if not col['nullable']:
                col_def += " NOT NULL"
            if col['default']:
                col_def += f" DEFAULT {col['default']}"
            cols_ddl.append(col_def)
        
        # FK constraints in DDL
        for fk in fks:
            fk_def = f"  FOREIGN KEY ({', '.join(fk['constrained_columns'])}) REFERENCES {fk['referred_table']}({', '.join(fk['referred_columns'])})"
            cols_ddl.append(fk_def)
        
        ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(cols_ddl) + "\n);"
        
        tables_data.append({
            "table_name": table_name,
            "columns": columns,
            "ddl": ddl,
            "foreign_keys": fks,
            "indexes": indexes
        })
        
        context.log.info(f"Extracted: {table_name} ({len(columns)} columns, {len(fks)} FKs)")
    
    engine.dispose()
    
    result = {
        "database_name": config.database_name,
        "schema": config.schema_name,
        "tables": tables_data,
        "table_count": len(tables_data)
    }
    
    # Metadata cho UI tracking
    return MaterializeResult(
        metadata={
            "database_name": MetadataValue.text(config.database_name),
            "table_count": MetadataValue.int(len(tables_data)),
            "tables": MetadataValue.json(
                {t["table_name"]: len(t["columns"]) for t in tables_data}
            )
        },
        value=result
    )
```

```python
# defs/assets/embedding_assets.py

from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
import hashlib


@asset(
    group_name="schema",
    deps=["raw_schema_metadata"],
    description="Sinh embedding vectors cho schema và lưu vào Qdrant",
    kinds={"qdrant", "ollama"},
)
def schema_embeddings(
    context: AssetExecutionContext,
    raw_schema_metadata: dict,
    ollama_resource: "OllamaResource",
    qdrant_resource: "QdrantResource"
) -> dict:
    """
    Bước 2: Tạo embeddings cho mỗi table DDL → lưu Qdrant
    
    Mỗi table DDL được embed thành 1 vector (768 dims).
    Khi user hỏi, system sẽ tìm table DDL gần nhất (semantic search).
    """
    database_name = raw_schema_metadata["database_name"]
    tables = raw_schema_metadata["tables"]
    
    embedded_count = 0
    
    for table in tables:
        # Tạo text để embed (DDL + column descriptions)
        embed_text = f"Database: {database_name}\n{table['ddl']}"
        
        # Tạo deterministic ID
        point_id = hashlib.md5(
            f"{database_name}:{table['table_name']}".encode()
        ).hexdigest()
        
        # Sinh embedding vector (768 dims từ nomic-embed-text)
        vector = ollama_resource.embed(embed_text)
        
        # Payload metadata
        payload = {
            "database_name": database_name,
            "table_name": table["table_name"],
            "ddl": table["ddl"],
            "column_count": len(table["columns"]),
            "columns": [c["name"] for c in table["columns"]],
            "foreign_keys": table["foreign_keys"]
        }
        
        # Upsert vào Qdrant
        qdrant_resource.upsert(
            id=point_id,
            vector=vector,
            payload=payload
        )
        
        embedded_count += 1
        context.log.info(f"Embedded: {table['table_name']} (ID: {point_id[:8]}...)")
    
    return MaterializeResult(
        metadata={
            "database_name": MetadataValue.text(database_name),
            "embedded_tables": MetadataValue.int(embedded_count),
            "vector_dimension": MetadataValue.int(768),
            "collection": MetadataValue.text(qdrant_resource.collection_name)
        },
        value={
            "database_name": database_name,
            "embedded_count": embedded_count
        }
    )
```

```python
# defs/assets/training_assets.py

from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue


@asset(
    group_name="training",
    deps=["raw_schema_metadata"],
    description="Train Vanna/LangChain với DDL và documentation",
    kinds={"ollama"},
)
def trained_model(
    context: AssetExecutionContext,
    raw_schema_metadata: dict,
) -> dict:
    """
    Bước 3: Training Vanna với DDL từ database nguồn
    
    Training data bao gồm:
    1. DDL (cấu trúc bảng)
    2. Documentation (mô tả nghiệp vụ, nếu có)
    3. Quan hệ giữa các bảng (FK)
    """
    database_name = raw_schema_metadata["database_name"]
    tables = raw_schema_metadata["tables"]
    
    # Import Vanna client (lazy import)
    # from shared.vanna_client import train_ddl, train_documentation
    
    trained_items = 0
    
    for table in tables:
        # Train DDL
        # train_ddl(table["ddl"])
        trained_items += 1
        context.log.info(f"Trained DDL: {table['table_name']}")
    
    # Train documentation (mô tả quan hệ)
    relationships = []
    for table in tables:
        for fk in table.get("foreign_keys", []):
            rel = f"Table {table['table_name']}.{', '.join(fk['constrained_columns'])} references {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
            relationships.append(rel)
    
    if relationships:
        doc = f"Database {database_name} relationships:\n" + "\n".join(relationships)
        # train_documentation(doc)
        trained_items += 1
        context.log.info(f"Trained relationships documentation ({len(relationships)} FKs)")
    
    return MaterializeResult(
        metadata={
            "database_name": MetadataValue.text(database_name),
            "trained_tables": MetadataValue.int(len(tables)),
            "trained_relationships": MetadataValue.int(len(relationships)),
            "total_training_items": MetadataValue.int(trained_items)
        },
        value={
            "database_name": database_name,
            "trained_items": trained_items
        }
    )
```

## 4.2. Pipeline 2: Schema Sync (cập nhật thay đổi)

```
Cron: mỗi 6 giờ
       │
       ▼
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Extract Current │────▶│  Compare with     │────▶│  Update changed  │
│  Schema          │     │  Stored Schema    │     │  embeddings      │
└──────────────────┘     └───────────────────┘     └──────────────────┘
                                │
                                ▼
                         ┌──────────────────┐
                         │  Alert (nếu có   │
                         │  breaking change)│
                         └──────────────────┘
```

```python
# defs/assets/schema_sync_assets.py

from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from dagster import Config


class SchemaSyncConfig(Config):
    connection_string: str
    database_name: str


@asset(
    group_name="sync",
    description="Phát hiện thay đổi schema (thêm/xóa/sửa bảng/cột)",
    kinds={"postgres"},
)
def schema_diff(
    context: AssetExecutionContext,
    config: SchemaSyncConfig,
    qdrant_resource: "QdrantResource"
) -> dict:
    """
    So sánh schema hiện tại của DB nguồn với schema đã lưu trong Qdrant.
    
    Trả về danh sách thay đổi:
    - added_tables: bảng mới
    - removed_tables: bảng đã xóa
    - modified_tables: bảng có cột thay đổi
    """
    from sqlalchemy import create_engine, inspect
    
    # 1. Lấy schema hiện tại từ DB nguồn
    engine = create_engine(config.connection_string)
    inspector = inspect(engine)
    current_tables = set(inspector.get_table_names(schema="public"))
    
    current_schema = {}
    for table in current_tables:
        cols = inspector.get_columns(table, schema="public")
        current_schema[table] = {col["name"]: str(col["type"]) for col in cols}
    
    engine.dispose()
    
    # 2. Lấy schema đã lưu từ Qdrant metadata
    client = qdrant_resource.get_client()
    stored_points = client.scroll(
        collection_name=qdrant_resource.collection_name,
        scroll_filter={
            "must": [
                {"key": "database_name", "match": {"value": config.database_name}}
            ]
        },
        limit=1000
    )[0]
    
    stored_tables = {p.payload["table_name"] for p in stored_points}
    stored_schema = {}
    for p in stored_points:
        stored_schema[p.payload["table_name"]] = {
            col: "" for col in p.payload.get("columns", [])
        }
    
    # 3. So sánh
    added = current_tables - stored_tables
    removed = stored_tables - current_tables
    common = current_tables & stored_tables
    
    modified = {}
    for table in common:
        current_cols = set(current_schema.get(table, {}).keys())
        stored_cols = set(stored_schema.get(table, {}).keys())
        
        new_cols = current_cols - stored_cols
        dropped_cols = stored_cols - current_cols
        
        if new_cols or dropped_cols:
            modified[table] = {
                "added_columns": list(new_cols),
                "removed_columns": list(dropped_cols)
            }
    
    diff = {
        "database_name": config.database_name,
        "added_tables": list(added),
        "removed_tables": list(removed),
        "modified_tables": modified,
        "has_changes": bool(added or removed or modified)
    }
    
    context.log.info(
        f"Schema diff: +{len(added)} tables, -{len(removed)} tables, "
        f"~{len(modified)} modified"
    )
    
    return MaterializeResult(
        metadata={
            "has_changes": MetadataValue.bool(diff["has_changes"]),
            "added_tables": MetadataValue.int(len(added)),
            "removed_tables": MetadataValue.int(len(removed)),
            "modified_tables": MetadataValue.int(len(modified)),
            "details": MetadataValue.json(diff)
        },
        value=diff
    )


@asset(
    group_name="sync",
    deps=["schema_diff"],
    description="Cập nhật embeddings cho các bảng đã thay đổi",
)
def updated_embeddings(
    context: AssetExecutionContext,
    schema_diff: dict,
    ollama_resource: "OllamaResource",
    qdrant_resource: "QdrantResource"
) -> dict:
    """
    Chỉ cập nhật embeddings cho bảng thay đổi (incremental update).
    Không re-embed toàn bộ → tiết kiệm thời gian.
    """
    if not schema_diff["has_changes"]:
        context.log.info("No schema changes detected. Skipping.")
        return {"updated": 0, "removed": 0}
    
    updated = 0
    removed = 0
    
    # Xóa embeddings của bảng đã bị xóa
    for table in schema_diff["removed_tables"]:
        # qdrant_resource.delete_by_table(schema_diff["database_name"], table)
        removed += 1
        context.log.info(f"Removed embedding: {table}")
    
    # Re-embed bảng mới và bảng đã sửa
    tables_to_embed = (
        schema_diff["added_tables"] + 
        list(schema_diff["modified_tables"].keys())
    )
    
    for table_name in tables_to_embed:
        # Re-extract DDL và embed (tương tự schema_embeddings asset)
        context.log.info(f"Re-embedding: {table_name}")
        updated += 1
    
    return MaterializeResult(
        metadata={
            "tables_updated": MetadataValue.int(updated),
            "tables_removed": MetadataValue.int(removed)
        },
        value={"updated": updated, "removed": removed}
    )
```

## 4.3. Pipeline 3: Analytics & Monitoring

```python
# defs/assets/analytics_assets.py

from dagster import asset, AssetExecutionContext, DailyPartitionsDefinition, MetadataValue, MaterializeResult

daily_partitions = DailyPartitionsDefinition(start_date="2026-01-01")


@asset(
    group_name="analytics",
    partitions_def=daily_partitions,
    description="Thống kê queries hàng ngày",
    kinds={"postgres"},
)
def daily_query_stats(context: AssetExecutionContext) -> dict:
    """
    Thống kê cho ngày partition_key:
    - Tổng số queries
    - Số queries thành công (SQL valid + có kết quả)
    - Số queries thất bại
    - Query phổ biến nhất
    - Thời gian xử lý trung bình
    """
    date = context.partition_key  # "2026-03-01"
    
    # Query từ bảng messages trong PostgreSQL
    # stats = db.query(...).filter(Message.created_at == date)...
    
    stats = {
        "date": date,
        "total_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "avg_response_time_ms": 0,
        "unique_users": 0,
        "top_tables_queried": []
    }
    
    context.log.info(f"Analytics for {date}: {stats['total_queries']} queries")
    
    return MaterializeResult(
        metadata={
            "date": MetadataValue.text(date),
            "total_queries": MetadataValue.int(stats["total_queries"]),
            "success_rate": MetadataValue.float(
                stats["successful_queries"] / max(stats["total_queries"], 1) * 100
            ),
            "unique_users": MetadataValue.int(stats["unique_users"])
        },
        value=stats
    )


@asset(
    group_name="analytics",
    deps=["daily_query_stats"],
    description="Phân tích accuracy của SQL generation",
)
def sql_accuracy_report(
    context: AssetExecutionContext,
    daily_query_stats: dict
) -> dict:
    """
    Báo cáo chất lượng SQL generation:
    - % SQL valid (parse được)
    - % SQL thực thi thành công
    - Lỗi phổ biến nhất
    - Bảng hay bị truy vấn sai
    """
    report = {
        "total_queries": daily_query_stats.get("total_queries", 0),
        "sql_parse_success_rate": 0.0,
        "sql_execution_success_rate": 0.0,
        "common_errors": [],
        "problematic_tables": []
    }
    
    return MaterializeResult(
        metadata={
            "parse_success_rate": MetadataValue.float(report["sql_parse_success_rate"]),
            "execution_success_rate": MetadataValue.float(report["sql_execution_success_rate"])
        },
        value=report
    )
```

---

# Phần 5: Sensors & Schedules — Automation

## 5.1. Sensor: Phát hiện database mới

```python
# defs/sensors/new_connection_sensor.py

from dagster import sensor, RunRequest, SensorEvaluationContext, SensorResult, SkipReason
import requests


@sensor(
    description="Phát hiện khi user thêm database connection mới → trigger schema import",
    minimum_interval_seconds=30
)
def new_connection_sensor(context: SensorEvaluationContext) -> SensorResult:
    """
    Polling API server mỗi 30 giây:
    - GET /text-to-data/connections?since={cursor}
    - Nếu có connection mới → RunRequest cho schema import pipeline
    """
    last_checked_id = int(context.cursor) if context.cursor else 0
    
    try:
        # Gọi FastAPI server
        response = requests.get(
            "http://server:8000/text-to-data/connections",
            params={"since_id": last_checked_id},
            timeout=10
        )
        
        if response.status_code != 200:
            return SkipReason(f"API returned {response.status_code}")
        
        new_connections = response.json().get("connections", [])
        
        if not new_connections:
            return SkipReason("No new connections")
        
        run_requests = []
        max_id = last_checked_id
        
        for conn in new_connections:
            run_requests.append(
                RunRequest(
                    run_key=f"schema_import_{conn['id']}_{conn['database_name']}",
                    run_config={
                        "ops": {
                            "raw_schema_metadata": {
                                "config": {
                                    "connection_string": conn["connection_string"],
                                    "database_name": conn["database_name"]
                                }
                            }
                        }
                    },
                    tags={
                        "database_name": conn["database_name"],
                        "trigger": "new_connection_sensor"
                    }
                )
            )
            max_id = max(max_id, conn["id"])
        
        context.update_cursor(str(max_id))
        
        return SensorResult(
            run_requests=run_requests,
            cursor=str(max_id)
        )
    
    except requests.RequestException as e:
        return SkipReason(f"Cannot reach server: {str(e)}")
```

## 5.2. Sensor: Phát hiện schema thay đổi

```python
# defs/sensors/schema_change_sensor.py

from dagster import sensor, RunRequest, SensorEvaluationContext, SkipReason
from sqlalchemy import create_engine, inspect
import json


@sensor(
    description="Kiểm tra schema DB nguồn có thay đổi không (mỗi 5 phút)",
    minimum_interval_seconds=300  # 5 phút
)
def schema_change_sensor(context: SensorEvaluationContext):
    """
    So sánh fingerprint của schema hiện tại với lần check trước.
    Nếu khác → trigger schema sync pipeline.
    
    Fingerprint = hash(sorted table names + column names)
    """
    import hashlib
    
    # Lấy danh sách connections cần monitor
    # connections = get_active_connections()
    connections = []  # Placeholder
    
    cursor_data = json.loads(context.cursor) if context.cursor else {}
    
    for conn in connections:
        try:
            engine = create_engine(conn["connection_string"])
            inspector = inspect(engine)
            tables = sorted(inspector.get_table_names(schema="public"))
            
            # Tạo fingerprint
            schema_parts = []
            for table in tables:
                cols = sorted([c["name"] for c in inspector.get_columns(table)])
                schema_parts.append(f"{table}:{','.join(cols)}")
            
            fingerprint = hashlib.sha256("|".join(schema_parts).encode()).hexdigest()
            engine.dispose()
            
            # So sánh với lần trước
            db_key = conn["database_name"]
            prev_fingerprint = cursor_data.get(db_key)
            
            if prev_fingerprint and prev_fingerprint != fingerprint:
                context.log.info(f"Schema change detected in {db_key}")
                yield RunRequest(
                    run_key=f"sync_{db_key}_{fingerprint[:8]}",
                    run_config={
                        "ops": {
                            "schema_diff": {
                                "config": {
                                    "connection_string": conn["connection_string"],
                                    "database_name": db_key
                                }
                            }
                        }
                    },
                    tags={"trigger": "schema_change_sensor", "database": db_key}
                )
            
            cursor_data[db_key] = fingerprint
        
        except Exception as e:
            context.log.warning(f"Error checking {conn.get('database_name', '?')}: {e}")
    
    context.update_cursor(json.dumps(cursor_data))
```

## 5.3. Schedule: Định kỳ sync

```python
# defs/schedules/sync_schedule.py

from dagster import schedule, ScheduleEvaluationContext, RunRequest


@schedule(
    cron_schedule="0 */6 * * *",  # Mỗi 6 giờ
    description="Đồng bộ schema mỗi 6 giờ cho tất cả database đã kết nối"
)
def schema_sync_schedule(context: ScheduleEvaluationContext):
    """
    Chạy lúc: 00:00, 06:00, 12:00, 18:00
    Trigger schema sync cho tất cả active connections.
    """
    # connections = get_all_active_connections()
    connections = []  # Placeholder
    
    for conn in connections:
        yield RunRequest(
            run_key=f"scheduled_sync_{conn['database_name']}_{context.scheduled_execution_time.isoformat()}",
            run_config={
            "ops": {
                    "schema_diff": {
                        "config": {
                            "connection_string": conn["connection_string"],
                            "database_name": conn["database_name"]
                        }
                    }
                }
            },
            tags={
                "trigger": "schema_sync_schedule",
                "scheduled_time": context.scheduled_execution_time.isoformat()
            }
        )
```

```python
# defs/schedules/analytics_schedule.py

from dagster import schedule, ScheduleEvaluationContext, RunRequest


@schedule(
    cron_schedule="30 0 * * *",  # Mỗi ngày lúc 00:30
    description="Tạo báo cáo analytics hàng ngày"
)
def daily_analytics_schedule(context: ScheduleEvaluationContext):
    """
    Chạy lúc 00:30 mỗi ngày.
    Tạo thống kê query ngày hôm trước.
    """
    yesterday = (
        context.scheduled_execution_time.date() 
        - __import__('datetime').timedelta(days=1)
    )
    
    return RunRequest(
        run_key=f"analytics_{yesterday.isoformat()}",
        tags={
            "trigger": "daily_analytics_schedule",
            "date": yesterday.isoformat()
        }
    )
```

---

# Phần 6: Tracking & Observability

## 6.1. Dagster UI — Tổng quan

Truy cập: **http://localhost:3000** (dagster_webserver)

### Các tab chính:

| Tab | Chức năng |
|-----|----------|
| **Asset Catalog** | Xem tất cả assets, dependency graph, materialization history |
| **Runs** | Lịch sử tất cả pipeline runs, status, duration, logs |
| **Jobs** | Danh sách jobs, trigger thủ công |
| **Schedules** | Quản lý schedules (bật/tắt, xem history) |
| **Sensors** | Quản lý sensors (bật/tắt, xem ticks) |
| **Resources** | Xem configured resources |

### Asset Graph Visualization

```
Dagster UI tự động vẽ dependency graph:

┌───────────────────┐
│ raw_schema_metadata│ ← Group: schema
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────┐
│schema_ │  │trained_  │ ← Group: training
│embeddings│  │model     │
└────────┘  └──────────┘

┌────────────┐
│schema_diff │ ← Group: sync
└─────┬──────┘
      │
      ▼
┌──────────────┐
│updated_      │
│embeddings    │
└──────────────┘

┌─────────────────┐
│daily_query_stats│ ← Group: analytics (partitioned)
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│sql_accuracy_report │
└────────────────────┘
```

## 6.2. Run Tracking

Mỗi pipeline run được Dagster ghi lại đầy đủ:

```
Run ID:     a1b2c3d4-5678-...
Status:     ✅ SUCCESS / ❌ FAILURE / 🔄 IN_PROGRESS
Start:      2026-03-01 10:00:00
End:        2026-03-01 10:02:35
Duration:   2m 35s
Tags:       trigger=new_connection_sensor, database=ecommerce_db
```

### Structured Logs

```python
# Trong ops/assets, dùng context.log
context.log.info(f"Processing table: {table_name}")
context.log.warning(f"Slow query detected: {duration}ms")
context.log.error(f"Failed to connect: {error}")

# Logs hiển thị trong UI với timestamps, severity levels
```

### Metadata Tracking

```python
# Metadata được hiển thị trong Asset Catalog
return MaterializeResult(
    metadata={
        # Hiển thị dạng text
        "database": MetadataValue.text("ecommerce_db"),
        
        # Hiển thị dạng số
        "table_count": MetadataValue.int(15),
        
        # Hiển thị dạng JSON (expandable)
        "details": MetadataValue.json({"tables": ["orders", "customers"]}),
        
        # Hiển thị dạng markdown
        "summary": MetadataValue.md("## Report\n- 15 tables\n- 120 columns"),
        
        # Hiển thị progress
        "success_rate": MetadataValue.float(95.5),
        
        # Link
        "dashboard": MetadataValue.url("http://localhost:3000/assets"),
    }
)
```

## 6.3. Alerting (khi pipeline lỗi)

```python
# defs/resources/alerting.py

from dagster import failure_hook, success_hook, HookContext
import requests


@failure_hook
def notify_on_failure(context: HookContext):
    """Hook: Gửi thông báo khi pipeline thất bại"""
    message = (
        f"🔴 Pipeline Failed!\n"
        f"Op: {context.op.name}\n"
        f"Run ID: {context.run_id}\n"
        f"Error: {context.op_exception}"
    )
    
    context.log.error(message)
    
    # Gửi webhook (Slack, Discord, etc.)
    # requests.post(WEBHOOK_URL, json={"text": message})


@success_hook
def log_on_success(context: HookContext):
    """Hook: Log khi pipeline thành công"""
    context.log.info(f"✅ Op {context.op.name} completed successfully")


# Sử dụng hook trong job
from dagster import job

@job(hooks={notify_on_failure, log_on_success})
def schema_import_job():
    ...
```

## 6.4. Data Lineage (theo dõi nguồn gốc data)

Dagster tự động tracking **data lineage** — biết mỗi asset được tạo từ đâu:

```
Lineage cho "schema_embeddings":
────────────────────────────────
Upstream:
  └── raw_schema_metadata (last materialized: 2026-03-01 10:00)
      └── Source: postgres://user@host/ecommerce_db

Downstream:
  └── trained_model (stale - cần re-materialize)
```

**Stale detection**: Dagster tự biết khi upstream asset thay đổi → downstream trở nên "stale" và cần cập nhật.

## 6.5. Monitoring Queries

Dagster lưu mọi thứ vào PostgreSQL. Có thể query trực tiếp:

```sql
-- Xem run history
SELECT run_id, status, start_time, end_time, 
       end_time - start_time as duration
FROM runs 
ORDER BY start_time DESC 
LIMIT 20;

-- Xem thống kê theo status
SELECT status, COUNT(*) as count
FROM runs
WHERE start_time > NOW() - INTERVAL '7 days'
GROUP BY status;

-- Xem materialization history cho 1 asset
SELECT * FROM event_logs
WHERE dagster_event_type = 'ASSET_MATERIALIZATION'
  AND asset_key = 'schema_embeddings'
ORDER BY timestamp DESC;
```

---

# Phần 7: Cấu hình & Deployment

## 7.1. dagster.yaml — Chi tiết

File cấu hình hiện tại của dự án:

```yaml
# orchestration/dagster.yaml

# ========================================
# Storage: Lưu run history, event logs
# ========================================
storage:
  postgres:
    postgres_db:
      username:
        env: DAGSTER_PG_USERNAME      # dagster
      password:
        env: DAGSTER_PG_PASSWORD      # dagster_password
      hostname:
        env: DAGSTER_PG_HOSTNAME      # postgres
      db_name:
        env: DAGSTER_PG_DB            # dagster
      port: 5432

# ========================================
# Run Launcher: Cách launch mỗi pipeline run
# ========================================
run_launcher:
  module: dagster.core.launcher
  class: DefaultRunLauncher
  # DefaultRunLauncher: chạy run trong cùng process
  # Có thể thay bằng:
  # - DockerRunLauncher (mỗi run = 1 container)
  # - K8sRunLauncher (mỗi run = 1 pod)

# ========================================
# Run Coordinator: Quản lý hàng đợi runs
# ========================================
run_coordinator:
  module: dagster.core.run_coordinator
  class: QueuedRunCoordinator        # Hàng đợi FIFO
  config:
    max_concurrent_runs:
      env: DAGSTER_OVERALL_CONCURRENCY_LIMIT  # 10

# ========================================
# Compute Logs: Stdout/stderr logs
# ========================================
compute_logs:
  module: dagster.core.storage.local_compute_log_manager
  class: LocalComputeLogManager
  config:
    base_dir: /opt/dagster/dagster_home/compute_logs

# ========================================
# Local Artifact Storage
# ========================================
local_artifact_storage:
  module: dagster.core.storage.root
  class: LocalArtifactStorage
  config:
    base_dir: /opt/dagster/dagster_home/local_artifact_storage

# ========================================
# Telemetry & Threading
# ========================================
telemetry:
  enabled: true

sensors:
  use_threads: true
  num_workers: 3         # 3 threads cho sensors

schedules:
  use_threads: true
  num_workers: 3         # 3 threads cho schedules
```

## 7.2. workspace.yaml — Code Locations

```yaml
# orchestration/workspace.yaml
load_from:
  - grpc_server:
      host: dagster_orchestration    # Container name
      port: 4000                     # gRPC port
      location_name: "orchestration"
```

**Giải thích:** Webserver và Daemon kết nối tới Code Server qua gRPC. Code Server chứa toàn bộ pipeline code (definitions.py + defs/).

## 7.3. Docker containers

### Container 1: dagster_webserver

```yaml
# Entrypoint
entrypoint:
  - dagster-webserver
  - -h "0.0.0.0"
  - -p "3000"
  - -w /opt/dagster/dagster_home/workspace.yaml

# Volumes
volumes:
  - ./orchestration/dagster.yaml:/opt/dagster/dagster_home/dagster.yaml
  - ./orchestration/workspace.yaml:/opt/dagster/dagster_home/workspace.yaml
  - ./volumes/dagster_home:/opt/dagster/dagster_home
```

### Container 2: dagster_daemon

```yaml
# Entrypoint
working_dir: /opt/dagster/dagster_home
entrypoint:
  - dagster-daemon
  - run

# Daemon chạy:
# - SchedulerDaemon: tick schedules theo cron
# - SensorDaemon: tick sensors theo interval
# - QueuedRunCoordinatorDaemon: dequeue runs
```

### Container 3: dagster_orchestration (Code Server)

```yaml
# Entrypoint
entrypoint:
  - dagster api grpc
  - -h "0.0.0.0"
  - -p "4000"
  - -m orchestration.definitions    # Module chứa @definitions
```

## 7.4. Biến môi trường

```bash
# orchestration/.env (copy từ orchestration.env.example)

# PostgreSQL cho Dagster storage
DAGSTER_PG_USERNAME=dagster
DAGSTER_PG_PASSWORD=dagster_password
DAGSTER_PG_HOSTNAME=postgres
DAGSTER_PG_PORT=5432
DAGSTER_PG_DB=dagster

# Concurrency
DAGSTER_OVERALL_CONCURRENCY_LIMIT=10

# (Thêm nếu cần) Kết nối TalkingWithData services
# TALKWDATA_SERVER_URL=http://server:8000
# OLLAMA_BASE_URL=http://ollama:11434
# QDRANT_HOST=qdrant
# QDRANT_PORT=6333
```

---

# Phần 8: Best Practices & Troubleshooting

## 8.1. Best Practices

### Naming Convention

```python
# Assets: danh từ (mô tả data)
@asset
def raw_schema_metadata(): ...    # ✅ noun
def extract_schema(): ...          # ❌ verb

# Ops: động từ (mô tả hành động)
@op
def extract_tables(): ...          # ✅ verb
def tables(): ...                  # ❌ noun

# Groups: theo domain
group_name="schema"      # Schema management assets
group_name="training"    # ML training assets
group_name="sync"        # Sync/update assets
group_name="analytics"   # Reporting assets
```

### Idempotency (chạy lại an toàn)

```python
@asset
def schema_embeddings(context, raw_schema_metadata):
    """
    IDEMPOTENT: Dùng upsert thay vì insert.
    Chạy lại N lần → kết quả giống nhau.
    """
    for table in raw_schema_metadata["tables"]:
        # ✅ Upsert (update nếu tồn tại, insert nếu chưa)
        qdrant_resource.upsert(id=table_id, vector=vector, payload=payload)
        
        # ❌ Insert (chạy lại → duplicate)
        # qdrant_resource.insert(vector=vector, payload=payload)
```

### Error Handling

```python
from dagster import Failure, RetryPolicy

@asset(
    retry_policy=RetryPolicy(
        max_retries=3,
        delay=10  # seconds between retries
    )
)
def fragile_asset(context):
    try:
        result = call_external_api()
        return result
    except ConnectionError as e:
        # Dagster sẽ retry 3 lần
        raise Failure(
            description=f"API connection failed: {e}",
            metadata={"error": str(e)}
        )
    except ValueError as e:
        # Không retry — lỗi logic
        raise Failure(
            description=f"Invalid data: {e}",
            metadata={"error": str(e)},
        )
```

### Resource Cleanup

```python
from contextlib import contextmanager

class SourceDatabaseResource(ConfigurableResource):
    connection_string: str
    
    @contextmanager
    def get_connection(self):
        """Context manager đảm bảo connection được đóng"""
        engine = create_engine(self.connection_string)
        conn = engine.connect()
        try:
            yield conn
        finally:
            conn.close()
            engine.dispose()

# Sử dụng
@asset
def my_asset(source_db: SourceDatabaseResource):
    with source_db.get_connection() as conn:
        result = conn.execute(text("SELECT * FROM ..."))
```

## 8.2. Testing

```python
# tests/test_assets.py

from dagster import materialize
from orchestration.defs.assets.schema_assets import raw_schema_metadata


def test_raw_schema_metadata():
    """Test asset với mock config"""
    result = materialize(
        [raw_schema_metadata],
        run_config={
            "ops": {
                "raw_schema_metadata": {
                    "config": {
                        "connection_string": "postgresql://test:test@localhost:5432/test_db",
                        "database_name": "test_db"
                    }
                }
            }
        }
    )
    
    assert result.success
    
    # Kiểm tra output
    output = result.output_for_node("raw_schema_metadata")
    assert "tables" in output
    assert output["database_name"] == "test_db"


def test_schema_diff_no_changes():
    """Test schema diff khi không có thay đổi"""
    # Mock resources
    from unittest.mock import MagicMock
    
    mock_qdrant = MagicMock()
    mock_qdrant.get_client.return_value.scroll.return_value = ([], None)
    
    result = materialize(
        [schema_diff],
        resources={"qdrant_resource": mock_qdrant},
        run_config={...}
    )
    
    assert result.success
    output = result.output_for_node("schema_diff")
    assert output["has_changes"] == False
```

## 8.3. Troubleshooting

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `Could not connect to gRPC server` | Code Server chưa khởi động xong | Đợi container `dagster_orchestration` healthy |
| `DagsterEventLogInvalidForRun` | PostgreSQL storage lỗi | Restart dagster_webserver, kiểm tra disk space |
| `RunQueueFull` | Quá nhiều runs đang chờ | Tăng `DAGSTER_OVERALL_CONCURRENCY_LIMIT` |
| `ModuleNotFoundError` | Code chưa được install trong container | Rebuild image: `docker compose build dagster_orchestration` |
| `Sensor tick failed` | API server không reachable | Kiểm tra network giữa containers |

### Debug commands

```bash
# Xem logs webserver
docker logs talkwdata_dagster_webserver --tail 100

# Xem logs daemon (schedules, sensors)
docker logs talkwdata_dagster_daemon --tail 100

# Xem logs code server
docker logs talkwdata_dagster_orchestration --tail 100

# Restart code server (sau khi update code)
docker compose restart dagster_orchestration

# Rebuild (sau khi thêm dependencies)
docker compose build dagster_orchestration
docker compose up -d dagster_orchestration dagster_webserver dagster_daemon

# Kiểm tra gRPC connection
docker exec talkwdata_dagster_webserver dagster api grpc-health-check -p 4000 -h dagster_orchestration

# Chạy pipeline cục bộ (dev mode)
cd orchestration
dagster dev -m orchestration.definitions
```

### Chạy local (development)

```bash
# Cài đặt
cd orchestration
pip install -e ".[dev]"

# Khởi động Dagster dev (gộp webserver + daemon + code server)
dagster dev -m orchestration.definitions

# Mở browser: http://localhost:3000
```

## 8.4. Flow tracking tổng hợp

Dưới đây là **full picture** về cách Dagster tracking mọi data flow trong TalkingWithData:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DAGSTER TRACKING                              │
│                                                                      │
│  📊 Asset Catalog                                                    │
│  ├── schema/raw_schema_metadata     [✅ Fresh   | 2026-03-01 10:00] │
│  ├── schema/schema_embeddings       [✅ Fresh   | 2026-03-01 10:02] │
│  ├── training/trained_model         [⚠️ Stale   | 2026-02-28 08:00] │
│  ├── sync/schema_diff               [✅ Fresh   | 2026-03-01 12:00] │
│  ├── sync/updated_embeddings        [✅ Fresh   | 2026-03-01 12:01] │
│  └── analytics/daily_query_stats    [📅 15/15 partitions]            │
│                                                                      │
│  🔄 Recent Runs                                                      │
│  ├── Run a1b2c3d4 | schema_import  | ✅ SUCCESS  | 2m 35s           │
│  ├── Run e5f6g7h8 | schema_sync    | ✅ SUCCESS  | 45s              │
│  ├── Run i9j0k1l2 | analytics      | ❌ FAILURE  | 12s (retry 1/3) │
│  └── Run m3n4o5p6 | schema_import  | 🔄 RUNNING  | 1m 20s...       │
│                                                                      │
│  ⏰ Schedules                                                        │
│  ├── schema_sync_schedule     | */6h  | Next: 18:00 | ✅ ON          │
│  └── daily_analytics_schedule | 00:30 | Next: 00:30 | ✅ ON          │
│                                                                      │
│  👁 Sensors                                                           │
│  ├── new_connection_sensor    | 30s   | Last: 5 min ago | ✅ ON      │
│  └── schema_change_sensor    | 5min  | Last: 3 min ago | ✅ ON      │
│                                                                      │
│  📈 Metadata (per asset materialization)                             │
│  └── schema_embeddings [Run a1b2c3d4]:                               │
│      ├── database_name: "ecommerce_db"                               │
│      ├── embedded_tables: 15                                         │
│      ├── vector_dimension: 768                                       │
│      └── collection: "talkwdata_schemas"                             │
└──────────────────────────────────────────────────────────────────────┘
```

**Mọi thứ đều observable qua UI tại http://localhost:3000** — không cần đọc log thủ công.
