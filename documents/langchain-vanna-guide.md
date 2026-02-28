# LangChain & Vanna 2.0 — Hướng dẫn Chi tiết

> **Ngữ cảnh:** Áp dụng trong dự án TalkingWithData (Text-to-SQL với Ollama local)  
> **Cập nhật:** 01/03/2026

---

## Mục lục

- [Phần 1: LangChain — Tổng quan & Kiến trúc](#phần-1-langchain--tổng-quan--kiến-trúc)
- [Phần 2: LangChain — Cách sử dụng Chi tiết](#phần-2-langchain--cách-sử-dụng-chi-tiết)
- [Phần 3: Vanna 2.0 — Text-to-SQL chuyên biệt](#phần-3-vanna-20--text-to-sql-chuyên-biệt)
- [Phần 4: So sánh LangChain vs Vanna 2.0](#phần-4-so-sánh-langchain-vs-vanna-20)
- [Phần 5: Áp dụng vào TalkingWithData](#phần-5-áp-dụng-vào-talkingwithdata)

---

# Phần 1: LangChain — Tổng quan & Kiến trúc

## 1.1. LangChain là gì?

LangChain là một **framework mã nguồn mở** (Python/JS) để xây dựng ứng dụng sử dụng Large Language Models (LLMs). Nó cung cấp các abstraction layer giúp:

- Kết nối LLM với dữ liệu bên ngoài (databases, APIs, documents)
- Xây dựng chuỗi xử lý (chains) phức tạp
- Quản lý bộ nhớ hội thoại (memory)
- Tạo AI agents có khả năng sử dụng tools

```
                    LangChain Ecosystem
┌──────────────────────────────────────────────────┐
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │langchain │  │langchain │  │ langchain-     │  │
│  │-core     │  │-community│  │ ollama/openai/ │  │
│  │(base)    │  │(3rd party│  │ (integrations) │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │LangGraph │  │LangSmith │  │ LangServe     │  │
│  │(agents,  │  │(tracing, │  │ (deploy as    │  │
│  │ workflow) │  │ debug)   │  │  REST API)    │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
└──────────────────────────────────────────────────┘
```

## 1.2. Các thành phần chính

| Thành phần | Mô tả | Ví dụ |
|------------|--------|-------|
| **Models** | Kết nối với LLM | ChatOllama, ChatOpenAI |
| **Prompts** | Template quản lý prompt | ChatPromptTemplate, FewShotPromptTemplate |
| **Chains** | Chuỗi xử lý tuần tự | LLMChain, SequentialChain, LCEL |
| **Memory** | Bộ nhớ hội thoại | ConversationBufferMemory |
| **Retrievers** | Truy xuất dữ liệu | VectorStoreRetriever |
| **Agents** | LLM tự chọn tool | SQL Agent, Custom Agent |
| **Tools** | Công cụ cho Agent | SQLDatabaseTool, PythonREPL |
| **Output Parsers** | Parse output LLM | StrOutputParser, JsonOutputParser |

## 1.3. Cài đặt

```bash
# Core packages
pip install langchain langchain-core langchain-community

# Ollama integration (dùng LLM local — phù hợp TalkingWithData)
pip install langchain-ollama

# SQL & Database tools
pip install langchain-experimental

# Vector store - Qdrant
pip install langchain-qdrant
```

---

# Phần 2: LangChain — Cách sử dụng Chi tiết

## 2.1. Kết nối LLM (Ollama)

### Cơ bản — Chat Model

```python
from langchain_ollama import ChatOllama

# Kết nối Ollama local (giống TalkingWithData)
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",  # hoặc http://ollama:11434 trong Docker
    temperature=0,  # 0 = deterministic (tốt cho SQL generation)
)

# Gọi đơn giản
response = llm.invoke("Xin chào, bạn là ai?")
print(response.content)
```

### Embedding Model

```python
from langchain_ollama import OllamaEmbeddings

# Dùng nomic-embed-text (đã có trong TalkingWithData)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Tạo vector cho 1 đoạn text
vector = embeddings.embed_query("Danh sách đơn hàng tháng 1")
print(f"Vector dimension: {len(vector)}")  # 768

# Tạo vectors cho nhiều texts
vectors = embeddings.embed_documents([
    "Table orders: id, customer_id, total, created_at",
    "Table customers: id, name, email, phone"
])
```

## 2.2. Prompt Templates

### ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

# Tạo prompt template cho text-to-SQL
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL expert. Convert questions to PostgreSQL queries.
Only output the SQL query, nothing else.
Use this schema:
{schema}"""),
    ("human", "{question}")
])

# Format prompt
formatted = prompt.format_messages(
    schema="Table: orders (id INT, customer VARCHAR, total DECIMAL, created_at DATE)",
    question="Tổng doanh thu tháng 1?"
)

# Gọi LLM
response = llm.invoke(formatted)
print(response.content)
# → SELECT SUM(total) FROM orders WHERE EXTRACT(MONTH FROM created_at) = 1
```

### FewShotPromptTemplate (dạy LLM bằng ví dụ)

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# Các ví dụ mẫu (rất quan trọng cho text-to-SQL)
examples = [
    {
        "input": "Có bao nhiêu khách hàng?",
        "output": "SELECT COUNT(*) FROM customers;"
    },
    {
        "input": "Top 5 sản phẩm bán chạy nhất?",
        "output": "SELECT product_name, SUM(quantity) as total_sold FROM order_items GROUP BY product_name ORDER BY total_sold DESC LIMIT 5;"
    },
    {
        "input": "Doanh thu trung bình mỗi tháng?",
        "output": "SELECT EXTRACT(MONTH FROM created_at) as month, AVG(total) as avg_revenue FROM orders GROUP BY month ORDER BY month;"
    }
]

# Template cho mỗi ví dụ
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# FewShot template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# Kết hợp vào prompt chính
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL expert. Schema:\n{schema}"),
    few_shot_prompt,
    ("human", "{question}")
])
```

## 2.3. Chains (LCEL — LangChain Expression Language)

LCEL là cách hiện đại để tạo chain trong LangChain, sử dụng toán tử `|` (pipe).

### Chain cơ bản

```python
from langchain_core.output_parsers import StrOutputParser

# Chain: prompt → LLM → parse output
chain = prompt | llm | StrOutputParser()

# Chạy chain
result = chain.invoke({
    "schema": "Table: orders (id INT, total DECIMAL, created_at DATE)",
    "question": "Tổng doanh thu?"
})
print(result)  # "SELECT SUM(total) FROM orders;"
```

### Chain phức tạp (multi-step)

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Step 1: Sinh SQL
sql_chain = sql_prompt | llm | StrOutputParser()

# Step 2: Thực thi SQL (custom function)
def execute_sql(sql_query: str) -> str:
    """Thực thi SQL trên database nguồn"""
    from sqlalchemy import create_engine, text
    engine = create_engine("postgresql://user:pass@localhost:5432/mydb")
    with engine.connect() as conn:
        result = conn.execute(text(sql_query))
        rows = result.fetchall()
        return str(rows)

# Step 3: Tạo câu trả lời tự nhiên
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Dựa trên kết quả SQL, trả lời câu hỏi bằng ngôn ngữ tự nhiên."),
    ("human", "Câu hỏi: {question}\nSQL: {sql}\nKết quả: {result}\n\nTrả lời:")
])

# Full chain
full_chain = (
    RunnablePassthrough.assign(
        sql=sql_chain  # bước 1: sinh SQL
    )
    | RunnablePassthrough.assign(
        result=lambda x: execute_sql(x["sql"])  # bước 2: chạy SQL
    )
    | answer_prompt  # bước 3: format prompt
    | llm           # bước 3: gọi LLM
    | StrOutputParser()  # bước 3: parse output
)

# Chạy
answer = full_chain.invoke({
    "schema": "Table: orders (id, total, created_at)",
    "question": "Tổng doanh thu tháng 1?"
})
```

## 2.4. Memory (Bộ nhớ hội thoại)

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder

# Memory giữ lại N lượt hội thoại gần nhất
memory = ConversationBufferWindowMemory(
    k=10,  # giữ 10 lượt cuối
    return_messages=True,
    memory_key="chat_history"
)

# Prompt có chỗ cho history
prompt_with_memory = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL expert. Schema:\n{schema}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Lưu tin nhắn
memory.save_context(
    {"input": "Có bao nhiêu đơn hàng?"},
    {"output": "SELECT COUNT(*) FROM orders;"}
)

# Lấy history
history = memory.load_memory_variables({})["chat_history"]
```

## 2.5. Vector Store — Qdrant

```python
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

# Kết nối Qdrant (đã có trong TalkingWithData)
qdrant_client = QdrantClient(host="localhost", port=6333)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Tạo vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="database_schemas",
    embedding=embeddings
)

# Thêm documents
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Table: orders (id INT PK, customer_id INT FK, total DECIMAL, created_at TIMESTAMP)",
        metadata={"database": "ecommerce", "table": "orders"}
    ),
    Document(
        page_content="Table: customers (id INT PK, name VARCHAR, email VARCHAR, phone VARCHAR)",
        metadata={"database": "ecommerce", "table": "customers"}
    ),
]

vector_store.add_documents(docs)

# Tìm kiếm (semantic search)
results = vector_store.similarity_search(
    query="đơn hàng của khách hàng",
    k=3
)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

### Retriever (dùng trong chain)

```python
# Tạo retriever từ vector store
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Dùng trong chain
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "schema": retriever | format_docs,  # tự động tìm schema phù hợp
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Chạy — tự động tìm schema liên quan rồi sinh SQL
sql = rag_chain.invoke("Tổng doanh thu theo từng khách hàng?")
```

## 2.6. SQL Database Tools & Agent

LangChain có sẵn tools cho SQL:

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

# Kết nối database
db = SQLDatabase.from_uri("postgresql://user:pass@localhost:5432/mydb")

# Xem thông tin
print(db.get_usable_table_names())
print(db.get_table_info())  # Schema đầy đủ

# Tạo toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Tạo SQL Agent (tự động sinh SQL, chạy, sửa lỗi)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type="tool-calling",
    verbose=True  # In ra từng bước suy luận
)

# Hỏi
result = agent.invoke({
    "input": "Top 3 khách hàng có doanh thu cao nhất?"
})
print(result["output"])
```

**SQL Agent tự động:**
1. Xem schema database
2. Sinh câu SQL
3. Thực thi SQL
4. Nếu lỗi → tự sửa SQL và chạy lại
5. Format kết quả thành câu trả lời

## 2.7. Custom Tools

```python
from langchain_core.tools import tool

@tool
def search_schema(query: str) -> str:
    """Tìm kiếm schema database phù hợp với câu hỏi."""
    results = vector_store.similarity_search(query, k=5)
    return "\n".join(doc.page_content for doc in results)

@tool
def execute_sql_query(sql_query: str) -> str:
    """Thực thi câu SQL trên database và trả về kết quả."""
    from sqlalchemy import create_engine, text
    engine = create_engine("postgresql://user:pass@localhost:5432/mydb")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = list(result.keys())
            return str([dict(zip(columns, row)) for row in rows[:50]])
    except Exception as e:
        return f"Error: {str(e)}"

# Agent với custom tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst assistant. 
    When asked a question:
    1. First search for relevant database schema
    2. Generate a SQL query based on the schema
    3. Execute the SQL query
    4. Provide a clear answer"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=[search_schema, execute_sql_query],
    prompt=agent_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_schema, execute_sql_query],
    verbose=True,
    max_iterations=5
)

# Chạy
result = agent_executor.invoke({"input": "Tổng doanh thu tháng này là bao nhiêu?"})
```

## 2.8. Output Parsers

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Định nghĩa output format
class SQLResult(BaseModel):
    sql_query: str = Field(description="The SQL query")
    explanation: str = Field(description="Brief explanation of the query")

parser = JsonOutputParser(pydantic_object=SQLResult)

prompt_with_format = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL expert. {format_instructions}"),
    ("human", "Schema: {schema}\nQuestion: {question}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt_with_format | llm | parser

result = chain.invoke({
    "schema": "Table: orders (id, total, created_at)",
    "question": "Tổng doanh thu?"
})
# result = {"sql_query": "SELECT SUM(total) FROM orders;", "explanation": "..."}
```

---

# Phần 3: Vanna 2.0 — Text-to-SQL chuyên biệt

## 3.1. Vanna là gì?

**Vanna** là thư viện Python **chuyên biệt cho Text-to-SQL**, sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)**. Không giống LangChain (general-purpose), Vanna tập trung hoàn toàn vào việc chuyển ngôn ngữ tự nhiên thành SQL.

```
┌─────────────────────────────────────────────────────┐
│                    Vanna 2.0                         │
│                                                     │
│   ┌───────────┐     ┌───────────┐     ┌──────────┐ │
│   │  Training │     │    RAG    │     │   SQL    │ │
│   │  Data     │────▶│  Retrieval│────▶│Generation│ │
│   │  (DDL,    │     │  (Vector  │     │  (LLM)   │ │
│   │   Q&A,    │     │   Search) │     │          │ │
│   │   docs)   │     └───────────┘     └──────────┘ │
│   └───────────┘                                     │
│                                                     │
│   Supported LLMs: Ollama, OpenAI, Anthropic, ...    │
│   Supported VectorDBs: Qdrant, ChromaDB, ...        │
│   Supported DBs: PostgreSQL, MySQL, SQLite, ...     │
└─────────────────────────────────────────────────────┘
```

### Vanna 2.0 vs 1.x

| Tính năng | Vanna 1.x | Vanna 2.0 |
|-----------|-----------|-----------|
| Kiến trúc | Monolithic | Modular (plugin-based) |
| LLM | Chỉ OpenAI/Mistral | Bất kỳ (Ollama, OpenAI, ...) |
| Vector Store | ChromaDB built-in | Bất kỳ (Qdrant, ChromaDB, ...) |
| Customization | Hạn chế | Tự do kết hợp components |
| Training | Cơ bản | DDL + Documentation + Q&A pairs |

## 3.2. Cài đặt

```bash
# Core
pip install vanna

# Với Ollama (local LLM)
pip install 'vanna[ollama]'

# Với Qdrant (vector store)
pip install 'vanna[qdrant]'

# Hoặc cài hết
pip install 'vanna[ollama,qdrant]'
```

## 3.3. Kiến trúc Vanna 2.0

Vanna 2.0 dùng **mixin pattern** — bạn tạo class kết hợp LLM + VectorStore tùy ý:

```python
# Kết hợp: Ollama (LLM) + Qdrant (Vector Store)
from vanna.ollama import Ollama
from vanna.qdrant import Qdrant_VectorStore

class MyVanna(Qdrant_VectorStore, Ollama):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
```

Các kết hợp có thể:

| LLM (chọn 1) | Vector Store (chọn 1) |
|---------------|----------------------|
| `Ollama` | `Qdrant_VectorStore` |
| `OpenAI_Chat` | `ChromaDB_VectorStore` |
| `Anthropic_Chat` | `Pinecone_VectorStore` |
| `Mistral` | `FAISS_VectorStore` |
| Custom class | Custom class |

## 3.4. Setup Vanna với Ollama + Qdrant

```python
from vanna.ollama import Ollama
from vanna.qdrant import Qdrant_VectorStore


class TalkWithDataVanna(Qdrant_VectorStore, Ollama):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


# Khởi tạo
vn = TalkWithDataVanna(config={
    # Ollama config
    "model": "llama3.2",
    "ollama_host": "http://localhost:11434",
    
    # Qdrant config  
    "qdrant_host": "localhost",
    "qdrant_port": 6333,
    "collection_name": "talkwdata_schemas",
    
    # Embedding config (Vanna tự dùng Ollama để tạo embedding)
    "embedding_model": "nomic-embed-text"
})

# Kết nối database nguồn
vn.connect_to_postgres(
    host="localhost",
    port=5432,
    dbname="sample_db",
    user="user",
    password="password"
)
```

## 3.5. Training — Dạy Vanna hiểu Database

### Training bằng DDL (cấu trúc bảng)

```python
# Cách 1: Train từ DDL string
vn.train(ddl="""
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(id),
        total DECIMAL(10,2) NOT NULL,
        status VARCHAR(20) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE customers (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(255) UNIQUE,
        phone VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE order_items (
        id SERIAL PRIMARY KEY,
        order_id INTEGER REFERENCES orders(id),
        product_id INTEGER REFERENCES products(id),
        quantity INTEGER NOT NULL,
        unit_price DECIMAL(10,2) NOT NULL
    );
    
    CREATE TABLE products (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        category VARCHAR(50),
        price DECIMAL(10,2) NOT NULL,
        stock INTEGER DEFAULT 0
    );
""")

# Cách 2: Auto-train từ database (đọc information_schema)
# Vanna tự kết nối DB và extract DDL
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
plan = vn.get_training_plan_generic(df_information_schema)
vn.train(plan=plan)
```

### Training bằng Documentation (mô tả nghiệp vụ)

```python
# Giúp LLM hiểu ngữ cảnh nghiệp vụ
vn.train(documentation="""
    Hệ thống quản lý bán hàng:
    - Bảng orders: lưu đơn hàng, status có thể là 'pending', 'confirmed', 'shipped', 'delivered', 'cancelled'
    - Bảng customers: thông tin khách hàng  
    - Doanh thu = SUM(total) của orders có status = 'delivered'
    - Khách hàng VIP = khách có tổng đơn hàng > 10 triệu
    - Tháng tài chính bắt đầu từ ngày 1
""")

# Mô tả từng bảng
vn.train(documentation="Bảng orders.status: 'pending'=chờ xác nhận, 'confirmed'=đã xác nhận, 'shipped'=đang giao, 'delivered'=đã giao, 'cancelled'=đã hủy")
vn.train(documentation="Bảng products.category: 'electronics', 'clothing', 'food', 'books'")
```

### Training bằng Question-SQL pairs (ví dụ mẫu)

```python
# Dạy Vanna bằng các cặp câu hỏi-SQL mẫu
vn.train(
    question="Có bao nhiêu đơn hàng trong tháng này?",
    sql="SELECT COUNT(*) FROM orders WHERE DATE_TRUNC('month', created_at) = DATE_TRUNC('month', CURRENT_DATE);"
)

vn.train(
    question="Top 5 khách hàng có doanh thu cao nhất?",
    sql="""
        SELECT c.name, SUM(o.total) as total_revenue
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        WHERE o.status = 'delivered'
        GROUP BY c.id, c.name
        ORDER BY total_revenue DESC
        LIMIT 5;
    """
)

vn.train(
    question="Sản phẩm nào bán chạy nhất?",
    sql="""
        SELECT p.name, SUM(oi.quantity) as total_sold
        FROM products p
        JOIN order_items oi ON p.id = oi.product_id
        GROUP BY p.id, p.name
        ORDER BY total_sold DESC
        LIMIT 1;
    """
)

vn.train(
    question="Doanh thu trung bình mỗi đơn hàng?",
    sql="SELECT AVG(total) as avg_order_value FROM orders WHERE status = 'delivered';"
)
```

### Xem dữ liệu training

```python
# Xem tất cả training data đã lưu
training_data = vn.get_training_data()
print(training_data)

# Xóa 1 training data
vn.remove_training_data(id="xxx")
```

## 3.6. Sinh SQL — Sử dụng

### Cơ bản

```python
# Sinh SQL từ câu hỏi
sql = vn.generate_sql("Có bao nhiêu đơn hàng trong tháng 1?")
print(sql)
# → SELECT COUNT(*) FROM orders WHERE EXTRACT(MONTH FROM created_at) = 1;
```

### Sinh SQL + Chạy + Trả kết quả

```python
# Chạy SQL và lấy kết quả (DataFrame)
df = vn.run_sql(sql)
print(df)

# Hoặc 1 bước: hỏi → SQL → chạy → kết quả
result = vn.ask("Tổng doanh thu theo tháng?")
# result chứa: sql, DataFrame, plotly chart (nếu có)
```

### ask() — Full pipeline

```python
result = vn.ask(
    question="Top 5 sản phẩm bán chạy nhất tháng này?",
    print_results=True,     # In kết quả
    auto_train=True,        # Tự động lưu Q&A pair nếu user confirm
    visualize=True          # Tạo chart Plotly
)

# result trả về:
# - result.sql: câu SQL
# - result.df: DataFrame kết quả
# - result.fig: Plotly figure (nếu phù hợp)
# - result.summary: Tóm tắt kết quả
```

## 3.7. Follow-up Questions

```python
# Vanna gợi ý câu hỏi tiếp theo
followups = vn.generate_followup_questions(
    question="Tổng doanh thu tháng này?",
    sql=sql,
    df=df
)
print(followups)
# → ["Doanh thu so với tháng trước thế nào?",
#     "Khách hàng nào đóng góp nhiều nhất?",
#     "Xu hướng doanh thu 6 tháng gần đây?"]
```

## 3.8. Vanna Flask UI (bonus)

```python
from vanna.flask import VannaFlaskApp

# Tạo web UI đơn giản
app = VannaFlaskApp(vn)
app.run()
# → Mở browser http://localhost:8084
```

---

# Phần 4: So sánh LangChain vs Vanna 2.0

## 4.1. Bảng so sánh

| Tiêu chí | LangChain | Vanna 2.0 |
|----------|-----------|-----------|
| **Mục đích** | General-purpose LLM framework | Chuyên biệt Text-to-SQL |
| **Độ phức tạp** | Cao (nhiều concept) | Thấp (focus vào SQL) |
| **Learning curve** | Dốc | Dễ tiếp cận |
| **Customization** | Rất linh hoạt | Vừa phải |
| **Text-to-SQL** | Cần tự build chain/agent | Built-in, out-of-the-box |
| **Training (RAG)** | Tự implement | `vn.train()` — 1 dòng |
| **Auto-correction** | Tự implement | Tự động retry khi SQL lỗi |
| **Visualization** | Không có | Plotly charts tự động |
| **Memory** | Có (nhiều loại) | Hạn chế |
| **Agent** | Mạnh (multi-tool) | Không có |
| **Ecosystem** | Rất lớn | Nhỏ, chuyên biệt |
| **Ollama support** | ✅ Tốt | ✅ Tốt |
| **Qdrant support** | ✅ Tốt | ✅ Tốt |

## 4.2. Khi nào dùng cái nào?

### Dùng **Vanna 2.0** khi:
- Focus chính là Text-to-SQL
- Muốn setup nhanh, ít code
- Cần training data management built-in
- Cần auto-correction SQL
- Demo/prototype nhanh

### Dùng **LangChain** khi:
- Cần control chi tiết từng bước
- Cần memory/conversation management phức tạp
- Cần kết hợp nhiều tools (không chỉ SQL)
- Cần custom agent behavior
- Dự án phức tạp, nhiều integration

### Kết hợp cả hai:
- Dùng **Vanna** cho core Text-to-SQL engine
- Dùng **LangChain** cho conversation management, memory, và các tính năng phụ

---

# Phần 5: Áp dụng vào TalkingWithData

## 5.1. Phương án đề xuất

Dựa trên kiến trúc hiện tại của TalkingWithData (Ollama + Qdrant + PostgreSQL + FastAPI), có 3 phương án:

### Phương án A: Dùng Vanna 2.0 (🏆 Đề xuất cho Demo)

```
Ưu điểm: Setup nhanh, ít code, tự động training từ DB, auto-correction
Nhược điểm: Ít kiểm soát chi tiết

User Question → Vanna (RAG + LLM) → SQL → Execute → Answer
```

### Phương án B: Dùng LangChain

```
Ưu điểm: Linh hoạt, kiểm soát mọi bước, conversation memory
Nhược điểm: Code nhiều hơn, phải tự xử lý error recovery

User Question → Retriever (Qdrant) → Prompt + Schema → LLM → SQL → Execute → LLM → Answer
```

### Phương án C: Kết hợp Vanna + LangChain

```
Ưu điểm: Best of both worlds
Nhược điểm: Complexity cao hơn

Vanna: Text-to-SQL core
LangChain: Conversation memory + Additional tools
```

## 5.2. Triển khai Phương án A — Vanna 2.0

### Bước 1: Cập nhật `requirements.txt`

Thêm:
```
vanna[ollama,qdrant]==0.7.5
```

### Bước 2: Tạo Vanna instance — `shared/vanna_client.py`

```python
"""
Vanna 2.0 client cho TalkingWithData
Kết hợp: Ollama (LLM local) + Qdrant (Vector Store)
"""
from vanna.ollama import Ollama
from vanna.qdrant import Qdrant_VectorStore
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / ".server.env"
load_dotenv(dotenv_path=env_path)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


class TalkWithDataVanna(Qdrant_VectorStore, Ollama):
    """Custom Vanna class cho TalkingWithData"""
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


# Singleton instance
_vanna_instance = None


def get_vanna() -> TalkWithDataVanna:
    """Lấy Vanna instance (singleton pattern)"""
    global _vanna_instance
    
    if _vanna_instance is None:
        _vanna_instance = TalkWithDataVanna(config={
            # Ollama
            "model": OLLAMA_DEFAULT_MODEL,
            "ollama_host": OLLAMA_BASE_URL,
            
            # Qdrant
            "qdrant_host": QDRANT_HOST,
            "qdrant_port": QDRANT_PORT,
            "collection_name": "talkwdata_vanna",
            
            # Embedding (Vanna dùng Ollama embed)
            "embedding_model": "nomic-embed-text"
        })
    
    return _vanna_instance


def connect_database(connection_string: str):
    """Kết nối Vanna đến database nguồn"""
    vn = get_vanna()
    # Parse connection string
    # postgresql://user:pass@host:port/dbname
    from urllib.parse import urlparse
    parsed = urlparse(connection_string)
    
    vn.connect_to_postgres(
        host=parsed.hostname,
        port=parsed.port or 5432,
        dbname=parsed.path.lstrip('/'),
        user=parsed.username,
        password=parsed.password
    )


def train_from_database(connection_string: str):
    """Auto-train Vanna từ database schema"""
    vn = get_vanna()
    connect_database(connection_string)
    
    # Lấy thông tin schema
    df = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = 'public'")
    
    # Tạo training plan
    plan = vn.get_training_plan_generic(df)
    
    # Train
    vn.train(plan=plan)
    
    return {"message": "Training completed", "tables": len(df['table_name'].unique())}


def train_ddl(ddl: str):
    """Train Vanna bằng DDL string"""
    vn = get_vanna()
    vn.train(ddl=ddl)


def train_documentation(doc: str):
    """Train Vanna bằng documentation"""
    vn = get_vanna()
    vn.train(documentation=doc)


def train_question_sql(question: str, sql: str):
    """Train Vanna bằng cặp question-SQL"""
    vn = get_vanna()
    vn.train(question=question, sql=sql)


def ask_question(question: str) -> dict:
    """
    Hỏi dữ liệu bằng ngôn ngữ tự nhiên
    
    Returns:
        dict: {
            "question": str,
            "sql_query": str,
            "answer": str,
            "query_result": list[dict] | None,
            "followup_questions": list[str]
        }
    """
    vn = get_vanna()
    
    # 1. Sinh SQL
    sql = vn.generate_sql(question)
    
    if not sql or "CANNOT" in sql.upper():
        return {
            "question": question,
            "sql_query": None,
            "answer": "Không thể tạo câu truy vấn từ câu hỏi này.",
            "query_result": None,
            "followup_questions": []
        }
    
    # 2. Thực thi SQL
    try:
        df = vn.run_sql(sql)
        query_result = df.to_dict(orient='records') if df is not None else None
    except Exception as e:
        return {
            "question": question,
            "sql_query": sql,
            "answer": f"Lỗi khi thực thi SQL: {str(e)}",
            "query_result": None,
            "followup_questions": []
        }
    
    # 3. Tạo câu trả lời tự nhiên
    try:
        summary = vn.generate_summary(question=question, df=df)
    except Exception:
        summary = f"Kết quả truy vấn: {query_result}"
    
    # 4. Gợi ý câu hỏi tiếp theo
    try:
        followups = vn.generate_followup_questions(
            question=question, sql=sql, df=df
        )
    except Exception:
        followups = []
    
    return {
        "question": question,
        "sql_query": sql,
        "answer": summary,
        "query_result": query_result,
        "followup_questions": followups or []
    }


def get_training_data():
    """Xem dữ liệu training đã lưu"""
    vn = get_vanna()
    return vn.get_training_data()


def remove_training_data(training_id: str):
    """Xóa 1 training data"""
    vn = get_vanna()
    vn.remove_training_data(id=training_id)
```

### Bước 3: Cập nhật text_to_data service dùng Vanna

Thay thế `text_to_data_service.py` phần `process_question()`:

```python
# Trong module/text_to_data/service/text_to_data_service.py

from shared.vanna_client import (
    ask_question, train_from_database, train_ddl, 
    train_documentation, train_question_sql, connect_database
)


def process_question(db: Session, question: str, database_name: str = None) -> dict:
    """
    Core: Hỏi dữ liệu qua Vanna 2.0
    
    Vanna tự động:
    1. Tìm schema phù hợp (RAG từ Qdrant)
    2. Sinh SQL (LLM Ollama)
    3. Thực thi SQL
    4. Tạo câu trả lời tự nhiên
    """
    # Nếu có database_name, kết nối đến DB đó trước
    if database_name:
        connection = db.query(DatabaseConnection).filter(
            DatabaseConnection.name == database_name,
            DatabaseConnection.is_active == True
        ).first()
        if connection:
            connect_database(connection.connection_string)
    
    result = ask_question(question)
    
    return {
        "question": result["question"],
        "sql_query": result.get("sql_query"),
        "answer": result.get("answer", "Không thể trả lời."),
        "query_result": result.get("query_result"),
        "schema_context": None
    }


def import_and_train(db: Session, data: SchemaImportRequest) -> dict:
    """Import schema + Auto-train Vanna"""
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == data.connection_id
    ).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    # Vanna auto-train từ database
    result = train_from_database(connection.connection_string)
    
    return result
```

### Bước 4: Thêm Training endpoints

```python
# module/text_to_data/endpoint/training_endpoint.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from shared.vanna_client import (
    train_ddl, train_documentation, train_question_sql,
    get_training_data, remove_training_data
)

router = APIRouter()


class TrainDDLRequest(BaseModel):
    ddl: str

class TrainDocRequest(BaseModel):
    documentation: str

class TrainQARequest(BaseModel):
    question: str
    sql: str


@router.post("/train/ddl")
def train_with_ddl(
    data: TrainDDLRequest,
    current_user: User = Depends(get_current_user_oauth2)
):
    """Train Vanna bằng DDL (cấu trúc bảng)"""
    train_ddl(data.ddl)
    return {"message": "DDL training completed"}


@router.post("/train/documentation")
def train_with_doc(
    data: TrainDocRequest,
    current_user: User = Depends(get_current_user_oauth2)
):
    """Train Vanna bằng documentation (mô tả nghiệp vụ)"""
    train_documentation(data.documentation)
    return {"message": "Documentation training completed"}


@router.post("/train/question-sql")
def train_with_qa(
    data: TrainQARequest,
    current_user: User = Depends(get_current_user_oauth2)
):
    """Train Vanna bằng cặp câu hỏi - SQL mẫu"""
    train_question_sql(data.question, data.sql)
    return {"message": "Q&A training completed"}


@router.get("/train/data")
def list_training_data(
    current_user: User = Depends(get_current_user_oauth2)
):
    """Xem danh sách training data"""
    data = get_training_data()
    return {"training_data": data.to_dict(orient='records') if data is not None else []}


@router.delete("/train/data/{training_id}")
def delete_training_data(
    training_id: str,
    current_user: User = Depends(get_current_user_oauth2)
):
    """Xóa 1 training data"""
    remove_training_data(training_id)
    return {"message": "Training data removed"}
```

## 5.3. Triển khai Phương án B — LangChain

### Bước 1: Cập nhật `requirements.txt`

```
langchain==0.3.20
langchain-core==0.3.40
langchain-community==0.3.18
langchain-ollama==0.3.2
langchain-qdrant==0.2.2
langchain-experimental==0.3.4
```

### Bước 2: LangChain-based text-to-SQL — `shared/langchain_client.py`

```python
"""
LangChain client cho TalkingWithData
Text-to-SQL pipeline: Question → Schema Retrieval → SQL Generation → Execution → Answer
"""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain.memory import ConversationBufferWindowMemory
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / ".server.env"
load_dotenv(dotenv_path=env_path)

# ================================================================
# CONFIG
# ================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# ================================================================
# COMPONENTS
# ================================================================

# LLM
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,  # Deterministic cho SQL generation
)

# Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

# Qdrant
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="langchain_schemas",
    embedding=embeddings
)

# Retriever
schema_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ================================================================
# PROMPTS
# ================================================================

SQL_SYSTEM_PROMPT = """You are a PostgreSQL expert. Convert natural language questions to SQL queries.

RULES:
1. ONLY generate SELECT queries
2. Use the provided schema exactly as given
3. Return ONLY the raw SQL query, no markdown, no explanation
4. If you cannot answer, respond with exactly: CANNOT_ANSWER
5. Use PostgreSQL syntax
"""

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", """Database schema:
{schema}

Question: {question}

SQL query:""")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful data analyst. Answer the question based on the SQL result. Use Vietnamese if the question is in Vietnamese. Be concise."),
    ("human", """Question: {question}
SQL: {sql_query}
Result: {query_result}

Answer:""")
])

# ================================================================
# MEMORY (per conversation)
# ================================================================

# Lưu trữ memory theo conversation_id
_memories = {}


def get_memory(conversation_id: str) -> ConversationBufferWindowMemory:
    """Lấy hoặc tạo memory cho conversation"""
    if conversation_id not in _memories:
        _memories[conversation_id] = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="chat_history"
        )
    return _memories[conversation_id]


def clear_memory(conversation_id: str):
    """Xóa memory khi xóa conversation"""
    _memories.pop(conversation_id, None)


# ================================================================
# SCHEMA MANAGEMENT
# ================================================================

def index_schema(database_name: str, table_name: str, columns: str, description: str = ""):
    """Lưu schema vào Qdrant Vector Store"""
    content = f"Table: {table_name} ({columns})"
    if description:
        content += f" -- {description}"
    
    doc = Document(
        page_content=content,
        metadata={
            "database_name": database_name,
            "table_name": table_name,
            "columns": columns,
            "description": description
        }
    )
    vector_store.add_documents([doc])


def index_schemas_batch(schemas: list[dict]):
    """Batch index nhiều schemas"""
    docs = []
    for s in schemas:
        content = f"Table: {s['table_name']} ({s['columns']})"
        if s.get('description'):
            content += f" -- {s['description']}"
        docs.append(Document(page_content=content, metadata=s))
    
    vector_store.add_documents(docs)


# ================================================================
# TEXT-TO-SQL PIPELINE
# ================================================================

def _format_schema_docs(docs: list[Document]) -> str:
    """Format retrieved documents thành schema string"""
    return "\n".join(doc.page_content for doc in docs)


def _clean_sql(sql: str) -> str:
    """Clean SQL output từ LLM"""
    sql = sql.strip()
    # Remove markdown code blocks
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()


def _execute_sql(sql_query: str, connection_string: str) -> list[dict]:
    """Thực thi SQL và trả kết quả"""
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        result = conn.execute(text(sql_query))
        rows = result.fetchall()
        columns = list(result.keys())
        data = [dict(zip(columns, row)) for row in rows]
        return data[:100]  # Giới hạn 100 rows


def process_question(
    question: str,
    connection_string: str,
    conversation_id: str = None
) -> dict:
    """
    Full Text-to-SQL pipeline với LangChain
    
    Flow:
    1. Retrieve relevant schema (Qdrant)
    2. Generate SQL (LLM + schema context + chat history)
    3. Execute SQL
    4. Generate natural language answer (LLM)
    5. Save to memory
    """
    # 1. Retrieve schema
    schema_docs = schema_retriever.invoke(question)
    schema_context = _format_schema_docs(schema_docs)
    
    if not schema_context:
        return {
            "question": question,
            "sql_query": None,
            "answer": "Không tìm thấy schema phù hợp.",
            "query_result": None,
            "schema_context": None
        }
    
    # 2. Get chat history (if conversation exists)
    chat_history = []
    if conversation_id:
        memory = get_memory(conversation_id)
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
    
    # 3. Generate SQL
    sql_chain = SQL_GENERATION_PROMPT | llm | StrOutputParser()
    
    raw_sql = sql_chain.invoke({
        "schema": schema_context,
        "question": question,
        "chat_history": chat_history
    })
    
    sql_query = _clean_sql(raw_sql)
    
    # Check if cannot answer
    if "CANNOT_ANSWER" in sql_query.upper():
        return {
            "question": question,
            "sql_query": None,
            "answer": "Không thể tạo truy vấn từ câu hỏi này với schema hiện tại.",
            "query_result": None,
            "schema_context": schema_context
        }
    
    # Validate SELECT only
    if not sql_query.upper().strip().startswith("SELECT"):
        return {
            "question": question,
            "sql_query": sql_query,
            "answer": "Chỉ cho phép truy vấn SELECT.",
            "query_result": None,
            "schema_context": schema_context
        }
    
    # 4. Execute SQL
    try:
        query_result = _execute_sql(sql_query, connection_string)
    except Exception as e:
        # Retry: gửi lỗi lại cho LLM để sửa SQL
        retry_prompt = ChatPromptTemplate.from_messages([
            ("system", SQL_SYSTEM_PROMPT),
            ("human", """Schema: {schema}
Question: {question}
Previous SQL: {previous_sql}
Error: {error}

Fix the SQL query. Return ONLY the corrected SQL:""")
        ])
        
        retry_chain = retry_prompt | llm | StrOutputParser()
        fixed_sql = retry_chain.invoke({
            "schema": schema_context,
            "question": question,
            "previous_sql": sql_query,
            "error": str(e)
        })
        sql_query = _clean_sql(fixed_sql)
        
        try:
            query_result = _execute_sql(sql_query, connection_string)
        except Exception as e2:
            return {
                "question": question,
                "sql_query": sql_query,
                "answer": f"Lỗi thực thi SQL (đã thử sửa): {str(e2)}",
                "query_result": None,
                "schema_context": schema_context
            }
    
    # 5. Generate answer
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    
    answer = answer_chain.invoke({
        "question": question,
        "sql_query": sql_query,
        "query_result": str(query_result[:20])  # Limit cho LLM
    })
    
    # 6. Save to memory
    if conversation_id:
        memory = get_memory(conversation_id)
        memory.save_context(
            {"input": question},
            {"output": f"SQL: {sql_query}\nAnswer: {answer}"}
        )
    
    return {
        "question": question,
        "sql_query": sql_query,
        "answer": answer,
        "query_result": query_result,
        "schema_context": schema_context
    }
```

**Ưu điểm so với viết tay (Phase 6 trong server-implementation-guide):**
- **Auto-retry**: Khi SQL lỗi, tự gửi error cho LLM sửa lại
- **Conversation memory**: Nhớ context hội thoại
- **LCEL chains**: Code rõ ràng, dễ debug

## 5.4. Triển khai Phương án C — Kết hợp Vanna + LangChain

```python
"""
shared/hybrid_client.py
Kết hợp: Vanna (Text-to-SQL core) + LangChain (Memory + Enhancement)
"""
from shared.vanna_client import get_vanna, ask_question
from langchain.memory import ConversationBufferWindowMemory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / ".server.env"
load_dotenv(dotenv_path=env_path)

# LangChain LLM (cho phần answer enhancement)
llm = ChatOllama(
    model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.3
)

# Memory per conversation
_memories = {}


def get_memory(conversation_id: str) -> ConversationBufferWindowMemory:
    if conversation_id not in _memories:
        _memories[conversation_id] = ConversationBufferWindowMemory(
            k=10, return_messages=True, memory_key="chat_history"
        )
    return _memories[conversation_id]


# Enhanced answer prompt
ENHANCED_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful data analyst for the TalkingWithData platform.
Answer questions based on SQL results. Use Vietnamese if the user asks in Vietnamese.
Be concise and informative. Format numbers with thousands separators."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", """Question: {question}
SQL query used: {sql_query}
Query result: {query_result}

Provide a clear answer:""")
])


def process_question_hybrid(
    question: str,
    conversation_id: str = None
) -> dict:
    """
    Hybrid pipeline:
    - Vanna: RAG + SQL Generation + Execution  (core engine)
    - LangChain: Memory + Enhanced Answer      (enhancement)
    """
    # 1. Lấy chat history
    chat_history = []
    if conversation_id:
        memory = get_memory(conversation_id)
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
    
    # 2. Enhance question với context (nếu có history)
    enhanced_question = question
    if chat_history:
        # Nếu câu hỏi có tham chiếu (ví dụ: "còn tháng 2 thì sao?")
        # → LLM rewrite thành câu hỏi đầy đủ
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the follow-up question to be self-contained, using the conversation history. If it's already clear, return it as-is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Follow-up question: {question}\n\nRewritten question:")
        ])
        rewrite_chain = rewrite_prompt | llm | StrOutputParser()
        enhanced_question = rewrite_chain.invoke({
            "question": question,
            "chat_history": chat_history
        }).strip()
    
    # 3. Vanna: Generate SQL + Execute
    vanna_result = ask_question(enhanced_question)
    
    # 4. LangChain: Enhanced answer
    answer_chain = ENHANCED_ANSWER_PROMPT | llm | StrOutputParser()
    
    enhanced_answer = answer_chain.invoke({
        "question": question,
        "sql_query": vanna_result.get("sql_query", "N/A"),
        "query_result": str(vanna_result.get("query_result", []))[:2000],
        "chat_history": chat_history
    })
    
    # 5. Save to memory
    if conversation_id:
        memory = get_memory(conversation_id)
        memory.save_context(
            {"input": question},
            {"output": enhanced_answer}
        )
    
    return {
        "question": question,
        "sql_query": vanna_result.get("sql_query"),
        "answer": enhanced_answer,
        "query_result": vanna_result.get("query_result"),
        "schema_context": None,
        "followup_questions": vanna_result.get("followup_questions", [])
    }
```

**Luồng hybrid:**
```
User: "Tổng doanh thu tháng 1?"
  │
  ├─ LangChain Memory: (trống, lượt đầu)
  ├─ Vanna: RAG → SQL → Execute → Raw result
  ├─ LangChain: Enhanced answer formatting
  └─ Lưu vào Memory

User: "Còn tháng 2 thì sao?"
  │
  ├─ LangChain Memory: có context "tổng doanh thu tháng 1"
  ├─ LangChain Rewrite: "Tổng doanh thu tháng 2?" (đầy đủ)
  ├─ Vanna: RAG → SQL → Execute
  ├─ LangChain: Enhanced answer (so sánh với tháng 1)
  └─ Lưu vào Memory
```

## 5.5. API endpoints bổ sung cho Training

```
POST /text-to-data/train/auto         ← Auto-train từ database (Vanna)
POST /text-to-data/train/ddl          ← Train bằng DDL
POST /text-to-data/train/documentation ← Train bằng mô tả nghiệp vụ
POST /text-to-data/train/question-sql  ← Train bằng cặp Q&A
GET  /text-to-data/train/data         ← Xem training data
DELETE /text-to-data/train/data/{id}  ← Xóa training data
```

## 5.6. Tổng kết — Đề xuất cho TalkingWithData

| Phương án | Độ khó | Thời gian | Chất lượng SQL | Ghi chú |
|-----------|--------|----------|---------------|---------|
| **A: Vanna** | ⭐ Dễ | 1-2 ngày | ⭐⭐⭐ Tốt | **Đề xuất cho demo** — ít code, tự training |
| **B: LangChain** | ⭐⭐⭐ Khó | 3-5 ngày | ⭐⭐ Trung bình | Linh hoạt, control cao |
| **C: Hybrid** | ⭐⭐ Vừa | 2-3 ngày | ⭐⭐⭐⭐ Rất tốt | Best of both — conversation + SQL |

**Đề xuất:** Bắt đầu với **Phương án A (Vanna)** cho demo, sau đó nâng cấp lên **Phương án C (Hybrid)** nếu cần conversation context.
