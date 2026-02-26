# API Documentation - TalkingWithData

## 1. Authentication APIs

### 1.1 Register
**POST** `/auth/register`

**Request:**
```json
{
  "email": "user@example.com",
  "username": "username",
  "password": "password123",
  "full_name": "John Doe"
}
```

**Response:** `201 Created`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "username": "username",
    "full_name": "John Doe",
    "is_active": true,
    "is_superuser": false,
    "created_at": "2026-02-15T10:00:00Z"
  }
}
```

### 1.2 Sign In
**POST** `/auth/signin`

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "username": "username",
    "full_name": "John Doe",
    "is_active": true,
    "is_superuser": false,
    "created_at": "2026-02-15T10:00:00Z"
  }
}
```

### 1.3 OAuth2 Token
**POST** `/auth/token`

**Request:** `application/x-www-form-urlencoded`
```
username=user@example.com
password=password123
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 1.4 Get Current User
**GET** `/auth/me`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "username": "username",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2026-02-15T10:00:00Z"
}
```

### 1.5 Sign Out
**POST** `/auth/signout`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `200 OK`
```json
{
  "message": "Successfully signed out"
}
```

---

## 2. Database Connection APIs

### 2.1 Create Connection
**POST** `/connections`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "name": "Production Database",
  "description": "Main production PostgreSQL database",
  "db_type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "username": "dbuser",
  "password": "dbpass",
  "schema": "public"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "name": "Production Database",
  "description": "Main production PostgreSQL database",
  "db_type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "schema": "public",
  "is_active": true,
  "created_at": "2026-02-15T10:00:00Z",
  "updated_at": "2026-02-15T10:00:00Z"
}
```

### 2.2 List Connections
**GET** `/connections`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 10)

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": "uuid",
      "name": "Production Database",
      "description": "Main production PostgreSQL database",
      "db_type": "postgresql",
      "host": "localhost",
      "port": 5432,
      "database": "mydb",
      "schema": "public",
      "is_active": true,
      "created_at": "2026-02-15T10:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 10
}
```

### 2.3 Get Connection
**GET** `/connections/{connection_id}`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "name": "Production Database",
  "description": "Main production PostgreSQL database",
  "db_type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "schema": "public",
  "is_active": true,
  "created_at": "2026-02-15T10:00:00Z"
}
```

### 2.4 Update Connection
**PUT** `/connections/{connection_id}`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "name": "Production Database Updated",
  "description": "Updated description",
  "is_active": true
}
```

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "name": "Production Database Updated",
  "description": "Updated description",
  "db_type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "schema": "public",
  "is_active": true,
  "updated_at": "2026-02-15T11:00:00Z"
}
```

### 2.5 Delete Connection
**DELETE** `/connections/{connection_id}`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `204 No Content`

### 2.6 Test Connection
**POST** `/connections/{connection_id}/test`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Connection successful"
}
```

---

## 3. Conversation APIs

### 3.1 Create Conversation
**POST** `/conversations`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "title": "Sales Analysis Q1 2026",
  "connection_id": "uuid"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "connection_id": "uuid",
  "title": "Sales Analysis Q1 2026",
  "created_at": "2026-02-15T10:00:00Z",
  "updated_at": "2026-02-15T10:00:00Z"
}
```

### 3.2 List Conversations
**GET** `/conversations`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `connection_id` (optional): Filter by connection
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 10)

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": "uuid",
      "title": "Sales Analysis Q1 2026",
      "connection_id": "uuid",
      "message_count": 5,
      "created_at": "2026-02-15T10:00:00Z",
      "updated_at": "2026-02-15T10:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 10
}
```

### 3.3 Get Conversation
**GET** `/conversations/{conversation_id}`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "connection_id": "uuid",
  "title": "Sales Analysis Q1 2026",
  "created_at": "2026-02-15T10:00:00Z",
  "updated_at": "2026-02-15T10:00:00Z",
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "Show me total sales by month",
      "created_at": "2026-02-15T10:00:00Z"
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "Here are the total sales by month...",
      "sql_query": "SELECT DATE_TRUNC('month', order_date) as month, SUM(total) as sales FROM orders GROUP BY month",
      "result_data": [...],
      "created_at": "2026-02-15T10:00:05Z"
    }
  ]
}
```

### 3.4 Update Conversation
**PUT** `/conversations/{conversation_id}`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "title": "Updated Title"
}
```

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "title": "Updated Title",
  "updated_at": "2026-02-15T11:00:00Z"
}
```

### 3.5 Delete Conversation
**DELETE** `/conversations/{conversation_id}`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `204 No Content`

---

## 4. Message APIs

### 4.1 Send Message (Ask Question)
**POST** `/conversations/{conversation_id}/messages`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "content": "What are the top 5 products by revenue in Q1 2026?"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "conversation_id": "uuid",
  "role": "assistant",
  "content": "Here are the top 5 products by revenue in Q1 2026:",
  "sql_query": "SELECT product_name, SUM(revenue) as total_revenue FROM sales WHERE date >= '2026-01-01' AND date < '2026-04-01' GROUP BY product_name ORDER BY total_revenue DESC LIMIT 5",
  "result_data": [
    {
      "product_name": "Product A",
      "total_revenue": 150000
    },
    {
      "product_name": "Product B",
      "total_revenue": 120000
    }
  ],
  "execution_time": 0.25,
  "created_at": "2026-02-15T10:00:00Z"
}
```

### 4.2 List Messages
**GET** `/conversations/{conversation_id}/messages`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 50)

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": "uuid",
      "role": "user",
      "content": "What are the top 5 products?",
      "created_at": "2026-02-15T10:00:00Z"
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "Here are the top 5 products...",
      "sql_query": "SELECT...",
      "result_data": [...],
      "created_at": "2026-02-15T10:00:05Z"
    }
  ],
  "total": 2,
  "page": 1,
  "limit": 50
}
```

---

## 5. Training APIs (Vanna AI)

### 5.1 Train with DDL
**POST** `/training/ddl`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "connection_id": "uuid",
  "ddl": "CREATE TABLE products (id SERIAL PRIMARY KEY, name VARCHAR(255), price DECIMAL(10,2));"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "DDL trained successfully"
}
```

### 5.2 Train with Documentation
**POST** `/training/documentation`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "connection_id": "uuid",
  "documentation": "The products table contains all product information. The price column is in USD."
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Documentation trained successfully"
}
```

### 5.3 Train with SQL Question-Answer
**POST** `/training/sql`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "connection_id": "uuid",
  "question": "What are the top 5 products by revenue?",
  "sql": "SELECT product_name, SUM(revenue) as total FROM sales GROUP BY product_name ORDER BY total DESC LIMIT 5"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "SQL trained successfully"
}
```

### 5.4 Auto-Train from Database
**POST** `/training/auto`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "connection_id": "uuid",
  "include_information_schema": true
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Auto-training completed",
  "tables_trained": 15,
  "columns_trained": 120
}
```

### 5.5 Get Training Status
**GET** `/training/status/{connection_id}`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `200 OK`
```json
{
  "connection_id": "uuid",
  "is_trained": true,
  "training_data_count": 45,
  "last_trained_at": "2026-02-15T09:00:00Z"
}
```

---

## 6. Query/Analysis APIs

### 6.1 Generate SQL from Question
**POST** `/query/generate`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "connection_id": "uuid",
  "question": "What are the top 5 customers by total orders?"
}
```

**Response:** `200 OK`
```json
{
  "sql": "SELECT customer_name, COUNT(*) as order_count FROM orders JOIN customers ON orders.customer_id = customers.id GROUP BY customer_name ORDER BY order_count DESC LIMIT 5",
  "confidence": 0.95
}
```

### 6.2 Execute SQL Query
**POST** `/query/execute`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "connection_id": "uuid",
  "sql": "SELECT * FROM products LIMIT 10"
}
```

**Response:** `200 OK`
```json
{
  "columns": ["id", "name", "price"],
  "data": [
    [1, "Product A", 99.99],
    [2, "Product B", 149.99]
  ],
  "row_count": 2,
  "execution_time": 0.15
}
```

### 6.3 Ask Question (Combined Generate + Execute)
**POST** `/query/ask`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "connection_id": "uuid",
  "question": "Show me revenue by product category",
  "save_to_conversation_id": "uuid"
}
```

**Response:** `200 OK`
```json
{
  "question": "Show me revenue by product category",
  "sql": "SELECT category, SUM(revenue) as total FROM products JOIN sales ON products.id = sales.product_id GROUP BY category",
  "result": {
    "columns": ["category", "total"],
    "data": [
      ["Electronics", 500000],
      ["Clothing", 300000]
    ],
    "row_count": 2
  },
  "execution_time": 0.2,
  "explanation": "This query joins products and sales tables, groups by category and sums the revenue."
}
```

---

## 7. Search/Vector APIs

### 7.1 Search Similar Questions
**POST** `/search/questions`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "connection_id": "uuid",
  "query": "revenue by product",
  "limit": 5
}
```

**Response:** `200 OK`
```json
{
  "items": [
    {
      "question": "What is the total revenue by product?",
      "sql": "SELECT product_name, SUM(revenue) FROM...",
      "similarity": 0.92
    },
    {
      "question": "Show me top products by revenue",
      "sql": "SELECT product_name, SUM(revenue) FROM...",
      "similarity": 0.88
    }
  ]
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid input data"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid or expired token"
}
```

### 403 Forbidden
```json
{
  "detail": "Account is inactive"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error",
  "error": "Error message"
}
```

---

## Authentication Flow

1. User registers or signs in → receives `access_token` and `refresh_token`
2. Include `access_token` in all subsequent requests:
   ```
   Authorization: Bearer {access_token}
   ```
3. When `access_token` expires, use `refresh_token` to get new tokens
4. Use OAuth2 flow (`/auth/token`) for Swagger UI testing

---

## Typical Usage Flow

1. **Setup:**
   - Register/Sign in → Get token
   - Create database connection → Get `connection_id`
   - Train Vanna with schema/examples

2. **Conversation:**
   - Create conversation → Get `conversation_id`
   - Send questions → Get SQL + results
   - View conversation history

3. **Direct Query:**
   - Use `/query/ask` directly without saving to conversation
   - Or use `/query/generate` + `/query/execute` separately
