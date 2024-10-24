# ER図
---

```mermaid　
erDiagram
  users {
    bigint id PK
    string name "ユーザー名"
    timestamp created_at
    timestamp deleted_at
  }
```