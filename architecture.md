```mermaid
graph TD
    A[Streamlit UI] --> B[generate_rewritten_line]
    B --> C[OpenAI Response API调用]
    C --> D[responses.create]
    D --> E[工具函数调用]
    E --> F[处理结果]
    F --> B
    B --> A
```

[[Architecture Diagram]]