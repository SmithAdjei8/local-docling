%% Chain workflow (linear)
graph TD
    A[User Query] --> B[Enrich Query]
    B --> C[Retrieve Data from DB]
    C --> D[Summarize & Format Answer]
    D --> E[Output to Staff]

%% LangGraph workflow (graph-based with branching)
graph TD
    A[User Query] --> B[Enrich Query]
    B --> C1{Query Type?}
    C1 -->|Resident| D1[Retrieve Resident Records]
    C1 -->|Staff| D2[Retrieve Staff Shifts]
    C1 -->|Compliance| D3[Retrieve Compliance Logs]
    D1 --> E[Summarize & Format Answer]
    D2 --> E
    D3 --> E
    E --> F[Output to Staff]