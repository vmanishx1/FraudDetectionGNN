# Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │Dashboard │Transact. │ Graph    │Prediction│ Analytics│   │
│  │   Page   │   Page   │   Page   │   Page   │   Page   │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
│                   │              │                           │
│              ┌────┴──────────────┴────┐                      │
│              │   UI Components        │                      │
│              └───────────┬────────────┘                      │
└──────────────────────────┼──────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────┼──────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Auth    │  │   API    │  │ Services │  │  Models  │   │
│  │  Router  │  │  Routes  │  │(Business │  │ (Pydantic│   │
│  │          │  │          │  │  Logic)  │  │ Schemas)  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                       │                                      │
│              ┌────────┴────────┐                             │
│              │   SQLAlchemy    │                             │
│              │   Database      │                             │
│              └────────┬────────┘                             │
└───────────────────────┼─────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│            ML Pipeline (PyTorch Geometric)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Preprocess│  │  Graph   │  │  GNN     │  │Evaluate  │   │
│  │  Data    │  │  Builder │  │  Models  │  │ & Explain│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
│                        │                                      │
│                   ┌────┴────┐                                │
│                   │ Saved   │                                │
│                   │ Models  │                                │
│                   └─────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Data Ingestion**: IEEE-CIS dataset loaded and preprocessed
2. **Feature Engineering**: Missing values handled, categorical encoding, scaling
3. **Graph Construction**: Customers → Nodes, Transactions → Edges
4. **Model Training**: GCN, GraphSAGE, GAT trained and compared
5. **Inference**: New transactions classified as fraud/legitimate
6. **Explainability**: GNNExplainer identifies important features and nodes
7. **Dashboard**: Results displayed via interactive Streamlit UI

## Component Diagram

### Frontend (Streamlit)
- `app/main.py` - Entry point, navigation, auth
- `app/pages/` - Individual dashboard pages
- `app/components/` - Reusable UI components
- `app/styles/` - Theming and CSS

### Backend (FastAPI)
- `backend/main.py` - FastAPI application
- `backend/api/routes.py` - API endpoints
- `backend/auth/jwt_handler.py` - JWT authentication
- `backend/database/` - SQLAlchemy models and connection
- `backend/services/` - Business logic

### ML Pipeline
- `ml/preprocessing/` - Data preprocessing
- `ml/graph_builder/` - Graph construction
- `ml/models/` - GNN model architectures
- `ml/training/` - Training loop
- `ml/evaluation/` - Metrics computation
- `ml/explainability/` - Model explanations
- `ml/inference/` - Prediction engine
