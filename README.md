# MolPredict — GATv2 HOMO-LUMO Gap Inference API

**Stack**:
`PyTorch, PyTorch Geometric, RDKit, FastAPI, Docker, MCP, Uvicorn`
 
**Architecture: GATv2 Message Passing Network**
```
SMILES string
    ↓ RDKit
Molecular Graph (nodes=atoms, edges=bonds)
    ↓
Atom feature embedding (atomic number, valence, 
    hybridization, aromaticity, charge)
    ↓
3× GATv2Conv layers (with residual connections)
    ↓
Global mean + max pooling (concatenated)
    ↓
MLP head (256 → 128 → 1)
    ↓
HOMO-LUMO gap (eV)
```
 
Node features (8-dim): atomic number, degree, formal charge, hybridization, aromaticity, H count, ring membership, chirality
Edge features (4-dim): bond type, conjugation, ring membership, stereo
 
**Service Architecture**
```
┌─────────────────────────────────────┐
│           Docker Compose            │
│                                     │
│  ┌──────────────┐  ┌─────────────┐  │
│  │  FastAPI     │  │  MCP Server │  │
│  │  :8000       │◄─►  :8001      │  │
│  │              │  │             │  │
│  │ POST /predict│  │ tool:       │  │
│  │ GET /health  │  │ predict_gap │  │
│  │ GET /batch   │  │ batch_      │  │
│  └──────┬───────┘  │ predict     │  │
│         │          └─────────────┘  │
│  ┌──────▼───────┐                   │
│  │ Model Service│                   │
│  │ (loaded once │                   │
│  │  at startup) │                   │
│  └──────────────┘                   │
└─────────────────────────────────────┘
```
 
Three endpoints on FastAPI:
 
* `POST /predict `— single SMILES → gap (eV) + confidence interval via MC dropout
* `POST /batch` — list of SMILES → predictions (async, job ID pattern)
* `GET /health` — model loaded, device, version
 
MCP exposes `predict_gap` and `batch_predict` as callable tools — so an LLM agent can query your model directly. 
 
**Timeline (realistic, public-ready)**
Phase 1: Data + model
 
* QM9 loading with PyG's built-in dataset class
* Feature engineering (RDKit graph constructor)
* GATv2 architecture + training loop with CUDA-aware device handling
* Baseline: compare against SchNet published numbers on QM9
 
Phase 2: Serving
 
* FastAPI service with proper async batch endpoint
* MC dropout for uncertainty estimates (one genuinely useful addition)
* Pydantic request/response models
* Unit tests for the graph constructor
 
Phase 3: Docker + MCP + deployment
 
* Docker Compose setup (API + MCP server)
* MCP tool server wrapping the FastAPI endpoints
* Deploy API to Fly.io free tier (they support Docker natively, no credit card for small instances)
* GitHub Actions CI for build + test on push
 
Phase 4: Polish + README
 
* Benchmark table (your MAE vs SchNet vs DimeNet published numbers)
* Colab notebook for interactive inference (links to your deployed endpoint)
* Clean README with architecture diagram
