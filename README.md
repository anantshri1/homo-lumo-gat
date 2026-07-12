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

---
## A Gentle Guide to Graph Neural Networks

**Graph Neural Networks** (GNNs) are a part of an emerging research paradigm called **Geometric Deep Learning** -- devising neural network architectures that respect the invariances and symmetries in data. 

Molecules are a great example of an object from nature that can easily be represented as a graph of atoms (nodes) connected by bonds (edges). 
A popular application of GNNs in chemistry is the task of **Molecular Property Prediction**. The goal is to train a GNN model from historical experimental data that can predict useful properties of drug-like molecules. The model's predictions can then be used to guide the drug design process.

<img src="https://github.com/chaitjo/dump/raw/main/molproppred.png">

One famous example of GNNs being used in molecular property prediction is in the world of **antibiotic discovery**, an area with a potentially massive impact on humanity and infamously little innovation. A GNN trained to predict how much a molecule would inhibit a bacteria was able identify the previously overlooked compound [**Halicin**](https://www.wikiwand.com/en/Halicin) during virtual screening. Not only did halicin show powerful results during *in vitro* (in cell) testing but it also had a completely novel mechanism of action that no bacteria has developed resistance to (yet).


### The QM9 Dataset

QM9 (Quantum Mechanics dataset 9) is a dataset consisting of about **130,000 small molecules** with 19 regression targets. Since being used by [MoleculeNet](https://arxiv.org/abs/1703.00564), it has become a popular dataset to benchmark new architectures for molecular property prediction.

Specifically, we will be predicting the [HOMO-LUMO gap](https://en.wikipedia.org/wiki/HOMO_and_LUMO) of drug-like molecules. 

### Message Passing Neural Networks in PyTorch Geometric

As a gentle introduction to PyTorch Geometric, we will walk you through the first steps of developing a GNN in the **Message Passing** flavor.

<img src="https://github.com/chaitjo/dump/raw/main/gnn-layers.png">

**Formalism**

Firstly, let us formalise our molecular property prediction pipeline. 

**Graph** 

Consider a molecular graph $\mathcal{G} = \left( \mathcal{V}, \mathcal{E} \right)$, where $\mathcal{V}$ is a set of $n$ nodes, and $\mathcal{E}$ is a set of edges associated with the nodes. For each node $i \in \mathcal{V}$, we are given a $d_n$-dimensional initial feature vector $h_i \in \mathbb{R}^{d_n}$.
For each edge $(i, j) \in \mathcal{E}$, we are given a $d_e$-dimensional initial feature vector $e_{ij} \in \mathbb{R}^{d_e}$. For QM9 graphs, $d_n = 11, d_e = 4$.

**Label/target**

Associated with each graph $\mathcal{G}$ is a scalar target or label $y \in \mathbb{R}^{1}$, which we would like to predict. 

We will design a Message Passing Neural Network for graph property prediction to do this. Our MPNN will consist of several layers of message passing, followed by a global pooling and prediction head.

**MPNN Layer**

The Message Passing operation iteratively updates node features $h_i^{\ell} \in \mathbb{R}^d$ from layer $\ell$ to layer $\ell+1$ via the following equation:
$h_i^{\ell+1}$ = $\phi$ $\Bigg( h_i^{\ell}$, $\oplus_{j \in \mathcal{N}_i}$ $\Big( \psi \left( h_i^{\ell}, h_j^{\ell}, e_{ij} \right) \Big) \Bigg),$
where $\psi, \phi$ are Multi-Layer Perceptrons (MLPs), and $\oplus$ is a permutation-invariant local neighborhood aggregation function such as summation, maximization, or averaging.

Let us break down the MPNN layer into three pedagogical steps:
- **Step (1): Message.** For each pair of linked nodes $i, j$, the network first computes a message $m_{ij} =  \psi \left( h_i^{\ell}, h_j^{\ell}, e_{ij} \right)$. The MLP $\psi: \mathbb{R}^{2d + d_e} → \mathbb{R}^d$ takes as input the concatenation of the feature vectors from the source node, destination node, and edge.
    > Note that for the first layer $\ell=0$, $h_i^{\ell=0} = W_{in} \left( h_i \right)$, where $W_{in} \in \mathbb{R}^{d_n}  \rightarrow \mathbb{R}^{d}$ is a simple linear projection (`torch.nn.Linear`) for the initial node features to hidden dimension $d$.
- **Step (2): Aggregate.** At each node $i$, the incoming messages from all its neighbors are then aggregated as $m_{i} = \oplus_{{j \in \mathcal{N}_i}} \left( m_{ij} \right)$, where $\oplus$ is a permutation-invariant function. We will use summation, i.e. $\oplus_{{j \in \mathcal{N}_i}} = \sum_{{j \in \mathcal{N}_i}}$.
- **Step (3): Update.** Finally, the network updates the node feature vector $h_i^{\ell+1} = \phi \left( h_i^{\ell}, m_i \right)$, by concatenating the aggregated message $m_i$ and the previous node feature vector $h_i^{\ell}$, and passing them through an MLP $\phi: \mathbb{R}^{2d} → \mathbb{R}^{d}$.

**Global Pooling and Prediction Head**

After $L$ layers of message passing, we obtain the final node features $h_i^{\ell=L}$. As we have a single target $y$ per graph, we must pool all node features into a single graph feature or graph embedding $h_G \in \mathbb{R}^d$ via another permutation-invariant function $R$, sometimes called the 'readout' function, as follows:
$h_G = R_{i \in \mathcal{V}} \left( h_i^{\ell=L} \right)$. 
$$ 
We will use global average pooling over all node features, i.e.
$$
h_G = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} h_i^{\ell=L}. 
$$

The graph embedding $h_G$ is passed through a linear prediction head $W_{pred} \in \mathbb{R}^{d} \rightarrow \mathbb{R}^1$ to obtain the overall prediction $\hat y \in \mathbb{R}^1$:
$$
\hat y = W_{pred} \left( h_G \right).
$$ 

**Loss Function**
Our MPNN graph property prediction model can be trained end-to-end via minimizing the standard mean-squared error loss for regression:
$\mathcal{L}_{MSE} = \lVert y - \hat y \rVert^2_2$.

---
## Serving Infrastructure

### Architecture
The inference service runs as two containers orchestrated by Docker Compose:

- **FastAPI** (`:8000`) — loads the GATv2 model at startup and exposes three endpoints:
  - `POST /predict` — single SMILES → gap (eV) + uncertainty via MC dropout
  - `POST /batch` — list of SMILES → predictions
  - `GET /health` — model status, device, version
- **MCP Server** (`:8001`) — wraps the FastAPI endpoints as callable tools (`predict_gap`, `batch_predict`) so any MCP-compatible LLM agent can query the model directly

### Docker

Two Dockerfiles serve different targets:

- `Dockerfile` — CUDA-enabled build using `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` as base. For GPU inference on any machine with an NVIDIA driver. PyTorch and cuDNN come pre-installed in the base image; no manual CUDA wheel management needed.
- `Dockerfile.flyio` — CPU-only build using `python:3.10-slim`. Installs PyTorch from the `+cpu` wheel index. Used for cloud deployment where no GPU is available. Image size: ~700MB vs ~3GB for the CUDA variant.

Key Docker decisions:
- Dependencies are installed before code is copied, so the expensive pip layers (PyTorch, PyG, RDKit) stay cached across code changes
- `torch-scatter` is installed from PyG's own wheel index with an explicit `torch-2.3.0+cpu` tag — it must match the PyTorch version exactly or import will silently fail
- `best_model.pt` is force-added to git (`.gitignore` has `*.pt`) and explicitly exempted in `.dockerignore` with `!checkpoints/best_model.pt`. Without the exemption the build context excludes the weights and the container crashes on startup.
- The MCP server uses a plain `python:3.10-slim` base — no ML dependencies, just `httpx` and `mcp`. It proxies requests to FastAPI rather than loading the model itself.

### MCP Integration
The MCP server speaks SSE (Server-Sent Events) transport over HTTP. This makes it network-addressable — any MCP-compatible LLM agent can connect to it without needing to spawn it as a local process (unlike stdio transport). Inside Docker Compose, the MCP server reaches FastAPI via the service hostname `http://fastapi:8000`, which Docker's internal DNS resolves automatically.

### Deployment
Deployed to Fly.io free tier at `https://homo-lumo-gat.fly.dev`.

- Region: Frankfurt (fra)
- Machine: shared-cpu-1x, 1GB RAM
- Auto-stop enabled (machine sleeps when idle, wakes on first request — expect ~2s cold start)
- Build handled remotely by Fly.io's Depot builder

Interactive API docs available at `https://homo-lumo-gat.fly.dev/docs`.

### What's not here (and why)
- Intermediate checkpoints (`checkpoint_epoch_*.pt`) are local-only — not in git, not in the container. Only `best_model.pt` ships.
- The MCP server is not deployed — it's a local/agent tool. The FastAPI endpoint is the public artifact.
- Model weights are in git for simplicity (16MB). Production pattern would be to pull from Hugging Face Hub or S3 at container startup.


---
# References
* **MoleculeNet: A Benchmark for Molecular Machine Learning**, Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, Vijay Pande. (2017). [arXiv:1703.00564](https://arxiv.org/abs/1703.00564)
* **Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures**, Shuo Zhang, Yang Liu, Lei Xie. (2020). [arXiv:2011.07457](https://arxiv.org/abs/2011.07457).


