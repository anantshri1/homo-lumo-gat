# MolPredict — GATv2 HOMO-LUMO Gap Inference API

`homo-lumo-gat` is a graph neural network trained on the QM9 dataset that predicts the HOMO-LUMO gap of small organic molecules from their SMILES string representation. The model is served via a production-grade FastAPI inference API, containerized with Docker, and exposed as a callable tool via an MCP server so that LLM agents can query it directly. The live API is deployed at https://homo-lumo-gat.fly.dev.

> **What is the HOMO-LUMO gap?** 
The HOMO-LUMO gap is the energy difference between a molecule's Highest Occupied Molecular Orbital and its Lowest Unoccupied Molecular Orbital, measured in electron volts (eV). It is one of the most important descriptors in computational chemistry — a large gap indicates a chemically stable, electrically insulating molecule, while a small gap indicates reactivity and potential conductivity. Predicting this gap cheaply (without running expensive quantum chemistry simulations) is a core problem in drug discovery and materials science.
<!---
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
--->
---
## A Gentle Guide to Graph Neural Networks

**Graph Neural Networks** (GNNs) are a part of an emerging research paradigm called **Geometric Deep Learning** -- devising neural network architectures that respect the invariances and symmetries in data. 

Molecules are naturally represented as graphs: atoms are nodes and bonds are edges. A Graph Neural Network exploits this structure directly. Rather than flattening a molecule into a fixed-size fingerprint (losing structural information), a GNN operates on the graph itself through a process called *message passing*. In each *message passing* layer, every atom aggregates feature information from its bonded neighbors, updates its own representation, and passes the result to the next layer. After several rounds of this, each atom's embedding encodes not just its own identity but its full local chemical context. A global pooling operation then collapses all atom embeddings into a single molecule-level vector, which a prediction head maps to the target property.

<img src="https://github.com/chaitjo/dump/raw/main/molproppred.png">

One famous example of GNNs being used in molecular property prediction is in the world of **antibiotic discovery**, an area with a potentially massive impact on humanity and infamously little innovation. A GNN trained to predict how much a molecule would inhibit a bacteria was able identify the previously overlooked compound [**Halicin**](https://www.wikiwand.com/en/Halicin) during virtual screening. Not only did halicin show powerful results during *in vitro* (in cell) testing but it also had a completely novel mechanism of action that no bacteria has developed resistance to (yet).


### The QM9 Dataset

QM9 (Quantum Mechanics dataset 9) is a dataset consisting of about **130,000 small molecules** with 19 regression targets. Since being used by [MoleculeNet](https://arxiv.org/abs/1703.00564), it has become a popular dataset to benchmark new architectures for molecular property prediction.

Specifically, we will be predicting the [HOMO-LUMO gap](https://en.wikipedia.org/wiki/HOMO_and_LUMO) of drug-like molecules. 

### Message Passing Neural Networks in PyTorch Geometric

GNNs come in three different flavors, as shown below:

<img src="https://github.com/chaitjo/dump/raw/main/gnn-layers.png">


Firstly, we formalise our molecular property prediction pipeline. 

**Graph** 

Consider a molecular graph $\mathcal{G} = \left( \mathcal{V}, \mathcal{E} \right)$, where $\mathcal{V}$ is a set of $n$ nodes, and $\mathcal{E}$ is a set of edges associated with the nodes. For each node $i \in \mathcal{V}$, we are given a $d_n$-dimensional initial feature vector $h_i \in \mathbb{R}^{d_n}$.
For each edge $(i, j) \in \mathcal{E}$, we are given a $d_e$-dimensional initial feature vector $e_{ij} \in \mathbb{R}^{d_e}$. For QM9 graphs, $d_n = 11, d_e = 4$.

**Label/target**

Associated with each graph $\mathcal{G}$ is a scalar target or label $y \in \mathbb{R}^{1}$, which we would like to predict. 

We will design a Message Passing Neural Network for graph property prediction to do this. Our MPNN will consist of several layers of message passing, followed by a global pooling and prediction head.

**MPNN Layer**

The Message Passing operation iteratively updates node features $h_i^{\ell} \in \mathbb{R}^d$ from layer $\ell$ to layer $\ell+1$ via the following equation:
$h_i^{(\ell+1)}=\phi\ \left(h_i^{(\ell)},\bigoplus_{j\in\mathcal{N}(i)}\psi\\left(h_i^{(\ell)},h_j^{(\ell)},e_{ij}\right)\right)$
where $\psi, \phi$ are Multi-Layer Perceptrons (MLPs), and $\bigoplus$ is a permutation-invariant local neighborhood aggregation function such as summation, maximization, or averaging.

Let us break down the MPNN layer into three pedagogical steps:
- **Step (1): Message.** For each pair of linked nodes $i, j$, the network first computes a message $m_{ij} =  \psi \left( h_i^{\ell}, h_j^{\ell}, e_{ij} \right)$. The MLP $\psi: \mathbb{R}^{2d + d_e} → \mathbb{R}^d$ takes as input the concatenation of the feature vectors from the source node, destination node, and edge.
    > Note that for the first layer $\ell=0$, $h_i^{\ell=0} = W_{in} \left( h_i \right)$, where $W_{in} \in \mathbb{R}^{d_n}  \rightarrow \mathbb{R}^{d}$ is a simple linear projection (`torch.nn.Linear`) for the initial node features to hidden dimension $d$.
- **Step (2): Aggregate.** At each node $i$, the incoming messages from all its neighbors are then aggregated as $m_{i} = \bigoplus_{j\in\mathcal{N}(i)}\left(m_{ij}\right)$, where $\bigoplus$ is a permutation-invariant function. We will use summation, i.e. $\oplus_{{j \in \mathcal{N}(i)}} = \sum_{{j \in \mathcal{N}(i)}}$.
- **Step (3): Update.** Finally, the network updates the node feature vector $h_i^{\ell+1} = \phi \left( h_i^{\ell}, m_i \right)$, by concatenating the aggregated message $m_i$ and the previous node feature vector $h_i^{\ell}$, and passing them through an MLP $\phi: \mathbb{R}^{2d} → \mathbb{R}^{d}$.

**Global Pooling and Prediction Head**

After $L$ layers of message passing, we obtain the final node features $h_i^{\ell=L}$. As we have a single target $y$ per graph, we must pool all node features into a single graph feature or graph embedding $h_G \in \mathbb{R}^d$ via another permutation-invariant function $R$, sometimes called the 'readout' function, as follows:
$h_G = R_{i \in \mathcal{V}} \left( h_i^{\ell=L} \right)$. We will use global average pooling over all node features, i.e. $h_G = \frac{1}{|V|}\sum_{i\in V}h_i^{\ell = L}$. The graph embedding $h_G$ is passed through a linear prediction head $W_{pred} \in \mathbb{R}^{d} \rightarrow \mathbb{R}^1$ to obtain the overall prediction $\hat y \in \mathbb{R}^1$: $\hat y = W_{pred} \left (h_G\right)$.

**Loss Function**

Our MPNN graph property prediction model can be trained end-to-end via minimizing the standard mean absolute error loss for regression:
$\mathcal{L}_{MAE} = \lVert y - \hat y \rVert$.

---

## Architecture
The model is a `GATv2` (Graph Attention Network v2) with three message-passing layers, residual connections, and a two-layer MLP prediction head.
Input features come directly from PyTorch Geometric's QM9 dataset class. Node features are 11-dimensional per atom (atomic number, degree, formal charge, hybridization, aromaticity, hydrogen count, ring membership, and others). Edge features are 4-dimensional per bond (bond type, conjugation, ring membership, stereo). These are the raw graph features QM9 provides — no 3D coordinates are used.

`GATv2` was chosen over the original `GAT` because it fixes a fundamental expressiveness limitation. In `GAT`, attention weights are computed before node features are combined, making the attention query-independent — the same pair of nodes always produces the same attention weight regardless of context. `GATv2` (Brody et al., 2022) fixes this by computing attention after concatenation and transformation, making it a strictly more expressive attention mechanism. In practice, this means the model can learn more context-sensitive atom interactions.

Each GATv2Conv layer uses 4 attention heads averaged (`concat=False`) to keep the hidden dimension at 256 across all layers. Residual connections are added at every layer with a linear projection to handle the dimension mismatch at layer 1 (11-dim input to 256-dim output). Without residuals, gradient flow degrades significantly in deeper GNNs.

After the third message-passing layer, the atom embeddings are reduced to a single molecule embedding via global mean pooling and global max pooling concatenated together (512-dim total). Mean pooling captures average atom character; max pooling captures the most salient atom in the molecule. Using both consistently outperforms either alone.
The MLP head maps `512 → 256 → 128 → 1` with ReLU activations and dropout between layers.

**Pipeline summary**

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

### Training

The model was trained on QM9 (approximately 130,000 molecules) with a standard 110k/10k/10k train/val/test split matching the split used in published baselines. Training used the `Adam` optimizer with an initial learning rate of `1e-3` and a `ReduceLROnPlateau` scheduler (`patience=10, factor=0.5`). Loss function is L1 (MAE), which is the standard benchmark metric for QM9 and is less sensitive to outlier molecules than MSE. Training ran for 200 epochs on a T4 GPU via Google Colab, with checkpointing to Google Drive after every 10 epochs and on every validation improvement.

**Results**
```
Test MAE: 0.1712 eV
Best validation MAE: 0.1675 eV (epoch 120, converged)
```
Published baselines on QM9 HOMO-LUMO gap:
```
SchNet: 0.0670 eV
DimeNet: 0.0469 eV
```
The gap to SchNet and DimeNet is expected and explained by a fundamental difference in input information. Both SchNet and DimeNet use 3D atomic coordinates — they know the actual geometry of the molecule in space, including bond lengths, bond angles, and dihedral angles. This project uses only 2D graph topology: which atoms are bonded to which, with no geometric information. A 2D-topology-only model cannot learn the same geometric features that drive quantum mechanical properties, so a higher MAE is the correct outcome, not a failure of the architecture. The more honest comparison is with other 2D-topology models on QM9, where this result is competitive.

### Inference and the RDKit graph builder

At inference time, a user submits an arbitrary SMILES string. The serving layer uses RDKit to parse it into a molecular graph and constructs a PyTorch Geometric Data object with the same feature specification as QM9. A critical implementation detail: QM9 was built by PyG with `sanitize=False`, which means hybridization features are always zero in the training data. The RDKit graph builder in this project deliberately replicates this behavior. Using `sanitize=True` at inference time would produce non-zero hybridization features, putting the input out of distribution relative to what the model was trained on. Correctness of the chemistry matters less here than consistency with the training distribution.

The `/predict` endpoint also supports MC dropout uncertainty estimation: `dropout` is left active at inference time and the forward pass is run `N` times. The mean of the `N` predictions is the reported gap value and the standard deviation is the reported uncertainty. This gives a cheap but meaningful confidence interval without requiring an ensemble of models.

---
## Serving Infrastructure

### Architecture
The inference service runs as two containers orchestrated by Docker Compose:

- **FastAPI** (`:8000`) — loads the GATv2 model at startup and exposes three endpoints:
  - `POST /predict` — single SMILES → gap (eV) + uncertainty via MC dropout
  - `POST /batch` — list of SMILES → predictions
  - `GET /health` — model status, device, version
- **MCP Server** (`:8001`) — wraps the FastAPI endpoints as callable tools (`predict_gap`, `batch_predict`) so any MCP-compatible LLM agent can query the model directly

CURL commands have the general form:
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "C1=CC=CC=C1"}'
```
which returns:
```
{"smiles":"C1=CC=CC=C1","gap_ev":8.5004,"uncertainty_ev":0.1698}
```
and:
```
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"smiles_list": ["c1ccccc1", "C", "CC(=O)O"]}'
```


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

### Calling the API

**As a developer**
Send a POST request to the deployed endpoint directly:

```python
import requests

response = requests.post(
    "https://homo-lumo-gat.fly.dev/predict",
    json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"}  # aspirin
)
print(response.json())
# {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "gap_ev": 6.21, "uncertainty_ev": 0.14}
```

**As an LLM agent**
Run the MCP server locally (requires Docker Compose):

```bash
docker compose up
```

Then point any MCP-compatible agent (Claude Desktop, LangChain, etc.) to `http://localhost:8001/sse`. 
The agent will discover two tools automatically — `predict_gap` (single molecule) and 
`batch_predict` (list of molecules) — and can call them in response to natural language queries 
like "What is the HOMO-LUMO gap of caffeine?".

The MCP server is not deployed publicly because it is a local bridge between an agent and the 
deployed API — the agent runs on your machine, the model runs on Fly.io.

---
# References
* **MoleculeNet: A Benchmark for Molecular Machine Learning**, Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, Vijay Pande. (2017). [arXiv:1703.00564](https://arxiv.org/abs/1703.00564)
* **Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures**, Shuo Zhang, Yang Liu, Lei Xie. (2020). [arXiv:2011.07457](https://arxiv.org/abs/2011.07457).
* **How Attentive are Graph Attention Networks?**, Shaked Brody, Uri Alon, Eran Yahav. (2022). [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)


