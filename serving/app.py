from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import os

from serving.graph_builder import smiles_to_graph
from serving.inference import load_model, predict, predict_with_uncertainty

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
DEVICE = "cpu"
VERSION = "0.1.0"

# Global model variable — loaded once at startup
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once
    global model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = load_model(CHECKPOINT_PATH, DEVICE)
    print(f"Model loaded on {DEVICE}")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")

app = FastAPI(
    title="MolPredict",
    description="GATv2 HOMO-LUMO gap prediction API",
    version=VERSION,
    lifespan=lifespan
)

# -------------- Pydantic contracts ------------------------

class PredictRequest(BaseModel):
    smiles: str

class PredictResponse(BaseModel):
    smiles: str
    gap_ev: float
    uncertainty_ev: float

class BatchRequest(BaseModel):
    smiles_list: list[str]

class BatchResponse(BaseModel):
    predictions: list[PredictResponse]

class HealthResponse(BaseModel):
    status: str
    device: str
    version: str


# ------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model is not None else "model not loaded",
        device=DEVICE,
        version=VERSION
    )

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    try:
        data = smiles_to_graph(request.smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    mean, std = predict_with_uncertainty(model, data, DEVICE)
    
    return PredictResponse(
        smiles=request.smiles,
        gap_ev=round(mean, 4),
        uncertainty_ev=round(std, 4)
    )

@app.post("/batch", response_model=BatchResponse)
async def batch_endpoint(request: BatchRequest):
    results = []
    errors = []
    
    for smiles in request.smiles_list:
        try:
            data = smiles_to_graph(smiles)
            mean, std = predict_with_uncertainty(model, data, DEVICE)
            results.append(PredictResponse(
                smiles=smiles,
                gap_ev=round(mean, 4),
                uncertainty_ev=round(std, 4)
            ))
        except ValueError as e:
            errors.append(f"{smiles}: {str(e)}")
    
    if errors:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process: {errors}"
        )
    
    return BatchResponse(predictions=results)