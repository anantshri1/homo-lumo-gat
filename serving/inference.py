import torch
from torch_geometric.data import Data, Batch
from serving.model import GATv2Model

def load_model(checkpoint_path: str, device: str = 'cpu') -> GATv2Model:
    checkpoint = torch.load(checkpoint_path, map_location=device)       # load the tensors onto this device instead
    
    config = checkpoint['config']
    model = GATv2Model(
        in_channels=11,
        edge_dim=4,
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()        # sets dropout and batchnorm to inference mode
    return model

def predict(model: GATv2Model, data: Data, device: str = 'cpu') -> float:
    model.eval()
    batch = Batch.from_data_list([data]).to(device)     # batch is a single graph in a batch format, which is what the model expects
    
    with torch.no_grad():       # tells PyTorch not to build a computation graph for backpropagation
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
    return out.item()

"""
* Why `model.train()` for inference? Normally `train()` means "we're about to do backprop". Here we're hijacking it just to keep dropout active. With `model.eval()`, dropout is disabled and every forward pass gives the same result. With `model.train()`, dropout randomly zeros different neurons each pass, giving slightly different predictions each time.
* What does the std tell us? High std = the model is uncertain about this molecule, probably because it's unlike anything in QM9. Low std = the model is confident. This is called MC (Monte Carlo) dropout — a cheap approximation to Bayesian uncertainty.
"""

def predict_with_uncertainty(
    model: GATv2Model, 
    data: Data, 
    device: str = 'cpu',
    n_samples: int = 20
) -> tuple[float, float]:
    
    model.train()  # activates dropout
    batch = Batch.from_data_list([data]).to(device)
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.append(out.item())
    
    model.eval()  # restore eval mode
    
    mean = sum(predictions) / len(predictions)
    variance = sum((p - mean) ** 2 for p in predictions) / len(predictions)
    std = variance ** 0.5
    
    return mean, std