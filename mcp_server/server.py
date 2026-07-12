# Block 1: Imports and configuration
import httpx
from mcp.server.fastmcp import FastMCP

FASTAPI_URL = "http://fastapi:8000"
mcp = FastMCP("molpredict", host="0.0.0.0", port=8001)

# Block 2: predict_gap tool
@mcp.tool()
async def predict_gap(smiles: str) -> dict:
    """Predict the HOMO-LUMO gap in eV for a molecule given its SMILES string."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{FASTAPI_URL}/predict", json={"smiles": smiles})
        response.raise_for_status()
        return response.json()

# Block 3: batch_predict tool
@mcp.tool()
async def batch_predict(smiles_list: list[str]) -> dict:
    """Predict HOMO-LUMO gaps for multiple molecules given a list of SMILES strings."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{FASTAPI_URL}/batch", json={"smiles_list": smiles_list})
        response.raise_for_status()
        return response.json()

# Block 4: entrypoint
if __name__ == "__main__":
    mcp.run(transport="sse")