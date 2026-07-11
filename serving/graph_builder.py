from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Must match QM9 exactly — order matters
ATOM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
BOND_TYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

"""
* Why `AddHs()`?: QM9 was built from SDF files that already had explicit hydrogens — every H atom appears as its own node in the graph. If we don't call `AddHs()`, benzene gives us 6 nodes (carbons only). With it, we get 12 nodes (6C + 6H). Our trained model learned on graphs with explicit H, so we must match that.
* Why check `mol is None`? RDKit returns `None` silently for invalid SMILES instead of raising an exception. Always check — this will be your first line of defense against bad user input in the API.
"""

def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    
    # Add explicit hydrogens — QM9 includes H atoms as nodes
    mol = Chem.AddHs(mol)

    # --- Node features --- direct translation of QM9 node features
    type_idx = []
    atomic_number = []
    aromatic = []
    sp, sp2, sp3 = [], [], []

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ATOM_TYPES:
            raise ValueError(f"Atom type '{symbol}' not in QM9 vocabulary: {list(ATOM_TYPES.keys())}")
        
        type_idx.append(ATOM_TYPES[symbol])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        
        hyb = atom.GetHybridization()
        sp.append(0)
        sp2.append(0)
        sp3.append(0)
    
    # --- Edge features ---
    rows, cols, edge_types = [], [], []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        
        if bond_type not in BOND_TYPES:
            raise ValueError(f"Bond type {bond_type} not in QM9 vocabulary")
        
        # Add both directions — PyG convention
        rows += [start, end]
        cols += [end, start]
        edge_types += [BOND_TYPES[bond_type], BOND_TYPES[bond_type]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=4).float()

    # --- num_hs: count H neighbors from edge structure ---
    z = torch.tensor(atomic_number, dtype=torch.long)
    N = len(atomic_number)
    hs = (z == 1).float()  # 1.0 where atom is hydrogen
    
    from torch_scatter import scatter
    num_hs = scatter(hs[rows], torch.tensor(cols), dim_size=N, reduce='sum')

    # --- Assemble node feature matrix ---
    x1 = F.one_hot(torch.tensor(type_idx), num_classes=5).float()  # 5 dims
    x2 = torch.tensor(
        [atomic_number, aromatic, sp, sp2, sp3, num_hs.tolist()],
        dtype=torch.float
    ).t().contiguous()  # 6 dims

    x = torch.cat([x1, x2], dim=-1)  # 11 dims total

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)