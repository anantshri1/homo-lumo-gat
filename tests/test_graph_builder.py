import pytest
import torch
from serving.graph_builder import smiles_to_graph

def test_methane_shape():
    data = smiles_to_graph('C')
    assert data.x.shape == (5, 11)
    assert data.edge_index.shape == (2, 8)
    assert data.edge_attr.shape == (8, 4)

def test_benzene_shape():
    data = smiles_to_graph('c1ccccc1')
    assert data.x.shape == (12, 11)
    assert data.edge_attr.shape == (24, 4)

def test_aromatic_bond():
    data = smiles_to_graph('c1ccccc1')
    # All C-C bonds in benzene are aromatic — index 3 in one-hot
    cc_edges = [i for i in range(data.edge_attr.shape[0]) 
                if data.edge_attr[i, 3] == 1.0]
    assert len(cc_edges) == 12  # 6 aromatic bonds × 2 directions

def test_invalid_smiles():
    with pytest.raises(ValueError, match="RDKit could not parse"):
        smiles_to_graph('not_a_molecule')

def test_carbon_node_matches_qm9():
    data = smiles_to_graph('C')
    carbon = data.x[0].tolist()
    expected = [0., 1., 0., 0., 0., 6., 0., 0., 0., 0., 4.]
    assert carbon == expected, f"Got {carbon}"