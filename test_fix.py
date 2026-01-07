
import sys
import torch
import numpy as np
import logging

# Add current directory to path
sys.path.append(".")

from src.data.graph_generator_minimal import MinimalGraphGenerator

def test_isolated_atom():
    print("Testing isolated atom graph generation...")
    generator = MinimalGraphGenerator()

    # Use standard Carbon but stripped of bonds?
    # Or just Water? Water has bonds.
    # Methane? C.
    smiles = "C"

    try:
        graph = generator.smiles_to_graph(smiles, spectrum=np.zeros(1000))
        print(f"Graph generated for {smiles}")
        print(f"Nodes: {graph.num_nodes}")
        print(f"Edges: {graph.num_edges}")
        print(f"Edge Attr Shape: {graph.edge_attr.shape}")

        expected_dim = generator.featurizer.get_edge_dim()
        # Shape should be (0, expected_dim)
        if graph.edge_attr.shape[1] == expected_dim:
            print(f"SUCCESS: Edge attribute dimension matches expected ({expected_dim})")
        else:
            print(f"FAILURE: Edge attribute dimension {graph.edge_attr.shape[1]} != expected {expected_dim}")
            sys.exit(1)

    except Exception as e:
        print(f"Error generating graph: {e}")
        # Print detailed traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_isolated_atom()
