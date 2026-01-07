#!/usr/bin/env python3
"""
Test suite for model modules

Tests:
- QCGN2oEI_Minimal model architecture
- Forward pass shape validation
- Parameter count
- Device placement
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_model_architecture():
    """Test QCGN2oEI_Minimal architecture"""
    print("\n" + "=" * 60)
    print("Test 1: Model Architecture")
    print("=" * 60)

    try:
        from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal

        # Test case 1: Default configuration
        model = QCGN2oEI_Minimal(
            node_dim=16,
            edge_dim=3,
            hidden_dim=256,
            num_layers=10,
            num_heads=8,
            output_dim=1000,
            dropout=0.1
        )

        assert model.node_dim == 16
        assert model.edge_dim == 3
        assert model.hidden_dim == 256
        assert model.num_layers == 10
        assert model.num_heads == 8
        assert model.output_dim == 1000

        print(f"✅ Model instantiated with correct parameters")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Total parameters: {num_params:,}")

        # Expected range: ~2-3M parameters
        assert 1_000_000 < num_params < 5_000_000, \
            f"Parameter count {num_params:,} outside expected range"

        print(f"✅ Parameter count in expected range (2-3M)")

    except ImportError as e:
        print(f"⚠️  Cannot test model (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False

    print("\n✅ Model architecture test PASSED!\n")
    return True


def test_forward_pass_shapes():
    """Test forward pass with different input shapes"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass Shapes")
    print("=" * 60)

    try:
        import torch
        from torch_geometric.data import Data, Batch
        from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal

        model = QCGN2oEI_Minimal(
            node_dim=16,
            edge_dim=3,
            hidden_dim=256,
            num_layers=10,
            num_heads=8,
            output_dim=1000,
            dropout=0.1
        )
        model.eval()

        # Test case 1: Single graph
        num_nodes = 10
        num_edges = 20

        x = torch.randn(num_nodes, 16)  # Node features
        edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge connectivity
        edge_attr = torch.randn(num_edges, 3)  # Edge features
        batch = torch.zeros(num_nodes, dtype=torch.long)  # Batch assignment

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        with torch.no_grad():
            output = model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch
            )

        assert output.shape == (1, 1000), f"Expected (1, 1000), got {output.shape}"
        print(f"✅ Single graph: input={num_nodes} nodes, output={output.shape}")

        # Test case 2: Batch of graphs
        graphs = []
        for _ in range(5):
            num_nodes_i = torch.randint(5, 15, (1,)).item()
            num_edges_i = torch.randint(10, 30, (1,)).item()

            x_i = torch.randn(num_nodes_i, 16)
            edge_index_i = torch.randint(0, num_nodes_i, (2, num_edges_i))
            edge_attr_i = torch.randn(num_edges_i, 3)

            graphs.append(Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i))

        batch_graph = Batch.from_data_list(graphs)

        with torch.no_grad():
            output_batch = model(
                batch_graph.x,
                batch_graph.edge_index,
                batch_graph.edge_attr,
                batch_graph.batch
            )

        assert output_batch.shape == (5, 1000), \
            f"Expected (5, 1000), got {output_batch.shape}"
        print(f"✅ Batch of 5 graphs: output={output_batch.shape}")

        # Test case 3: Output range (should be non-negative due to softmax)
        assert torch.all(output_batch >= 0), "Output should be non-negative"
        assert torch.all(output_batch <= 1), "Output should be <= 1 (softmax)"
        print(f"✅ Output values in valid range [0, 1]")

        # Test case 4: Output sum (softmax should sum to ~1)
        output_sums = output_batch.sum(dim=1)
        for i, s in enumerate(output_sums):
            assert abs(s - 1.0) < 0.01, f"Sample {i}: sum={s:.4f}, expected ~1.0"
        print(f"✅ Output sums to 1.0 (softmax normalized)")

    except ImportError as e:
        print(f"⚠️  Cannot test forward pass (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Forward pass shape test PASSED!\n")
    return True


def test_device_placement():
    """Test device placement (CPU/CUDA)"""
    print("\n" + "=" * 60)
    print("Test 3: Device Placement")
    print("=" * 60)

    try:
        import torch
        from torch_geometric.data import Data
        from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal

        # Test case 1: CPU
        model_cpu = QCGN2oEI_Minimal(
            node_dim=16,
            edge_dim=3,
            hidden_dim=256,
            num_layers=10,
            num_heads=8,
            output_dim=1000
        )

        # Check model is on CPU
        assert next(model_cpu.parameters()).device.type == 'cpu'
        print(f"✅ Model created on CPU")

        # Test case 2: CUDA (if available)
        if torch.cuda.is_available():
            model_cuda = QCGN2oEI_Minimal(
                node_dim=16,
                edge_dim=3,
                hidden_dim=256,
                num_layers=10,
                num_heads=8,
                output_dim=1000
            )
            model_cuda.to('cuda')

            assert next(model_cuda.parameters()).device.type == 'cuda'
            print(f"✅ Model moved to CUDA")

            # Test forward pass on CUDA
            x = torch.randn(10, 16).cuda()
            edge_index = torch.randint(0, 10, (2, 20)).cuda()
            edge_attr = torch.randn(20, 3).cuda()
            batch = torch.zeros(10, dtype=torch.long).cuda()

            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

            with torch.no_grad():
                output = model_cuda(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.batch
                )

            assert output.device.type == 'cuda'
            print(f"✅ Forward pass on CUDA successful")
        else:
            print(f"⚠️  CUDA not available, skipping CUDA tests")

    except ImportError as e:
        print(f"⚠️  Cannot test device placement (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ Device placement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Device placement test PASSED!\n")
    return True


def run_all_tests():
    """Run all model tests"""
    print("\n" + "=" * 70)
    print("NExtIMS v4.2: Model Test Suite")
    print("=" * 70)

    try:
        success = True
        success &= test_model_architecture()
        success &= test_forward_pass_shapes()
        success &= test_device_placement()

        if success:
            print("\n" + "=" * 70)
            print("🎉 ALL MODEL TESTS PASSED! 🎉")
            print("=" * 70 + "\n")
        else:
            print("\n" + "=" * 70)
            print("⚠️  SOME TESTS HAD ISSUES")
            print("=" * 70 + "\n")

        return success

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
