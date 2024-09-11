import unittest
import torch
from src.models.gnn_model import RoadNetworkGNN

class TestRoadNetworkGNN(unittest.TestCase):
    def setUp(self):
        self.model = RoadNetworkGNN(input_dim=10, hidden_dim=64, output_dim=32)

    def test_forward_pass(self):
        x = torch.randn(100, 10)  # 100 nodes, 10 features each
        edge_index = torch.randint(0, 100, (2, 300))  # 300 edges
        output = self.model(x, edge_index)
        self.assertEqual(output.shape, (100, 32))  # 100 nodes, 32 output features each

if __name__ == '__main__':
    unittest.main()
