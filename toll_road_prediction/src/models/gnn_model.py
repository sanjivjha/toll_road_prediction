import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class RoadNetworkGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RoadNetworkGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
