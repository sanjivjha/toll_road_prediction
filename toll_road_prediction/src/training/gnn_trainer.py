import torch
from torch_geometric.data import Data, DataLoader

class GNNTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, dataloader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)
                loss = self.loss_fn(out, batch.y)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# Usage
model = RoadNetworkGNN(input_dim=10, hidden_dim=64, output_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()
trainer = GNNTrainer(model, optimizer, loss_fn)

# Prepare data and train
# (This part depends on how you structure your graph data)
