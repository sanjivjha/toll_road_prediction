import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup

class TransformerTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)

    def prepare_data(self, trajectories, labels):
        input_ids = []
        attention_masks = []
        for trajectory in trajectories:
            encoded = self.tokenizer.encode_plus(
                trajectory,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        
        return TensorDataset(input_ids, attention_masks, labels)

    def train(self, train_dataset, val_dataset, epochs, batch_size):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            self.model.train()
            for batch in train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.compute_loss(outputs, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Validation
            self.model.eval()
            val_loss = 0
            for batch in val_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    batch_loss = self.compute_loss(outputs, labels)
                    val_loss += batch_loss.item()
            
            val_loss /= len(val_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")

    def compute_loss(self, outputs, labels):
        # Assuming outputs and labels are [batch_size, 3]
        # where 3 represents [is_toll_road, distance, charge]
        toll_road_loss = torch.nn.BCEWithLogitsLoss()(outputs[:, 0], labels[:, 0])
        distance_loss = torch.nn.MSELoss()(outputs[:, 1], labels[:, 1])
        charge_loss = torch.nn.MSELoss()(outputs[:, 2], labels[:, 2])
        return toll_road_loss + distance_loss + charge_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# Usage
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TollPredictionTransformer(BertConfig())
trainer = TransformerTrainer(model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data and train
# trajectories = [...]  # List of trajectory strings
# labels = [...]  # List of (is_toll_road, distance, charge) tuples
# train_dataset = trainer.prepare_data(trajectories, labels)
# trainer.train(train_dataset, val_dataset, epochs=3, batch_size=32)
