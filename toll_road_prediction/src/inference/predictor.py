import torch
from transformers import BertTokenizer

class TollRoadPredictor:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TollPredictionTransformer(config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def predict(self, trajectory):
        encoded = self.tokenizer.encode_plus(
            trajectory,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        is_toll_road = torch.sigmoid(outputs[0, 0]).item() > 0.5
        distance = outputs[0, 1].item()
        charge = outputs[0, 2].item()

        return {
            'is_toll_road': is_toll_road,
            'distance': distance,
            'charge': charge
        }

# Usage
config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
predictor = TollRoadPredictor('path/to/saved/model.pth', config)
result = predictor.predict("Sample trajectory string")
print(result)
