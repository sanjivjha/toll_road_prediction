import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class TollPredictionTransformer(nn.Module):
    def __init__(self, config):
        super(TollPredictionTransformer, self).__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction = self.classifier(sequence_output[:, 0, :])
        return prediction

# Usage
config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
model = TollPredictionTransformer(config)
