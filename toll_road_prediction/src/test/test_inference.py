import unittest
import torch
from transformers import BertConfig
from src.models.transformer_model import TollPredictionTransformer

class TestTollPredictionTransformer(unittest.TestCase):
    def setUp(self):
        config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=2, num_attention_heads=2)
        self.model = TollPredictionTransformer(config)

    def test_forward_pass(self):
        input_ids = torch.randint(0, 30522, (1, 512))
        attention_mask = torch.ones((1, 512))
        output = self.model(input_ids, attention_mask)
        self.assertEqual(output.shape, (1, 3))  # Batch size 1, 3 outputs (is_toll_road, distance, charge)

if __name__ == '__main__':
    unittest.main()
