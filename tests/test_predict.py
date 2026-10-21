import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import BatchEncoding

from src.runner.predict import Predictor


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(id2label={0: "乳品", 1: "鞋靴箱包"}, num_labels=2)

    def forward(self, input_ids):
        assert not self.training
        assert not torch.is_grad_enabled()
        return SimpleNamespace(logits=input_ids.float())


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Mock(side_effect=lambda texts, **kwargs: BatchEncoding(
            {"input_ids": torch.tensor([[3, 1], [1, 4]][:len(texts)])}
        ))
        self.predictor = Predictor(TinyClassifier(), self.tokenizer, "cpu")

    def test_batch_labels_scores_and_training_length(self):
        predictions = self.predictor.predict_details([" 牛奶 ", "运动鞋"])
        self.assertEqual([p["label"] for p in predictions], ["乳品", "鞋靴箱包"])
        self.assertEqual([p["label_id"] for p in predictions], [0, 1])
        self.assertEqual(predictions[0]["text"], "牛奶")
        self.assertAlmostEqual(predictions[0]["confidence"], torch.softmax(torch.tensor([3., 1.]), 0)[0].item())
        self.assertEqual(self.tokenizer.call_args.kwargs["max_length"], 128)
        self.assertTrue(self.tokenizer.call_args.kwargs["truncation"])

    def test_single_text_and_legacy_labels(self):
        self.assertEqual(self.predictor.predict("牛奶"), ["乳品"])
        self.assertEqual(self.predictor.labels, ["乳品", "鞋靴箱包"])

    def test_invalid_input(self):
        for value in [None, [], " ", [""], [42], {"text": "牛奶"}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.predictor.predict_details(value)


if __name__ == "__main__":
    unittest.main()
