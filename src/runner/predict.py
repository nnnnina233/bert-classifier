"""加载微调模型并预测商品类别。"""
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.configuration.config import INFERENCE_MODEL_PATH, MAX_LENGTH


class Predictor:
    def __init__(self, model, tokenizer, device, max_length=MAX_LENGTH):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 多个请求共用一份模型，限制并发推理的显存占用。
        self._lock = Lock()

    @classmethod
    def from_pretrained(cls, model_path=INFERENCE_MODEL_PATH, device=None):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"模型目录不存在：{model_path}。请先训练，或设置 CLASSIFIER_MODEL_PATH。"
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True,
        )
        return cls(model, tokenizer, device)

    @property
    def labels(self):
        return [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict_details(self, input_data):
        texts = [input_data] if isinstance(input_data, str) else input_data
        if not isinstance(texts, list) or not texts:
            raise ValueError("输入必须是非空文本或非空文本列表。")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("每条文本必须是非空字符串。")
        texts = [text.strip() for text in texts]
        with self._lock, torch.inference_mode():
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(self.device)
            logits = self.model(**inputs).logits
            scores, indices = torch.softmax(logits, dim=-1).max(dim=-1)
            scores, indices = scores.cpu().tolist(), indices.cpu().tolist()
        return [
            {"text": text, "label_id": index,
             "label": self.model.config.id2label[index], "confidence": score}
            for text, index, score in zip(texts, indices, scores)
        ]

    def predict(self, input_data):
        """返回类别名称列表。"""
        return [item["label"] for item in self.predict_details(input_data)]


def predict(texts, model_path=INFERENCE_MODEL_PATH, device=None):
    return Predictor.from_pretrained(model_path, device).predict_details(texts)
