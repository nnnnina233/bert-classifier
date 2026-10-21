from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, default_data_collator

from src.configuration.config import BATCH_SIZE, CKP_DIR, LOG_DIR
from src.process.dataset import get_dataset
from src.runner.train import Trainer, TrainConfig


def evaluate(model_path=CKP_DIR / "best", split="validation", device=None, batch_size=BATCH_SIZE):
    """在验证集或测试集上评估模型。"""
    if split not in ("validation", "test"):
        raise ValueError("split 必须是 validation 或 test。")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}，请先训练并保存模型。")

    device = torch.device(device if device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"评估模型：{model_path} | 数据集：{split} | 使用设备：{device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        model_path, local_files_only=True,
    ).to(device)

    valid_dataset = get_dataset(split)
    if len(valid_dataset) == 0:
        raise ValueError(f"{split} 数据集为空，无法评估。")
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=default_data_collator,
    )
    # 复用 Trainer 的评估方法，关闭断点恢复。
    train_loader = DataLoader(
        get_dataset("train"), batch_size=batch_size, shuffle=False,
        collate_fn=default_data_collator, generator=torch.Generator(),
    )
    config = TrainConfig(
        batch_size=batch_size, resume=False,
        log_dir=LOG_DIR / "evaluation", show_classification_report=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    runner = Trainer(
        model=model, tokenizer=tokenizer, optimizer=optimizer,
        train_loader=train_loader, valid_loader=valid_loader,
        device=device, training_config=config,
    )
    try:
        result = runner.evaluate()
    finally:
        runner.writer.close()

    print("Evaluation Metrics:")
    print(f"  loss: {result['loss']:.6f}")
    for metric, value in result["metrics"].items():
        print(f"  {metric}: {value:.6f}")
    return result

if __name__ == "__main__":
    evaluate(split="test")
