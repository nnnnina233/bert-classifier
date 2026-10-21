from dataclasses import dataclass
from pathlib import Path
import random
import pickle
import time
import math
from itertools import islice

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    default_data_collator,
)

from src.configuration.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    CKP_DIR,
    LOG_DIR,
    BERT_MODEL_PATH,
)
from src.process.dataset import get_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


def replace_checkpoint(temporary, destination):
    """Retry transient Windows sharing locks without rewriting the checkpoint."""
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (32, 33) or attempt == 7:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 1.0))


@dataclass
class TrainConfig:
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    output_dir: Path = CKP_DIR
    log_dir: Path = LOG_DIR / "bert-classifier"
    checkpoint_steps: int = 100
    checkpoint_limit: int = 3
    eval_steps: int = 1000
    log_steps: int = 100
    show_classification_report: bool = False
    resume: bool = True
    seed: int = 42
    early_stopping_patience: int = 5  # 连续无改善的验证次数；0 表示关闭
    early_stopping_min_delta: float = 0.0  # loss 降幅需严格大于此值


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        train_loader,
        valid_loader,
        device,
        training_config,
    ):
        # 模型已在外面移到指定设备
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device(device)
        self.training_config = training_config
        self.use_bf16 = False
        if self.device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self.device):
                self.use_bf16 = torch.cuda.is_bf16_supported()
        print(f"计算精度：{'BF16 混合精度' if self.use_bf16 else 'FP32'}")

        self.step = 0
        self.min_loss = float("inf")
        self.early_stopping_best_loss = float("inf")
        self.early_stopping_bad_evals = 0
        self.last_eval_step = -1
        self.start_epoch = 0
        self.next_batch = 0
        self.epoch_loss = 0.0
        self.epoch_samples = 0
        if not isinstance(training_config.early_stopping_patience, int) or training_config.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience 必须是非负整数。")
        if not math.isfinite(training_config.early_stopping_min_delta) or training_config.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta 必须是有限的非负数。")
        if training_config.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps 必须大于 0。")
        if training_config.checkpoint_limit < 1:
            raise ValueError("checkpoint_limit 必须至少为 1。")
        if training_config.eval_steps <= 0 or training_config.log_steps <= 0:
            raise ValueError("eval_steps 和 log_steps 必须大于 0。")
        if self.train_loader.generator is None:
            raise ValueError("训练 DataLoader 需要独立 generator，以便恢复打乱顺序。")
        if training_config.resume:
            self.load_checkpoint()

        self.writer = SummaryWriter(
            log_dir=str(training_config.log_dir),
            purge_step=self.step + 1 if self.step else None,
        )

    def checkpoint_signature(self):
        """恢复时确保数据、批次划分和类别含义没有改变。"""
        return {
            "dataset_size": len(self.train_loader.dataset),
            "dataset_fingerprint": getattr(self.train_loader.dataset, "_fingerprint", None),
            "batch_size": self.train_loader.batch_size,
            "drop_last": self.train_loader.drop_last,
            "seed": self.training_config.seed,
            "id2label": self.model.config.id2label,
        }

    def save_checkpoint(self, epoch, next_batch, total_loss=0.0, total_samples=0):
        """epoch 和 next_batch 都指向恢复后要继续的位置（从 0 开始）。"""
        checkpoint_dir = self.training_config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"checkpoint-{self.step}.pt"
        state = {
            "version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": epoch,
            "next_batch": next_batch,
            "epoch_loss": total_loss,
            "epoch_samples": total_samples,
            "min_loss": self.min_loss,
            "early_stopping_best_loss": self.early_stopping_best_loss,
            "early_stopping_bad_evals": self.early_stopping_bad_evals,
            "last_eval_step": self.last_eval_step,
            "signature": self.checkpoint_signature(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "python_rng": random.getstate(),
        }
        # 先完整写入临时文件，成功后替换；中断留下的 .tmp 不会被恢复。
        temporary = path.with_suffix(".pt.tmp")
        with temporary.open("wb") as stream:
            torch.save(state, stream)
        replace_checkpoint(temporary, path)
        # 新文件完整落盘后，只清理这个目录中更早的检查点文件。
        root = checkpoint_dir.resolve()
        saved = [p for p in checkpoint_dir.glob("checkpoint-*.pt")
                 if p.stem.removeprefix("checkpoint-").isdigit()]
        saved.sort(key=lambda p: int(p.stem.split("-")[-1]), reverse=True)
        for old_path in saved[self.training_config.checkpoint_limit:]:
            if old_path.resolve().parent != root or old_path.is_symlink():
                raise ValueError(f"检查点路径超出预期目录：{old_path}")
            old_path.unlink()
        tqdm.write(f"检查点已保存：{path}")

    def load_checkpoint(self):
        checkpoint_dir = self.training_config.output_dir / "checkpoints"
        paths = [p for p in checkpoint_dir.glob("checkpoint-*.pt")
                 if p.stem.removeprefix("checkpoint-").isdigit()]
        paths.sort(key=lambda p: int(p.stem.split("-")[-1]), reverse=True)
        for path in paths:
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, IndexError, ValueError) as error:
                print(f"检查点无法读取，尝试较早版本：{path} ({error})")
                continue
            if state["version"] != 1 or state["signature"] != self.checkpoint_signature():
                raise ValueError("检查点与当前数据/批次/标签配置不一致；新实验请设置 resume=False 或更换 output_dir。")
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.step = state["step"]
            self.start_epoch = state["epoch"]
            self.next_batch = state["next_batch"]
            self.epoch_loss = state["epoch_loss"]
            self.epoch_samples = state["epoch_samples"]
            self.min_loss = state["min_loss"]
            # 兼容加入早停前保存的 checkpoint。
            self.early_stopping_best_loss = state.get("early_stopping_best_loss", self.min_loss)
            self.early_stopping_bad_evals = state.get("early_stopping_bad_evals", 0)
            self.last_eval_step = state.get("last_eval_step", -1)
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            if self.device.type == "cuda" and state["cuda_rng"]:
                if len(state["cuda_rng"]) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
            print(f"已恢复 {path}：step={self.step}，下一位置 epoch={self.start_epoch + 1}，batch={self.next_batch + 1}")
            return
        if paths:
            raise RuntimeError("没有可读取的检查点，未自动从头训练。")
        print("未找到检查点，从头训练。")

    def save_model(self, name="final"):
        """保存模型和配套分词器，best 与 final 使用不同目录。"""
        save_dir = self.training_config.output_dir / name

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        tqdm.write(f"模型已保存到：{save_dir}")

    def compute_metrics(self, labels, predictions):
        """统一计算整个验证集的指标，P/R/F1 按类别样本数加权。"""
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        }

    def evaluate(self):
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        try:
            with torch.no_grad():
                for batch in tqdm(
                    self.valid_loader,
                    desc="[Evaluate]",
                    dynamic_ncols=True,
                    position=1,
                    leave=False,
                ):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_bf16,
                    ):
                        outputs = self.model(**batch)
                    total_loss += outputs.loss.item() * batch["labels"].size(0)
                    all_labels.extend(batch["labels"].cpu().tolist())
                    all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())

            if not all_labels:
                raise ValueError("验证集没有可用样本，无法计算验证 loss。")

            # 完整遍历后计算；除以样本数，而不是 batch 数。
            metrics = self.compute_metrics(all_labels, all_preds)
            if self.training_config.show_classification_report:
                tqdm.write("\nDetailed Classification Report:")
                tqdm.write(classification_report(all_labels, all_preds, zero_division=0))
            return {"loss": total_loss / len(all_labels), "metrics": metrics}
        finally:
            self.model.train(was_training)

    @property
    def should_stop_early(self):
        patience = self.training_config.early_stopping_patience
        return patience > 0 and self.early_stopping_bad_evals >= patience

    def validate_and_save_best(self):
        """按全局 step 记录验证结果，同一步不重复验证。"""
        if self.last_eval_step == self.step:
            return
        result = self.evaluate()
        val_loss = result["loss"]
        if not math.isfinite(val_loss):
            raise ValueError(f"验证 loss 不是有限数值：{val_loss}")
        self.writer.add_scalar("validation/loss", val_loss, self.step)
        for name, value in result["metrics"].items():
            self.writer.add_scalar(f"validation/{name}", value, self.step)
        self.writer.flush()
        tqdm.write(
            f"[Evaluate | Step: {self.step}] Loss: {val_loss:.6f} | "
            f"Accuracy: {result['metrics']['accuracy']:.2%} | "
            f"F1: {result['metrics']['f1']:.4f}"
        )
        if val_loss < self.min_loss:
            self.save_model("best")
            self.min_loss = val_loss
            tqdm.write(f"最佳验证 loss 更新为：{self.min_loss:.6f}")
        self.last_eval_step = self.step
        # 独立于 best 模型的保存阈值，允许多个小幅下降累计成为有效改善。
        if val_loss < self.early_stopping_best_loss - self.training_config.early_stopping_min_delta:
            self.early_stopping_best_loss = val_loss
            self.early_stopping_bad_evals = 0
        else:
            self.early_stopping_bad_evals += 1
        if self.training_config.early_stopping_patience > 0:
            tqdm.write(
                f"早停计数：{self.early_stopping_bad_evals}/"
                f"{self.training_config.early_stopping_patience}（按验证次数）"
            )
        if self.should_stop_early:
            tqdm.write("触发早停：验证 loss 持续未改善，最佳模型保存在 best 目录。")

    def train(self):
        """执行训练循环。"""
        try:
            if self.should_stop_early:
                tqdm.write("恢复的 checkpoint 已达到早停条件；提高 patience 或设为 0 可继续训练。")
                self.save_model()
                return
            for epoch in range(self.start_epoch, self.training_config.epochs):
                self.model.train()

                resuming = epoch == self.start_epoch
                skip_batches = self.next_batch if resuming else 0
                total_loss = self.epoch_loss if resuming else 0.0
                total_samples = self.epoch_samples if resuming else 0
                # 每轮使用独立固定种子，恢复时重建同一份样本顺序。
                self.train_loader.generator.manual_seed(self.training_config.seed + epoch)

                progress = tqdm(
                    islice(enumerate(self.train_loader), skip_batches, None),
                    total=len(self.train_loader),
                    initial=skip_batches,
                    desc=f"[Epoch: {epoch + 1}]",
                    dynamic_ncols=True,
                    position=0,
                    leave=True,
                )

                for batch_index, batch in progress:
                    # collator 已经把数据整理成 Tensor
                    batch = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                    }

                    # 清空上一批的梯度
                    self.optimizer.zero_grad(set_to_none=True)

                    # 前向计算；传入 labels 后模型会计算 loss
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_bf16,
                    ):
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    # BF16 不需要 GradScaler；反向传播放在 autocast 外。
                    loss.backward()
                    self.optimizer.step()

                    self.step += 1

                    loss_value = loss.item()
                    batch_size = batch["labels"].size(0)

                    total_loss += loss_value * batch_size
                    total_samples += batch_size

                    # 记录每一步的损失
                    self.writer.add_scalar(
                        "train/step_loss",
                        loss_value,
                        self.step,
                    )

                    progress.set_postfix(
                        loss=f"{loss_value:.4f}"
                    )
                    if self.step % self.training_config.log_steps == 0:
                        tqdm.write(
                            f"[Epoch: {epoch + 1} | Step: {self.step}] "
                            f"Loss: {loss_value:.6f}"
                        )
                    if self.step % self.training_config.eval_steps == 0:
                        self.validate_and_save_best()
                    if self.should_stop_early:
                        self.save_checkpoint(epoch, batch_index + 1, total_loss, total_samples)
                        self.save_model()
                        return
                    if self.step % self.training_config.checkpoint_steps == 0:
                        self.save_checkpoint(epoch, batch_index + 1, total_loss, total_samples)

                # 按样本数计算本轮平均损失
                average_loss = total_loss / total_samples

                self.writer.add_scalar(
                    "train/epoch_loss",
                    average_loss,
                    epoch + 1,
                )

                tqdm.write(
                    f"Epoch {epoch + 1} 完成，"
                    f"平均 loss：{average_loss:.4f}"
                )

                # 轮末也验证；若恰好已验证则自动跳过。
                self.validate_and_save_best()

                # 验证后再保存轮次边界，恢复时不重复验证，并保留最新 min_loss。
                self.save_checkpoint(epoch + 1, 0)
                if self.should_stop_early:
                    break

            # 所有轮次完成后保存
            self.save_model()

        finally:
            # 正常结束或发生异常都关闭日志
            self.writer.close()


def train():
    """准备训练所需对象，再启动训练器。"""
    training_config = TrainConfig()

    # 1. 选择设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"使用设备：{device}")

    # 2. 加载预处理后的训练集
    train_dataset = get_dataset("train")

    if len(train_dataset) == 0:
        raise ValueError("训练集为空，请检查预处理结果。")

    # 预处理时使用了 ClassLabel，直接读取其中的类别顺序
    labels = train_dataset.features["label"].names

    id2label = {
        i: label for i, label in enumerate(labels)
    }
    label2id = {
        label: i for i, label in id2label.items()
    }

    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator(),
    )

    valid_dataset = get_dataset("validation")
    if len(valid_dataset) == 0:
        raise ValueError("验证集为空，请检查预处理结果。")
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    # 4. 加载分词器
    # 数据已经编码过；这里加载是为了随训练后的模型一起保存
    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_PATH,
        local_files_only=True,
    )

    # 5. 加载分类模型，先移到设备，再创建优化器
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_PATH,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    ).to(device)

    # 6. 创建优化器，只创建一次
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
    )

    # 7. 创建并运行训练器
    runner = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        training_config=training_config,
    )
    runner.train()

if __name__ == "__main__":
    train()
