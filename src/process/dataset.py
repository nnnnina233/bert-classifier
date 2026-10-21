from datasets import load_from_disk
from transformers import default_data_collator
from torch.utils.data import DataLoader

from src.configuration.config import BATCH_SIZE, PROCESSED_DATA_DIR


def get_dataset(ds_type="train"):
    """加载预处理后数据集的指定划分。"""
    dataset = load_from_disk(str(PROCESSED_DATA_DIR))
    return dataset[ds_type]


def get_dataloader(ds_type="train"):
    """创建用于批量读取数据的 DataLoader。"""
    return DataLoader(
        get_dataset(ds_type),
        batch_size=BATCH_SIZE,
        shuffle=(ds_type == "train"),
        collate_fn=default_data_collator,
    )
