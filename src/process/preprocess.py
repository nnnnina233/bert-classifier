from src.configuration.config import (
    BERT_MODEL_PATH,
    MAX_LENGTH,
    PROCESSED_DATA_DIR,
    RAW_TEST_DATA_PATH,
    RAW_TRAIN_DATA_PATH,
    RAW_VALID_DATA_PATH,
    ROOT_DIR,
)
from transformers import AutoTokenizer
from datasets import load_dataset, ClassLabel


def preprocess():
    """编码商品标题并保存训练、验证和测试数据集。"""
    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(RAW_TRAIN_DATA_PATH),
            "validation": str(RAW_VALID_DATA_PATH),
            "test": str(RAW_TEST_DATA_PATH)
        },
        delimiter='\t',
    )

    all_labels = sorted(set(dataset['train']['label']))
    dataset = dataset.cast_column("label", ClassLabel(names=all_labels))

    # 保存训练集标签的排序结果，保证各划分类别编号一致。
    with open(ROOT_DIR / "label_mapping.txt", "w", encoding="utf-8") as f:
        for i, label in enumerate(all_labels):
            f.write(f"{i}\t{label}\n")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)

    def batch_encode(batch):
        inputs = tokenizer(
            batch["text_a"], padding="max_length",
            truncation=True, max_length=MAX_LENGTH,
        )
        inputs["label"] = batch["label"]
        return inputs

    dataset = dataset.map(batch_encode, batched=True, remove_columns=["label", "text_a"])

    dataset.save_to_disk(str(PROCESSED_DATA_DIR))
    print(f"预处理完成，数据集已保存至：{PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    preprocess()
