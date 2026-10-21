from pathlib import Path

# 项目路径
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CKP_DIR = ROOT_DIR / "ckp"

LOG_DIR = ROOT_DIR / "logs"

RAW_TRAIN_DATA_PATH = RAW_DATA_DIR / "train.txt"
RAW_VALID_DATA_PATH = RAW_DATA_DIR / "valid.txt"
RAW_TEST_DATA_PATH = RAW_DATA_DIR / "test.txt"
BERT_MODEL_PATH = ROOT_DIR / "models" / "bert-base-chinese"

# 训练参数
BATCH_SIZE = 16

LEARNING_RATE = 1e-5
EPOCHS = 10

# 推理与预处理使用相同的截断长度。
MAX_LENGTH = 128
INFERENCE_MODEL_PATH = CKP_DIR / "best"
