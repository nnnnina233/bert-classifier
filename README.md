# BERT Product Classifier

基于 BERT 的中文商品标题分类项目，支持数据预处理、模型微调、断点恢复、评估、FastAPI 批量预测和 React 可视化操作。当前标签映射包含 30 个商品类别。

## 功能

- **数据处理**：读取 TSV 数据，生成类别映射并完成文本编码。
- **模型训练**：支持 GPU / CPU、CUDA BF16 混合精度、早停和 TensorBoard 日志。
- **训练恢复**：保存模型、优化器、训练位置与随机状态，自动恢复最近检查点。
- **模型评估**：计算 loss、accuracy 及 weighted precision、recall、F1。
- **推理服务**：提供命令行预测、批量分类 API、健康检查和 Swagger 文档。
- **前端界面**：React 商品分类页面，支持批量输入、结果展示、JSON 导出和服务状态检查。

## 项目结构

```text
bert-product-classifier/
├── data/
│   └── examples/                 # 数据格式示例
│       ├── train.tsv
│       ├── valid.tsv
│       └── test.tsv
├── frontend/                      # React + Vite 前端
│   ├── public/favicon.svg
│   ├── src/
│   │   ├── App.jsx                # 分类页面与接口交互
│   │   ├── main.jsx               # React 入口
│   │   └── styles.css             # 响应式样式
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js             # 开发服务与 API 代理
├── src/
│   ├── __init__.py
│   ├── __main__.py                # 命令行入口
│   ├── configuration/
│   │   ├── __init__.py
│   │   └── config.py              # 路径与基础参数
│   ├── process/
│   │   ├── __init__.py
│   │   ├── preprocess.py          # 数据预处理
│   │   └── dataset.py             # 数据集加载
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── train.py               # 训练、验证与检查点管理
│   │   ├── evaluate.py            # 离线评估
│   │   └── predict.py             # 模型加载与预测
│   └── web/
│       ├── __init__.py
│       └── main.py                # FastAPI 应用
├── tests/
│   ├── test_api.py
│   ├── test_predict.py
│   └── test_train_checkpoint.py
├── .gitignore
├── label_mapping.txt              # 类别编号与名称
├── requirements.txt               # 运行依赖
├── requirements-dev.txt           # 测试依赖
└── README.md
```

模型、完整数据和训练产物不包含在仓库中，本地目录约定如下：

| 路径 | 用途 |
| --- | --- |
| `data/raw/` | 原始训练、验证和测试数据 |
| `data/processed/` | 预处理后的 Hugging Face 数据集 |
| `models/bert-base-chinese/` | BERT 基础模型与分词器 |
| `ckp/checkpoints/` | 可恢复的训练检查点 |
| `ckp/best/` | 验证 loss 最低的模型与分词器 |
| `ckp/final/` | 训练结束时的模型与分词器 |
| `logs/` | TensorBoard 日志 |

## 安装

推荐使用 Python 3.11 或 3.12，前端构建需要 Node.js 22.12+。以下命令均在项目根目录执行，并使用已安装项目依赖的 Python 环境。

```bash
python -m venv .venv
```

激活虚拟环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

安装运行依赖：

```bash
python -m pip install -r requirements.txt
```

PyTorch 的 CPU / CUDA 构建可根据运行环境选择，参见 [官方安装说明](https://pytorch.org/get-started/locally/)。

## 数据与模型准备

### 数据格式

在 `data/raw/` 下准备 `train.txt`、`valid.txt` 和 `test.txt`。文件使用 UTF-8 编码，以制表符分隔，包含 `label` 和 `text_a` 两列：

```text
label	text_a
乳品	纯牛奶整箱250ml早餐奶
鞋靴箱包	男士轻便跑步运动鞋
```

验证集和测试集的类别须包含在训练集中。`data/examples/` 提供少量手写格式示例，不能用于衡量模型效果。类别编号由训练集标签排序生成，并写入 `label_mapping.txt`。

### 模型文件

将 [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) 的模型配置、权重及分词器文件放入 `models/bert-base-chinese/`。

训练后，模型和分词器分别保存到 `ckp/best/` 与 `ckp/final/`。预测服务默认加载 `ckp/best/`；已有兼容的微调模型时，可直接启动服务。

## 使用

### 预处理、训练与评估

```bash
python -m src preprocess
python -m src train
python -m src evaluate --split validation
python -m src evaluate --split test
```

训练默认恢复已有检查点。新实验可将 `TrainConfig.resume` 设为 `False`，或使用新的 `output_dir`；恢复训练时须保持数据、批次和标签配置一致。

### 命令行预测

```bash
python -m src predict "纯牛奶整箱250ml" "男士跑步运动鞋"
```

可通过 `--model-path` 指定模型目录，通过 `--device cpu` 强制使用 CPU。

### 启动 API

```bash
python -m src serve --host 127.0.0.1 --port 8000
```

模型加载完成后，访问 <http://127.0.0.1:8000/docs> 调试接口。开发时可添加 `--reload`。

### 前端界面

首次使用时安装依赖并构建：

```bash
npm --prefix frontend ci
npm --prefix frontend run build
python -m src serve
```

构建完成后启动或重启后端，访问 **<http://127.0.0.1:8000>** 即可使用分类页面。前端和 API 使用同一个地址，无需额外启动 Node 服务。页面支持每行一条标题、填入示例、批量分类、结果导出和服务重连。

开发前端时，在后端运行的同时，另开终端执行：

```bash
npm --prefix frontend run dev
```

访问 <http://127.0.0.1:5173>。Vite 将接口请求代理至 `http://127.0.0.1:8000`，可在 `frontend/.env.local` 中通过 `VITE_API_TARGET` 更换后端地址。前端构建产物 `frontend/dist/` 和依赖 `node_modules/` 不提交到仓库。


## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | React 页面；未构建前端时跳转至 API 文档 |
| `GET` | `/health` | 返回模型状态、设备及类别数 |
| `GET` | `/labels` | 返回按类别编号排列的标签列表 |
| `POST` | `/predict` | 批量预测商品标题 |
| `GET` | `/docs` | Swagger UI |

### 预测请求

```http
POST /predict
Content-Type: application/json
```

```json
{
  "texts": ["纯牛奶整箱250ml", "男士跑步运动鞋"]
}
```

响应示例，具体类别和分数以实际模型输出为准：

```json
{
  "predictions": [
    {"text": "纯牛奶整箱250ml", "label_id": 3, "label": "乳品", "confidence": 0.98},
    {"text": "男士跑步运动鞋", "label_id": 26, "label": "鞋靴箱包", "confidence": 0.95}
  ]
}
```

每次请求支持 1–32 条文本，单条最多 2000 个字符。空白文本或不合法参数返回 `422`；推理失败返回 `500`。文本编码时截断至 128 tokens。`confidence` 为 softmax 分数，不等同于预测正确率。

## 配置

基础参数位于 `src/configuration/config.py`，训练策略位于 `src/runner/train.py` 的 `TrainConfig`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `BATCH_SIZE` | `16` | 训练批次大小 |
| `LEARNING_RATE` | `1e-5` | 学习率 |
| `EPOCHS` | `10` | 最大训练轮数 |
| `MAX_LENGTH` | `128` | 编码后的最大序列长度 |
| `checkpoint_steps` | `100` | 检查点保存间隔 |
| `checkpoint_limit` | `3` | 保留的最近检查点数量 |
| `eval_steps` | `1000` | 验证间隔 |
| `early_stopping_patience` | `5` | 允许连续无有效改善的验证次数，`0` 关闭早停 |

API 启动时还支持以下进程环境变量：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CLASSIFIER_MODEL_PATH` | `ckp/best/` | 微调模型与分词器目录 |
| `CLASSIFIER_DEVICE` | 自动选择 | `cpu` 或 `cuda`，默认优先使用 CUDA |

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

测试覆盖接口校验、预测结果、训练恢复、早停与检查点保留策略，无需完整数据集或 BERT 权重。CUDA 相关测试在无 GPU 时自动跳过。
