# BERT 与 LangChain 工作流练习

以中文商品标题分类为例，练习 BERT 微调、模型推理和 LangChain 流程编排。接口和页面用于查看分类结果与相似商品，现有模型包含 30 个商品类别。

## 练习内容

- **BERT**：使用预训练模型完成文本分类，理解分词、标签映射、训练与评估，以及模型保存和断点恢复。
- **LangChain**：把分类和检索函数封装成工具，使用 `RunnableLambda`、`RunnableParallel` 和 `|` 组合流程，观察数据如何在节点间传递。
- **向量检索**：用本地 embedding 模型编码商品标题，通过 FAISS 查找相似样本，理解相似度、元数据和索引保存加载。

BERT 使用 Hugging Face 提供的预训练模型；LangChain 部分采用代码预先确定的工作流，不调用 LLM。学习重点是理解各部分的输入输出，以及如何将它们串联起来。

## 目录

```text
bert-classifier/
├── data/examples/           # TSV 数据格式示例
├── frontend/
│   ├── public/favicon.svg
│   ├── src/
│   │   ├── App.jsx          # 分类页面
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── src/
│   ├── __main__.py          # 命令行入口
│   ├── configuration/
│   │   └── config.py        # 路径和训练参数
│   ├── process/
│   │   ├── preprocess.py    # 数据预处理
│   │   └── dataset.py       # 数据加载
│   ├── runner/
│   │   ├── train.py         # 训练与检查点管理
│   │   ├── evaluate.py      # 模型评估
│   │   └── predict.py       # 预测
│   ├── retrieval/
│   │   ├── corpus.py       # 样本清理与抽样
│   │   ├── index.py        # 本地 embedding 与 FAISS 索引
│   │   ├── workflow.py     # LangChain 工具与 LCEL 工作流
│   │   └── service.py      # 索引状态及加载
│   └── web/
│       └── main.py          # API 与前端静态文件服务
├── tests/
│   ├── test_api.py
│   ├── test_predict.py
│   ├── test_retrieval.py
│   └── test_train_checkpoint.py
├── label_mapping.txt
├── requirements.txt
├── requirements-dev.txt
├── requirements-retrieval.txt
├── .gitignore
└── README.md
```

## 环境

- Python 3.11 或 3.12
- Node.js 22.12+
- GPU 可选；未检测到 CUDA 时使用 CPU

在项目根目录安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux / macOS 用 `source .venv/bin/activate` 激活环境。PyTorch 安装可参考[官方说明](https://pytorch.org/get-started/locally/)，选择对应的 CPU 或 CUDA 版本。

## 数据和模型

完整数据集和模型权重不在仓库中，需要放到以下位置：

| 路径 | 内容 |
| --- | --- |
| `data/raw/train.txt` | 训练集 |
| `data/raw/valid.txt` | 验证集 |
| `data/raw/test.txt` | 测试集 |
| `models/bert-base-chinese/` | BERT 基础模型配置、权重和分词器 |

基础模型可从 [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) 下载。

数据为 UTF-8 编码的 TSV 文件，首行包含 `label` 和 `text_a`，两列之间用制表符分隔：

```text
label	text_a
乳品	纯牛奶整箱250ml
鞋靴箱包	男士跑步运动鞋
```

验证集和测试集的标签必须在训练集中出现。`data/examples/` 中是格式示例。预处理时按训练集标签排序，生成 `label_mapping.txt`。

## BERT 训练和评估

```bash
python -m src preprocess
python -m src train
python -m src evaluate --split validation
python -m src evaluate --split test
```

训练包含早停、断点恢复和 TensorBoard 日志。默认恢复最近的有效检查点；更换数据或重新训练时，在 `TrainConfig` 中设置 `resume=False` 或更换 `output_dir`。

| 输出目录 | 内容 |
| --- | --- |
| `data/processed/` | 编码后的数据集 |
| `ckp/checkpoints/` | 训练检查点，保留最近 3 个 |
| `ckp/best/` | 验证 loss 最低的模型和分词器 |
| `ckp/final/` | 训练结束时的模型和分词器 |
| `logs/` | TensorBoard 日志 |

路径、学习率、batch size 等参数见 `src/configuration/config.py`；验证间隔、检查点和早停设置见 `src/runner/train.py` 中的 `TrainConfig`。

## 运行页面

已有微调模型时，构建前端并启动服务：

```bash
npm --prefix frontend ci
npm --prefix frontend run build
python -m src serve
```

- 页面：<http://127.0.0.1:8000>
- API 文档：<http://127.0.0.1:8000/docs>

服务默认读取 `ckp/best/`。模型目录必须同时包含模型和分词器。更换端口用 `python -m src serve --port 8001`。

开发前端时，保持后端运行，另开终端执行 `npm --prefix frontend run dev`，访问 <http://127.0.0.1:5173>。接口代理默认指向 `http://127.0.0.1:8000`，可在 `frontend/.env.local` 中设置 `VITE_API_TARGET`。

## LangChain 工作流与相似商品检索

安装可选依赖并建立索引：

```bash
python -m pip install -r requirements-retrieval.txt
python -m src build-index
```

首次建库会下载 `BAAI/bge-small-zh-v1.5` 至 `models/bge-small-zh-v1.5/`。Embedding 在 CPU 上运行，完成下载后，服务只读取本地模型和索引，不调用 LLM 或云端接口。LangChain 的远程追踪在此工作流中关闭。

知识库只读取 `data/raw/train.txt`。建库时清理空标题、重复标题和标签冲突，再按每类最多 200 条、随机种子 42 抽样。使用全量合格样本可执行：

```bash
python -m src build-index --max-per-class 0
```

索引保存在 `data/indexes/products/`，包含版本指针、FAISS 文件和 JSON 元数据，不使用 pickle。重建失败时保留原索引；以前的完整版本保留在本地目录中。模型、标签、数据指纹在加载索引时核对，发生不匹配需要重新建库。索引和模型不提交到 Git。

重启服务并勾选页面上的“显示相似商品”，即可展开参考商品、原标签、余弦相似度和来源编号。默认最多 3 条，排除与输入完全相同的标题。相似度不是 BERT 的分类置信分数，参考商品不改变分类结果。

这部分将分类和检索作为两个工具，用 LangChain LCEL 并行执行，再合并结果：

```text
输入校验 → RunnableParallel ── BERT 分类工具 ──┐
                           └─ 商品检索工具 ──┴→ 按输入顺序合并
```

可以从 `src/retrieval/workflow.py` 开始阅读，重点看工具包装、并行节点和结果合并。检索依赖或索引缺失时，原分类功能仍可使用。

服务读取以下进程环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CLASSIFIER_MODEL_PATH` | `ckp/best/` | 微调模型目录 |
| `CLASSIFIER_DEVICE` | 自动选择 | 可设为 `cpu` 或 `cuda` |
| `RETRIEVAL_INDEX_PATH` | `data/indexes/products/` | 服务读取的索引目录 |
| `EMBEDDING_MODEL_PATH` | `models/bge-small-zh-v1.5/` | 服务读取的 embedding 目录 |

## 预测接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 模型状态、设备和类别数 |
| `GET` | `/labels` | 类别列表 |
| `POST` | `/predict` | 商品分类 |
| `GET` | `/retrieval/health` | 检索状态、模型、索引条数和不可用原因 |
| `POST` | `/analyze` | 分类并返回最多 3 条相似商品 |

`POST /predict` 接收 JSON：

```json
{"texts": ["纯牛奶整箱250ml"]}
```

返回格式如下，分数以模型实际输出为准：

```json
{
  "predictions": [
    {"text": "纯牛奶整箱250ml", "label_id": 3, "label": "乳品", "confidence": 0.98}
  ]
}
```

每次最多 32 条，每条最多 2000 个字符。分词后截断到 128 tokens。`confidence` 是 softmax 分数，不是准确率。不合法的输入返回 `422`，推理失败返回 `500`。

`POST /analyze` 使用相同的请求格式，每条分类结果额外包含 `neighbors`：

```json
{
  "predictions": [
    {
      "text": "纯牛奶整箱250ml",
      "label_id": 3,
      "label": "乳品",
      "confidence": 0.98,
      "neighbors": [
        {"source_id": "样本的SHA256编号", "text": "早餐纯牛奶", "label": "乳品", "label_id": 3, "similarity": 0.85}
      ]
    }
  ]
}
```

示例中的分数仅说明格式。参考商品不足 3 条时返回实际数量，没有参考时返回空列表。检索不可用时 `/analyze` 返回 `503`，详细原因可通过 `/retrieval/health` 查询；该健康接口始终以 `200` 返回 `ready` 或 `unavailable` 状态。

也可以在命令行预测：

```bash
python -m src predict "纯牛奶整箱250ml" "男士跑步运动鞋"
```

用 `--model-path` 指定模型目录，`--device cpu` 指定 CPU。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

测试使用固定向量、小模型和替代预测器，不下载模型。安装检索依赖后会运行索引、工作流和检索 API 测试；未安装时跳过依赖 FAISS 的测试。无 GPU 时跳过 CUDA 测试。
