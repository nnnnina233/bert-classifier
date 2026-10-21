import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from src.configuration.config import ROOT_DIR, RAW_TRAIN_DATA_PATH
from src.retrieval.corpus import file_hash, prepare_corpus, read_labels

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_PATH = ROOT_DIR / "models" / "bge-small-zh-v1.5"
INDEX_PATH = ROOT_DIR / "data" / "indexes" / "products"
LABEL_PATH = ROOT_DIR / "label_mapping.txt"
QUERY_PROMPT = "为这个句子生成表示以用于检索相关文章："


class RetrievalUnavailable(RuntimeError):
    pass


def dependencies():
    try:
        import faiss
        import numpy as np
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy
        from langchain_core.documents import Document
    except ImportError as error:
        raise RetrievalUnavailable("缺少检索依赖，请执行 python -m pip install -r requirements-retrieval.txt。") from error
    return faiss, np, InMemoryDocstore, FAISS, DistanceStrategy, Document


def model_fingerprint(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        raise RetrievalUnavailable("缺少本地 embedding 模型，请执行 python -m src build-index。")
    files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.relative_to(path).parts)
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise RetrievalUnavailable("embedding 模型权重不完整，请重新执行 python -m src build-index。")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(file_hash(item).encode())
    return digest.hexdigest()


def load_embeddings(model_path, progress=False):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={"device": "cpu", "local_files_only": True, "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_PROMPT},
            show_progress=progress,
        )
    except ImportError as error:
        raise RetrievalUnavailable("缺少 embedding 依赖，请安装 requirements-retrieval.txt。") from error


def download_model(model_path):
    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(MODEL_ID)
    weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    snapshot_download(
        repo_id=MODEL_ID, local_dir=str(model_path),
        allow_patterns=["*.json", "*.txt", weight, "1_Pooling/*"],
        ignore_patterns=["onnx/*", "openvino/*"],
    )


def build_index(source=RAW_TRAIN_DATA_PATH, index_path=INDEX_PATH, model_path=MODEL_PATH,
                label_path=LABEL_PATH, max_per_class=200, seed=42, embeddings=None):
    faiss, np, Store, FAISS, Strategy, Document = dependencies()
    labels = read_labels(label_path)
    label_hash = file_hash(label_path)
    source_hash = file_hash(source)
    records, stats = prepare_corpus(source, labels, max_per_class, seed)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    if embeddings is None:
        try:
            fingerprint = model_fingerprint(model_path)
        except RetrievalUnavailable:
            print(f"下载 embedding 模型：{MODEL_ID}", flush=True)
            download_model(model_path)
            fingerprint = model_fingerprint(model_path)
        embeddings = load_embeddings(model_path, progress=True)
    else:
        fingerprint = model_fingerprint(model_path)
    documents = [Document(page_content=item["text"], metadata=item) for item in records]
    vectorstore = FAISS.from_documents(documents, embeddings, distance_strategy=Strategy.MAX_INNER_PRODUCT)
    if file_hash(source) != source_hash:
        raise ValueError("建库期间训练集发生变化，请重新建库。")
    if file_hash(label_path) != label_hash:
        raise ValueError("建库期间标签映射发生变化，请重新建库。")
    index_path = Path(index_path).resolve()
    index_path.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex
    with TemporaryDirectory(prefix=".building-", dir=index_path) as folder:
        temporary = Path(folder).resolve()
        assert temporary.parent == index_path
        (temporary / "index.faiss").write_bytes(faiss.serialize_index(vectorstore.index).tobytes())
        (temporary / "documents.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        manifest = {
            "version": 1, "model": MODEL_ID, "model_fingerprint": fingerprint,
            "query_prompt": QUERY_PROMPT, "normalized": True,
            "dimension": vectorstore.index.d, "count": len(records), "labels": labels,
            "source_sha256": source_hash, "label_sha256": label_hash,
            "max_per_class": max_per_class, "seed": seed, "stats": stats,
            "index_sha256": file_hash(temporary / "index.faiss"),
            "documents_sha256": file_hash(temporary / "documents.json"),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.rename(index_path / generation)
    # 只替换小型指针文件，读者始终看到一个完整版本。
    pointer = index_path / f".CURRENT-{generation}.tmp"
    pointer.write_text(generation, encoding="ascii")
    pointer.replace(index_path / "CURRENT")
    print(f"索引已保存：{index_path}（{len(records)} 条）", flush=True)
    return manifest


def current_generation(index_path):
    try:
        generation = (Path(index_path) / "CURRENT").read_text(encoding="ascii").strip()
    except OSError as error:
        raise RetrievalUnavailable("尚未建立商品索引，请执行 python -m src build-index。") from error
    if not re.fullmatch(r"[a-f0-9]{32}", generation):
        raise RetrievalUnavailable("索引版本指针损坏，请执行 python -m src build-index。")
    return generation


class ProductRetriever:
    def __init__(self, vectorstore, embeddings, count, generation):
        self.vectorstore, self.embeddings = vectorstore, embeddings
        self.count, self.generation = count, generation

    @classmethod
    def load(cls, index_path, model_path, labels, source=RAW_TRAIN_DATA_PATH,
             label_path=LABEL_PATH, embeddings=None):
        generation = current_generation(index_path)
        folder = Path(index_path) / generation
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["version"] != 1 or manifest["labels"] != labels
                or manifest["model"] != MODEL_ID or manifest["query_prompt"] != QUERY_PROMPT
                or manifest["normalized"] is not True):
            raise RetrievalUnavailable("索引配置或标签映射不匹配，请重新建库。")
        if manifest["model_fingerprint"] != model_fingerprint(model_path):
            raise RetrievalUnavailable("embedding 模型与索引不匹配，请重新建库。")
        if Path(label_path).is_file() and manifest["label_sha256"] != file_hash(label_path):
            raise RetrievalUnavailable("标签映射文件已变化，请重新建库。")
        if Path(source).is_file() and manifest["source_sha256"] != file_hash(source):
            raise RetrievalUnavailable("训练集已变化，请重新建库。")
        if (manifest["index_sha256"] != file_hash(folder / "index.faiss")
                or manifest["documents_sha256"] != file_hash(folder / "documents.json")):
            raise RetrievalUnavailable("索引文件损坏，请重新建库。")
        faiss, np, Store, FAISS, Strategy, Document = dependencies()
        records = json.loads((folder / "documents.json").read_text(encoding="utf-8"))
        index = faiss.deserialize_index(np.frombuffer((folder / "index.faiss").read_bytes(), dtype="uint8"))
        if index.ntotal != len(records) or index.ntotal != manifest["count"] or index.d != manifest["dimension"]:
            raise RetrievalUnavailable("索引条数或向量维度不匹配，请重新建库。")
        documents = {}
        for item in records:
            if labels[item["label_id"]] != item["label"] or item["source_id"] in documents:
                raise RetrievalUnavailable("商品元数据与标签映射不匹配，请重新建库。")
            documents[item["source_id"]] = Document(page_content=item["text"], metadata=item)
        embeddings = embeddings or load_embeddings(model_path)
        store = FAISS(embeddings, index, Store(documents), {i: item["source_id"] for i, item in enumerate(records)},
                      distance_strategy=Strategy.MAX_INNER_PRODUCT)
        return cls(store, embeddings, len(records), generation)

    def search(self, texts, top_k=3):
        import numpy as np
        vectors = np.asarray(self.embeddings.embed_documents([QUERY_PROMPT + text for text in texts]), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.vectorstore.index.d:
            raise RetrievalUnavailable("embedding 输出维度与索引不匹配，请重新建库。")
        # 额外取一条，过滤输入标题自身后仍能保留 top_k 个参考。
        scores, indices = self.vectorstore.index.search(vectors, min(top_k + 1, self.count))
        result = []
        for text, row_scores, row_indices in zip(texts, scores, indices):
            neighbors = []
            for score, index in zip(row_scores, row_indices):
                if index < 0:
                    continue
                doc_id = self.vectorstore.index_to_docstore_id[int(index)]
                doc = self.vectorstore.docstore.search(doc_id)
                if doc.page_content == text.strip():
                    continue
                neighbors.append({**doc.metadata, "similarity": float(np.clip(score, -1, 1))})
                if len(neighbors) == top_k:
                    break
            result.append(neighbors)
        return result
