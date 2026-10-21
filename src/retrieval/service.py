from pathlib import Path
from threading import Lock

from src.retrieval.index import (
    INDEX_PATH, MODEL_PATH, MODEL_ID, ProductRetriever, RetrievalUnavailable, current_generation,
)


class RetrievalService:
    def __init__(self, predictor, index_path=INDEX_PATH, model_path=MODEL_PATH):
        self.predictor = predictor
        self.index_path, self.model_path = Path(index_path), Path(model_path)
        self._lock = Lock()
        self._retriever = self._workflow = None

    def ready(self):
        with self._lock:
            generation = current_generation(self.index_path)
            if self._retriever is None or self._retriever.generation != generation:
                try:
                    from src.retrieval.workflow import create_workflow
                    retriever = ProductRetriever.load(self.index_path, self.model_path, self.predictor.labels)
                    workflow = create_workflow(self.predictor, retriever)
                except RetrievalUnavailable:
                    raise
                except ImportError as error:
                    raise RetrievalUnavailable("缺少检索依赖，请安装 requirements-retrieval.txt。") from error
                except (OSError, ValueError, KeyError, IndexError, RuntimeError) as error:
                    raise RetrievalUnavailable("无法加载本地索引或 embedding 模型，请检查文件并重新执行 python -m src build-index。") from error
                self._retriever, self._workflow = retriever, workflow
            return self._workflow

    def health(self):
        try:
            self.ready()
            return {"status": "ready", "model": MODEL_ID, "count": self._retriever.count, "detail": None}
        except RetrievalUnavailable as error:
            return {"status": "unavailable", "model": MODEL_ID, "count": 0, "detail": str(error)}

    def analyze(self, texts):
        workflow = self.ready()
        from langsmith import tracing_context
        with tracing_context(enabled=False):
            return workflow.invoke(texts, config={"callbacks": []})
