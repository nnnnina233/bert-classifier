"""商品分类 API。"""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.configuration.config import INFERENCE_MODEL_PATH, ROOT_DIR
from src.runner.predict import Predictor
from src.retrieval.index import INDEX_PATH, MODEL_PATH, RetrievalUnavailable
from src.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[Text] = Field(
        min_length=1, max_length=32,
        description="1～32 条商品标题；每条最多 2000 字符，分词后截断至 128 tokens。",
        examples=[["纯牛奶整箱250ml", "男士跑步运动鞋"]],
    )


class Prediction(BaseModel):
    text: str
    label_id: int
    label: str
    confidence: float = Field(ge=0, le=1, description="该类别的 softmax 分数，不代表校准后的正确率。")


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    device: str
    num_labels: int


class Neighbor(BaseModel):
    source_id: str
    text: str
    label: str
    label_id: int
    similarity: float = Field(ge=-1, le=1, description="余弦相似度，与分类置信分数不同。")


class AnalyzedPrediction(Prediction):
    neighbors: list[Neighbor]


class AnalyzeResponse(BaseModel):
    predictions: list[AnalyzedPrediction]


class RetrievalHealthResponse(BaseModel):
    status: str
    model: str
    count: int
    detail: str | None


def create_app(predictor_factory=None, frontend_dir=ROOT_DIR / "frontend" / "dist", retrieval_factory=None):
    """创建应用；测试时可传入替代预测器。"""
    @asynccontextmanager
    async def lifespan(app):
        if predictor_factory is not None:
            predictor = predictor_factory()
        else:
            predictor = Predictor.from_pretrained(
                os.getenv("CLASSIFIER_MODEL_PATH") or INFERENCE_MODEL_PATH,
                device=os.getenv("CLASSIFIER_DEVICE") or None,
            )
        app.state.predictor = predictor
        app.state.retrieval = (
            retrieval_factory(predictor) if retrieval_factory else RetrievalService(
                predictor,
                index_path=os.getenv("RETRIEVAL_INDEX_PATH") or INDEX_PATH,
                model_path=os.getenv("EMBEDDING_MODEL_PATH") or MODEL_PATH,
            )
        )
        try:
            yield
        finally:
            app.state.predictor = None
            app.state.retrieval = None

    app = FastAPI(
        title="BERT 商品标题分类 API", version="1.0.0", lifespan=lifespan,
        description="将商品标题分类到 30 个类别。启动时加载本地微调模型，支持单条和批量预测。",
    )

    def get_predictor(request):
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(status_code=503, detail="模型尚未就绪。")
        return predictor

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        predictor = get_predictor(request)
        return {"status": "ok", "device": str(predictor.device),
                "num_labels": len(predictor.labels)}

    @app.get("/labels", response_model=dict[str, list[str]])
    def labels(request: Request):
        return {"labels": get_predictor(request).labels}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request):
        predictor = get_predictor(request)
        try:
            return {"predictions": predictor.predict_details(payload.texts)}
        except Exception:
            logger.exception("商品分类推理失败")
            raise HTTPException(status_code=500, detail="预测失败，请检查服务日志。") from None

    @app.get("/retrieval/health", response_model=RetrievalHealthResponse)
    def retrieval_health(request: Request):
        get_predictor(request)
        return request.app.state.retrieval.health()

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(payload: PredictRequest, request: Request):
        get_predictor(request)
        try:
            return request.app.state.retrieval.analyze(payload.texts)
        except RetrievalUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from None
        except Exception:
            logger.exception("分类检索失败")
            raise HTTPException(status_code=500, detail="分类检索失败，请检查服务日志。") from None

    # 静态文件挂载在 API 路由之后。
    if frontend_dir is not None and (Path(frontend_dir) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def index():
            return RedirectResponse("/docs")

    return app


app = create_app()
