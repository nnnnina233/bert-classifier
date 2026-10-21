"""商品标题分类 API：python -m uvicorn src.web.main:app。"""
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


def create_app(predictor_factory=None, frontend_dir=ROOT_DIR / "frontend" / "dist"):
    """通过工厂注入预测器，让接口测试不依赖大型模型文件。"""
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
        try:
            yield
        finally:
            app.state.predictor = None

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

    # API 路由优先匹配；没有前端构建时，仍提供 Swagger 入口。
    if frontend_dir is not None and (Path(frontend_dir) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def index():
            return RedirectResponse("/docs")

    return app


app = create_app()
