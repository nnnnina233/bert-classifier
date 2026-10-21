"""项目命令行入口。"""
import argparse
import json

from src.configuration.config import INFERENCE_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="BERT 商品标题分类：预处理、训练、评估与 API 服务")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preprocess", help="预处理 data/raw 下的数据")
    commands.add_parser("train", help="训练模型，自动恢复已有检查点")
    indexing = commands.add_parser("build-index", help="构建本地商品向量索引")
    indexing.add_argument("--max-per-class", type=int, default=200, help="每类最多抽取的样本数；0 表示全量")
    evaluation = commands.add_parser("evaluate", help="评估已训练模型")
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--model-path", default=str(INFERENCE_MODEL_PATH))
    prediction = commands.add_parser("predict", help="命令行预测商品标题")
    prediction.add_argument("texts", nargs="+")
    prediction.add_argument("--model-path", default=str(INFERENCE_MODEL_PATH))
    prediction.add_argument("--device", default=None, help="cpu 或 cuda；默认自动选择")
    server = commands.add_parser("serve", help="启动 FastAPI，默认监听 127.0.0.1:8000")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--reload", action="store_true", help="开发时自动重载")
    args = parser.parse_args()

    if args.command == "build-index":
        from src.retrieval.index import build_index
        if args.max_per_class < 0:
            parser.error("--max-per-class 不能小于 0")
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        build_index(max_per_class=args.max_per_class)
    elif args.command == "preprocess":
        from src.process.preprocess import preprocess
        preprocess()
    elif args.command == "train":
        from src.runner.train import train
        train()
    elif args.command == "evaluate":
        from src.runner.evaluate import evaluate
        evaluate(model_path=args.model_path, split=args.split)
    elif args.command == "predict":
        from src.runner.predict import predict
        print(json.dumps(predict(args.texts, args.model_path, args.device), ensure_ascii=False, indent=2))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("src.web.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
