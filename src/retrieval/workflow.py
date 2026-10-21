from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.tools import tool


def create_workflow(predictor, retriever):
    @tool
    def classify_products(texts: list[str]) -> list[dict]:
        """使用本地 BERT 对商品标题分类。"""
        return predictor.predict_details(texts)

    @tool
    def search_similar_products(texts: list[str]) -> list[list[dict]]:
        """检索训练集中相似的带标签商品。"""
        return retriever.search(texts)

    def merge(result):
        predictions, neighbors = result["predictions"], result["neighbors"]
        if len(predictions) != len(neighbors):
            raise RuntimeError("分类和检索返回的条数不一致。")
        return {"predictions": [{**item, "neighbors": matches} for item, matches in zip(predictions, neighbors)]}

    return RunnableParallel(
        predictions=RunnableLambda(lambda texts: classify_products.invoke({"texts": texts})),
        neighbors=RunnableLambda(lambda texts: search_similar_products.invoke({"texts": texts})),
    ) | RunnableLambda(merge)
