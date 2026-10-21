import contextlib
import importlib.util
import io
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.retrieval.corpus import prepare_corpus, read_labels
from src.retrieval.index import (
    QUERY_PROMPT, ProductRetriever, RetrievalUnavailable, build_index, current_generation,
)
from src.retrieval.service import RetrievalService
from src.web.main import create_app

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    Embeddings = object

HAS_RETRIEVAL = all(importlib.util.find_spec(name) is not None
                    for name in ("faiss", "langchain_community", "langchain_core"))


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        vectors = {"牛奶": [1., 0.], "酸奶": [0.9, 0.1], "跑步鞋": [0., 1.]}
        result = []
        for text in texts:
            vector = vectors.get(text.removeprefix(QUERY_PROMPT), [1., 0.])
            norm = math.sqrt(sum(x * x for x in vector))
            result.append([x / norm for x in vector])
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(TemporaryDirectory(dir=Path(__file__).parent))
        self.source = Path(self.directory) / "train.tsv"

    def test_cleaning_conflicts_stable_ids_and_sampling(self):
        self.source.write_text("label\ttext_a\n乳品\t 牛奶 \n乳品\t牛奶\n乳品\t\n乳品\t酸奶\n乳品\t冲突\n鞋靴箱包\t冲突\n鞋靴箱包\t跑步鞋\n", encoding="utf-8")
        labels = ["乳品", "鞋靴箱包"]
        records, stats = prepare_corpus(self.source, labels, max_per_class=0)
        self.assertEqual(stats["empty"], 1)
        self.assertEqual(stats["duplicates"], 1)
        self.assertEqual(stats["conflicting_titles"], 1)
        self.assertEqual({record["text"] for record in records}, {"牛奶", "酸奶", "跑步鞋"})
        lines = self.source.read_text(encoding="utf-8").splitlines()
        self.source.write_text(lines[0] + "\n" + "\n".join(reversed(lines[1:])) + "\n", encoding="utf-8")
        reversed_records, _ = prepare_corpus(self.source, labels, max_per_class=0)
        self.assertEqual(records, reversed_records)
        sampled, _ = prepare_corpus(self.source, labels, max_per_class=1)
        again, _ = prepare_corpus(self.source, labels, max_per_class=1)
        self.assertEqual(sampled, again)
        self.assertEqual(len(sampled), 2)

    def test_rejects_bad_source_and_labels(self):
        self.source.write_text("wrong\theader\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            prepare_corpus(self.source, ["乳品"])
        self.source.write_text("label\ttext_a\n未知\t商品\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            prepare_corpus(self.source, ["乳品"])
        self.source.write_text("1\t乳品\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_labels(self.source)
        with self.assertRaises(ValueError):
            prepare_corpus(self.source, ["乳品"], max_per_class=-1)


@unittest.skipUnless(HAS_RETRIEVAL, "未安装 requirements-retrieval.txt")
class IndexTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(TemporaryDirectory(dir=Path(__file__).parent))).resolve()
        self.source, self.labels_path = self.folder / "train.tsv", self.folder / "labels.txt"
        self.model, self.index = self.folder / "model", self.folder / "index"
        self.labels = ["乳品", "鞋靴箱包"]
        self.source.write_text("label\ttext_a\n乳品\t牛奶\n乳品\t酸奶\n鞋靴箱包\t跑步鞋\n", encoding="utf-8")
        self.labels_path.write_text("0\t乳品\n1\t鞋靴箱包\n", encoding="utf-8")
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"fixture weights")
        self.embeddings = FixedEmbeddings()
        self.build()

    def build(self, embeddings=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return build_index(self.source, self.index, self.model, self.labels_path,
                               embeddings=embeddings or self.embeddings)

    def load(self, labels=None):
        return ProductRetriever.load(self.index, self.model, labels or self.labels,
                                     self.source, self.labels_path, embeddings=self.embeddings)

    def test_roundtrip_order_scores_and_self_exclusion(self):
        retriever = self.load()
        matches = retriever.search(["牛奶", "跑步鞋"])
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][0]["text"], "酸奶")
        self.assertTrue(all(item["text"] != "牛奶" for item in matches[0]))
        self.assertTrue(all(item["text"] != "跑步鞋" for item in matches[1]))
        self.assertEqual(len(matches[0]), 2)
        self.assertGreater(matches[0][0]["similarity"], matches[0][1]["similarity"])
        self.assertAlmostEqual(matches[0][0]["similarity"], .9 / math.sqrt(.82), places=5)
        self.assertFalse(list(self.index.rglob("*.pkl")))
        self.assertEqual(retriever.count, 3)

    def test_one_document_can_return_no_neighbors(self):
        self.source.write_text("label\ttext_a\n乳品\t牛奶\n", encoding="utf-8")
        self.build()
        self.assertEqual(self.load().search(["牛奶"]), [[]])

    def test_missing_and_invalid_pointer(self):
        with self.assertRaises(RetrievalUnavailable):
            current_generation(self.folder / "missing")
        (self.index / "CURRENT").write_text("../../escape", encoding="ascii")
        with self.assertRaises(RetrievalUnavailable):
            self.load()

    def test_label_and_model_mismatch(self):
        with self.assertRaisesRegex(RetrievalUnavailable, "标签映射"):
            self.load(list(reversed(self.labels)))
        (self.model / "config.json").write_text('{"changed": true}', encoding="utf-8")
        with self.assertRaisesRegex(RetrievalUnavailable, "embedding 模型"):
            self.load()

    def test_changed_source_and_corrupt_documents(self):
        original = self.source.read_bytes()
        self.source.write_bytes(original + "乳品\t奶酪\n".encode())
        with self.assertRaisesRegex(RetrievalUnavailable, "训练集"):
            self.load()
        self.source.write_bytes(original)
        folder = self.index / current_generation(self.index)
        (folder / "documents.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(RetrievalUnavailable, "损坏"):
            self.load()

    def test_changed_label_file_requires_rebuild(self):
        with self.labels_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(RetrievalUnavailable, "标签映射文件"):
            self.load()

    def test_query_dimension_mismatch_requires_rebuild(self):
        retriever = self.load()
        retriever.embeddings.embed_documents = Mock(return_value=[[1., 0., 0.]])
        with self.assertRaisesRegex(RetrievalUnavailable, "维度"):
            retriever.search(["牛奶"])

    def test_failed_rebuild_preserves_previous_index(self):
        generation = current_generation(self.index)
        broken = FixedEmbeddings()
        broken.embed_documents = Mock(side_effect=RuntimeError("interrupted build"))
        with self.assertRaises(RuntimeError):
            self.build(broken)
        self.assertEqual(current_generation(self.index), generation)
        self.assertEqual(self.load().count, 3)
        self.build()
        self.assertNotEqual(current_generation(self.index), generation)
        self.assertEqual(self.load().count, 3)

    def test_workflow_preserves_classifier_and_input_order(self):
        from src.retrieval.workflow import create_workflow
        predictor = Mock()
        predictions = [{"text": "跑步鞋", "label_id": 1, "label": "鞋靴箱包", "confidence": .8},
                       {"text": "牛奶", "label_id": 0, "label": "乳品", "confidence": .9}]
        predictor.predict_details.return_value = predictions
        result = create_workflow(predictor, self.load()).invoke(["跑步鞋", "牛奶"])
        self.assertEqual([{key: value for key, value in item.items() if key != "neighbors"}
                          for item in result["predictions"]], predictions)
        self.assertEqual(result["predictions"][1]["neighbors"][0]["text"], "酸奶")
        predictor.predict_details.assert_called_once_with(["跑步鞋", "牛奶"])


class RetrievalApiTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Mock(labels=["乳品"], device="cpu")
        self.prediction = {"text": "牛奶", "label_id": 0, "label": "乳品", "confidence": .9}
        self.predictor.predict_details.return_value = [self.prediction]
        self.service = Mock()
        self.service.analyze.return_value = {"predictions": [{**self.prediction, "neighbors": []}]}
        self.service.health.return_value = {"status": "ready", "model": "test", "count": 1, "detail": None}
        self.client = self.enterContext(TestClient(create_app(
            lambda: self.predictor, frontend_dir=None, retrieval_factory=lambda predictor: self.service,
        )))

    def test_health_and_analyze(self):
        self.assertEqual(self.client.get("/retrieval/health").json()["status"], "ready")
        response = self.client.post("/analyze", json={"texts": [" 牛奶 "]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predictions"][0]["neighbors"], [])
        self.service.analyze.assert_called_once_with(["牛奶"])

    def test_validation_does_not_call_workflow(self):
        for payload in [{}, {"texts": []}, {"texts": [""]}, {"texts": [" "]}, {"texts": [1]},
                        {"texts": ["a"] * 33}, {"texts": ["a" * 2001]}, {"texts": ["a"], "extra": True}]:
            with self.subTest(payload=str(payload)[:50]):
                self.assertEqual(self.client.post("/analyze", json=payload).status_code, 422)
        self.service.analyze.assert_not_called()

    def test_unavailable_does_not_break_original_prediction(self):
        self.service.analyze.side_effect = RetrievalUnavailable("请执行 python -m src build-index。")
        response = self.client.post("/analyze", json={"texts": ["牛奶"]})
        self.assertEqual(response.status_code, 503)
        self.assertIn("build-index", response.json()["detail"])
        original = self.client.post("/predict", json={"texts": ["牛奶"]})
        self.assertEqual(original.json(), {"predictions": [self.prediction]})

    def test_internal_failure_returns_safe_message(self):
        self.service.analyze.side_effect = RuntimeError("private path")
        with self.assertLogs("src.web.main", level="ERROR"):
            response = self.client.post("/analyze", json={"texts": ["牛奶"]})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private path", response.text)

    def test_service_reports_missing_dependencies_and_bad_model(self):
        service = RetrievalService(self.predictor)
        with patch("src.retrieval.service.current_generation", return_value="x"), \
             patch("src.retrieval.index.ProductRetriever.load", side_effect=ImportError("missing")):
            self.assertEqual(service.health()["status"], "unavailable")
            with self.assertRaises(RetrievalUnavailable):
                service.analyze(["牛奶"])

    @unittest.skipUnless(HAS_RETRIEVAL, "未安装 requirements-retrieval.txt")
    def test_service_reuses_loaded_index_and_reloads_new_generation(self):
        service = RetrievalService(self.predictor)
        retriever = Mock(generation="a", count=3)
        with patch("src.retrieval.service.current_generation", return_value="a") as pointer, \
             patch("src.retrieval.index.ProductRetriever.load", return_value=retriever) as loader, \
             patch("src.retrieval.workflow.create_workflow", return_value=Mock()):
            self.assertEqual(service.health()["count"], 3)
            self.assertEqual(service.health()["status"], "ready")
            self.assertEqual(loader.call_count, 1)
            pointer.return_value = "b"
            loader.return_value = Mock(generation="b", count=4)
            self.assertEqual(service.health()["count"], 4)
            self.assertEqual(loader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
