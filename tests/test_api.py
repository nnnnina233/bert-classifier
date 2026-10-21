import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.web.main import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Mock(device="cpu", labels=["乳品", "鞋靴箱包"])
        self.predictor.predict_details.side_effect = lambda texts: [
            {"text": text, "label_id": 0, "label": "乳品", "confidence": 0.9}
            for text in texts
        ]
        self.factory = Mock(return_value=self.predictor)
        self.app = create_app(self.factory, frontend_dir=None)
        self.client = self.enterContext(TestClient(self.app))

    def test_model_loaded_once_and_cleaned_up(self):
        self.client.get("/health")
        self.client.post("/predict", json={"texts": ["牛奶"]})
        self.client.post("/predict", json={"texts": ["酸奶"]})
        self.factory.assert_called_once_with()
        self.client.close()
        # __exit__ 负责执行 lifespan 清理。
        self.doCleanups()
        self.assertIsNone(self.app.state.predictor)

    def test_health_labels_and_docs(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "device": "cpu", "num_labels": 2})
        self.assertEqual(self.client.get("/labels").json(), {"labels": ["乳品", "鞋靴箱包"]})
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertEqual(self.client.get("/", follow_redirects=False).headers["location"], "/docs")
        self.assertIn("/predict", self.client.get("/openapi.json").json()["paths"])

    def test_batch_preserves_order_and_strips_whitespace(self):
        response = self.client.post("/predict", json={"texts": [" 牛奶 ", "酸奶"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([p["text"] for p in response.json()["predictions"]], ["牛奶", "酸奶"])
        self.predictor.predict_details.assert_called_once_with(["牛奶", "酸奶"])

    def test_frontend_and_api_share_one_server(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root / "index.html").write_text("<html>Classifier UI</html>", encoding="utf-8")
            (root / "assets").mkdir()
            (root / "assets/app.js").write_text("console.log('app')", encoding="utf-8")
            with TestClient(create_app(self.factory, frontend_dir=root)) as client:
                response = client.get("/")
                self.assertEqual(response.status_code, 200)
                self.assertIn("Classifier UI", response.text)
                self.assertEqual(client.get("/assets/app.js").status_code, 200)
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/docs").status_code, 200)
                self.assertEqual(client.post("/predict", json={"texts": ["牛奶"]}).status_code, 200)
                self.assertEqual(client.get("/missing-file.js").status_code, 404)

    def test_invalid_inputs_rejected_before_inference(self):
        for payload in [
            {}, {"texts": []}, {"texts": [""]}, {"texts": ["  "]},
            {"texts": [3]}, {"texts": [None]}, {"texts": "牛奶"},
            {"texts": ["牛奶"] * 33}, {"texts": ["奶" * 2001]},
            {"texts": ["牛奶"], "unexpected": True},
        ]:
            with self.subTest(payload=str(payload)[:60]):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)
        self.predictor.predict_details.assert_not_called()

    def test_missing_lifespan_returns_503(self):
        with TestClient(create_app(lambda: None)) as client:
            self.assertEqual(client.get("/health").status_code, 503)
            self.assertEqual(client.post("/predict", json={"texts": ["牛奶"]}).status_code, 503)

    def test_startup_failure_is_not_hidden(self):
        factory = Mock(side_effect=FileNotFoundError("missing model"))
        with self.assertRaises(FileNotFoundError):
            with TestClient(create_app(factory)):
                pass

    def test_inference_failure_does_not_expose_internal_error(self):
        self.predictor.predict_details.side_effect = RuntimeError("internal model path")
        with self.assertLogs("src.web.main", level="ERROR"):
            response = self.client.post("/predict", json={"texts": ["牛奶"]})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("internal model path", response.text)


if __name__ == "__main__":
    unittest.main()
