import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_search import Document, create_index, load_corpus, load_queries
from semantic_search import model


class ModelAssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "cache"
        self.assets = {
            model.MODEL_FILENAME: b"fixed test model",
            model.TOKENIZER_FILENAME: b"fixed test tokenizer",
        }
        manifest = {
            name: (f"test/{name}", len(payload), hashlib.sha256(payload).hexdigest())
            for name, payload in self.assets.items()
        }
        manifest_patch = patch.object(model, "_ARTIFACTS", manifest)
        manifest_patch.start()
        self.addCleanup(manifest_patch.stop)

    def test_prepare_verifies_and_reuses_existing_assets_without_network(self):
        def response(url, *, timeout):
            return io.BytesIO(self.assets[url.rsplit("/", 1)[1]])

        with patch.object(model, "urlopen", side_effect=response) as download:
            self.assertEqual(model.prepare_model(self.directory), self.directory)
            self.assertEqual(download.call_count, len(self.assets))
        with patch.object(model, "urlopen", side_effect=AssertionError("unexpected network")):
            self.assertEqual(model.prepare_model(self.directory), self.directory)
            paths = model.load_model_files(self.directory)
        self.assertEqual({path.name: path.read_bytes() for path in paths}, self.assets)

    def test_local_load_rejects_missing_and_same_size_corrupt_assets(self):
        with patch.object(model, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(model.ModelError, "missing or invalid model asset"):
                model.load_model_files(self.directory)
            self.directory.mkdir()
            for name, payload in self.assets.items():
                (self.directory / name).write_bytes(payload)
            corrupted = self.directory / model.TOKENIZER_FILENAME
            corrupted.write_bytes(b"x" * len(self.assets[corrupted.name]))
            with self.assertRaisesRegex(model.ModelError, r"invalid model asset.*tokenizer\.json"):
                model.load_model_files(self.directory)

    def test_bad_download_does_not_publish_or_leave_temporary_file(self):
        invalid = b"x" * len(self.assets[model.MODEL_FILENAME])
        with patch.object(model, "urlopen", return_value=io.BytesIO(invalid)):
            with self.assertRaisesRegex(model.ModelError, "SHA-256 check"):
                model.prepare_model(self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])


@unittest.skipUnless(
    os.environ.get("SEMANTIC_TEST_MODEL_DIR"),
    "set SEMANTIC_TEST_MODEL_DIR to the verified local model directory for CPU integration tests",
)
class MiniLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = Path(os.environ["SEMANTIC_TEST_MODEL_DIR"])
        cls.documents = (
            Document("a", "Photovoltaic power", "Rooftop panels turn sunlight into electricity."),
            Document("b", "Food preservation", "Freeze sliced loaves to keep the crumb soft."),
            Document("c", "Rainwater storage", "A cistern collects building runoff."),
        )
        with patch.object(model, "urlopen", side_effect=AssertionError("unexpected network")):
            cls.index = create_index("semantic", cls.documents, model_dir=cls.model_dir)

    def test_cpu_paraphrase_scores_and_unit_vectors(self):
        import numpy as np

        self.assertEqual(self.index.encoder.session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(self.index.embeddings.shape, (len(self.documents), 384))
        self.assertEqual(self.index.embeddings.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(self.index.embeddings, axis=1), 1, atol=1e-6)
        ranking = self.index.search("How do I stop my bread from going stale?")
        self.assertEqual(ranking[0].document.id, "b")
        self.assertGreater(ranking[0].score, ranking[1].score)
        self.assertTrue(all(-1 <= result.score <= 1 for result in ranking))

    def test_top_10_matches_full_ranking_on_dev(self):
        documents = load_corpus()
        index = create_index("semantic", documents, model_dir=self.model_dir)
        for query in load_queries(split="dev"):
            with self.subTest(query=query.id):
                ranking = index.search(query.text, k=len(documents))
                self.assertEqual(len(ranking), len(documents))
                first = index.search(query.text, k=10)
                self.assertEqual(first, ranking[:10])
                self.assertEqual(index.search(query.text, k=10), first)

    def test_empty_corpus_blank_query_and_zero_k(self):
        with patch.object(self.index.encoder, "encode_batch", side_effect=AssertionError("unexpected encoding")):
            self.assertEqual(self.index.search(" \t"), [])
            self.assertEqual(self.index.search("bread", k=0), [])
        empty = create_index("semantic", (), model_dir="/missing-semantic-model")
        self.assertEqual(empty.search("bread"), [])

    def test_semantic_cli_with_local_model(self):
        command = [
            sys.executable, "-m", "semantic_search", "search", "--json", "--method",
            "semantic", "--model-dir", str(self.model_dir), "--top-k", "1",
            "How do I stop my bread from going stale?",
        ]
        result, = json.loads(subprocess.check_output(command, text=True))
        self.assertIn(result["id"], {"a03", "a06"})
        self.assertGreater(result["score"], 0)

    def test_semantic_cli_reports_missing_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [
                sys.executable, "-m", "semantic_search", "search", "--method", "semantic",
                "--model-dir", directory, "bread storage",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing or invalid model asset", result.stderr)
        self.assertIn("prepare-model", result.stderr)

    def test_hybrid_top_10_matches_full_ranking_on_dev(self):
        documents = load_corpus()
        index = create_index(
            "hybrid", documents, model_dir=self.model_dir, k1=0.6, b=0.75,
            lexical_weight=0.25, rrf_k=10,
        )
        for query in load_queries(split="dev"):
            with self.subTest(query=query.id):
                ranking = index.search(query.text, k=len(documents))
                self.assertEqual(len(ranking), len(documents))
                self.assertEqual(index.search(query.text, k=10), ranking[:10])

    def test_hybrid_cli_forwards_options_and_returns_expected_scores(self):
        documents = load_corpus()
        index = create_index(
            "hybrid", documents, model_dir=self.model_dir, k1=0.6, b=0.75,
            lexical_weight=0.4, rrf_k=60,
        )
        query = "How do I stop my bread from going stale?"
        command = [
            sys.executable, "-m", "semantic_search", "search", "--json", "--method",
            "hybrid", "--model-dir", str(self.model_dir), "--top-k", "3", "--k1",
            "0.6", "--b", "0.75", "--lexical-weight", "0.4", "--rrf-k", "60", query,
        ]
        expected = index.search(query, k=3)
        output = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual([item["id"] for item in output], [item.document.id for item in expected])
        for observed, result in zip(output, expected):
            self.assertAlmostEqual(observed["score"], result.score, places=12)


if __name__ == "__main__":
    unittest.main()
