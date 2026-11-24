import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from semantic_search import Document, create_index, load_corpus, load_queries
from tools.benchmark import query_metrics


def result_ids(results):
    return [result.document.id for result in results]


class DatasetTests(unittest.TestCase):
    def test_bundled_splits_have_distinct_queries_and_valid_judgments(self):
        documents = load_corpus()
        dev = load_queries(split="dev")
        test = load_queries(split="test")
        self.assertGreaterEqual(len(documents), 10)
        self.assertTrue(dev and test)
        document_ids = {document.id for document in documents}
        self.assertEqual(len(document_ids), len(documents))
        self.assertEqual(len({query.id for query in dev + test}), len(dev) + len(test))
        self.assertEqual(set(dev + test), set(load_queries()))
        normalized = {" ".join(query.text.casefold().split()) for query in dev + test}
        self.assertEqual(len(normalized), len(dev) + len(test))
        for query in dev + test:
            with self.subTest(query=query.id):
                self.assertTrue(query.relevant_ids)
                self.assertLessEqual(query.relevant_ids, document_ids)

    def test_duplicate_ids_report_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            row = json.dumps({"id": "one", "title": "Title", "text": "Text"})
            path.write_text(f"{row}\n\n{row}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":3:.*duplicate id"):
                load_corpus(path)

    def test_invalid_query_judgments_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            row = {"id": "one", "text": "Question", "split": "dev", "relevant_ids": ["x", "x"]}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":1:.*relevant_ids must be unique"):
                load_queries(path)


class TfIdfTests(unittest.TestCase):
    def test_score_matches_smoothed_idf_and_cosine(self):
        corpus = (
            Document("a", "apple banana", "banana"),
            Document("b", "banana", "cherry"),
        )
        result, = create_index("tfidf", corpus).search("APPLE unknown")
        idf = 1 + math.log(3 / 2)
        self.assertEqual(result.document.id, "a")
        self.assertAlmostEqual(result.score, idf / math.sqrt(idf**2 + 2**2), places=12)

    def test_document_ids_break_exact_ties_and_top_k_is_prefix(self):
        corpus = (
            Document("z", "Cycling", "Derailleur cable"),
            Document("a", "Cycling", "Derailleur cable"),
        )
        index = create_index("tfidf", iter(corpus))
        ranking = index.search("cycling", k=len(corpus))
        self.assertEqual(result_ids(ranking), ["a", "z"])
        self.assertEqual(index.search("cycling", k=1), ranking[:1])
        self.assertEqual(index.search("cycling", k=0), [])

    def test_empty_and_out_of_vocabulary_queries(self):
        index = create_index("tfidf", (Document("a", "Apple", "Fruits"),))
        for query in ("", "...", "bicycle"):
            with self.subTest(query=query):
                self.assertEqual(index.search(query), [])
        self.assertEqual(create_index("tfidf", ()).search("apple"), [])

    def test_duplicate_documents_rejected_through_api(self):
        document = Document("a", "Apple", "Fruits")
        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            create_index("tfidf", (document, document))


class BM25Tests(unittest.TestCase):
    def test_score_matches_idf_saturation_and_length_normalization(self):
        corpus = (
            Document("a", "apple", "apple pear"),
            Document("b", "apple", "pear pear pear pear"),
            Document("c", "pear", "pear pear pear"),
        )
        k1, b = 1.8, 0.25
        index = create_index("bm25", corpus, k1=k1, b=b)
        ranking = index.search("APPLE unknown")
        idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        average_length = (3 + 5 + 4) / 3
        self.assertEqual(result_ids(ranking), ["a", "b"])
        for result, frequency, length in zip(ranking, (2, 1), (3, 5)):
            length_norm = 1 - b + b * length / average_length
            expected = idf * frequency * (k1 + 1) / (frequency + k1 * length_norm)
            self.assertAlmostEqual(result.score, expected, places=12)

    def test_repeated_query_term_has_linear_weight(self):
        corpus = (Document("a", "Apple", "Fruits"), Document("b", "Apple", "Apple pie"))
        index = create_index("bm25", corpus)
        single = index.search("apple")
        repeated = index.search("APPLE apple")
        self.assertEqual(result_ids(repeated), result_ids(single))
        for original, doubled in zip(single, repeated):
            self.assertAlmostEqual(doubled.score, 2 * original.score, places=12)

    def test_parameter_endpoints_and_large_k1(self):
        corpus = (
            Document("a", "apple", "apple filler filler filler"),
            Document("z", "apple", ""),
        )
        no_saturation = create_index("bm25", corpus, b=1, k1=0).search("apple")
        self.assertEqual(result_ids(no_saturation), ["a", "z"])
        self.assertEqual(no_saturation[0].score, no_saturation[1].score)
        no_length_norm = create_index("bm25", corpus, b=0).search("apple")
        self.assertEqual(result_ids(no_length_norm), ["a", "z"])

        full_length_norm = create_index("bm25", corpus, b=1, k1=1e308).search("apple")
        self.assertEqual(result_ids(full_length_norm), ["z", "a"])
        idf = math.log(1 + 0.5 / 2.5)
        self.assertAlmostEqual(full_length_norm[0].score, idf * 3, places=12)
        self.assertAlmostEqual(full_length_norm[1].score, idf * 2 * 3 / 5, places=12)

    def test_invalid_parameters_are_rejected(self):
        for name, values in (
            ("k1", (-1, math.inf, -math.inf, math.nan)),
            ("b", (-0.01, 1.01, math.inf, -math.inf, math.nan)),
        ):
            for value in values:
                with self.subTest(parameter=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        create_index("bm25", (), **{name: value})

    def test_empty_corpus_and_empty_tokenized_document(self):
        self.assertEqual(create_index("bm25", ()).search("apple"), [])
        self.assertEqual(create_index("bm25", (Document("a", "...", ""),)).search("apple"), [])


class RankingTests(unittest.TestCase):
    def test_bundled_dev_top_10_is_prefix_of_full_ranking(self):
        documents = load_corpus()
        queries = load_queries(split="dev")
        configs = (("tfidf", {}), ("bm25", {}), ("bm25", {"k1": 0.6, "b": 0.75}))
        for method, options in configs:
            index = create_index(method, documents, **options)
            for query in queries:
                with self.subTest(method=method, options=options, query=query.id):
                    ranking = index.search(query.text, k=len(documents))
                    self.assertEqual(index.search(query.text, k=10), ranking[:10])


class CliTests(unittest.TestCase):
    def test_json_search_over_external_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            docs = [
                {"id": "a", "title": "Apple", "text": "Fruit"},
                {"id": "b", "title": "Cycling", "text": "Derailleur"},
            ]
            path.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
            command = [
                sys.executable, "-m", "semantic_search", "search", "--corpus", str(path),
                "--method", "tfidf", "--top-k", "1", "--json", "derailleur",
            ]
            output = json.loads(subprocess.check_output(command, text=True))
        result, = output
        self.assertEqual((result["id"], result["title"]), ("b", "Cycling"))
        self.assertGreater(result["score"], 0)

    def test_bm25_cli_uses_options(self):
        text = "bicycle brake pads"
        command = [
            sys.executable, "-m", "semantic_search", "search", "--json", "--method",
            "bm25", "--k1", "0", "--b", "1", "--top-k", "3", text,
        ]
        observed = json.loads(subprocess.check_output(command, text=True))
        index = create_index("bm25", load_corpus(), k1=0, b=1)
        expected = index.search(text, k=3)
        self.assertEqual([item["id"] for item in observed], result_ids(expected))
        self.assertEqual([item["score"] for item in observed], [item.score for item in expected])

    def test_tf_idf_cli_rejects_bm25_options(self):
        command = [sys.executable, "-m", "semantic_search", "search", "--b", "0", "brake"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("require --method bm25", result.stderr)


class MetricTests(unittest.TestCase):
    def test_short_top_k_and_first_relevant_outside_top_k(self):
        relevant = frozenset(("a", "b"))
        self.assertEqual(query_metrics(["x", "a"], ["x", "a"], relevant), (0.1, 0.5, 0.5))
        top = [str(number) for number in range(10)]
        self.assertEqual(query_metrics(top, top + ["b"], relevant), (0.0, 0.0, 1 / 11))
        self.assertEqual(query_metrics([], [], relevant), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
