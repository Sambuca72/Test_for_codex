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

    def test_bundled_dev_top_10_is_prefix_of_full_ranking(self):
        documents = load_corpus()
        index = create_index("tfidf", documents)
        for query in load_queries(split="dev"):
            with self.subTest(query=query.id):
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


class MetricTests(unittest.TestCase):
    def test_short_top_k_and_first_relevant_outside_top_k(self):
        relevant = frozenset(("a", "b"))
        self.assertEqual(query_metrics(["x", "a"], ["x", "a"], relevant), (0.1, 0.5, 0.5))
        top = [str(number) for number in range(10)]
        self.assertEqual(query_metrics(top, top + ["b"], relevant), (0.0, 0.0, 1 / 11))
        self.assertEqual(query_metrics([], [], relevant), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
