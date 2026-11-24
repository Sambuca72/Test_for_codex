import json
import math
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from semantic_search import Document, SearchResult, create_index, load_corpus


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.documents = tuple(Document(name, name, "term") for name in "dbca")
        self.docs = {document.id: document for document in self.documents}

    def ranking(self, ids):
        return [SearchResult(self.docs[key], -100.0 * ordinal) for ordinal, key in enumerate(ids)]

    def test_weighted_1_based_ranks_and_absent_contributions(self):
        lexical = Mock()
        semantic = Mock()
        lexical.search.return_value = self.ranking("ab")
        semantic.search.return_value = self.ranking("ca")
        with (
            patch("semantic_search.retrieval.BM25Index", return_value=lexical),
            patch("semantic_search.retrieval._semantic_index", return_value=semantic),
        ):
            index = create_index("hybrid", self.documents, lexical_weight=0.25, rrf_k=10)
        first_two = index.search("query", k=2)
        lexical.search.assert_called_once_with("query", k=len(self.documents))
        semantic.search.assert_called_once_with("query", k=len(self.documents))
        ranking = index.search("query", k=len(self.documents))
        self.assertEqual([result.document.id for result in ranking], ["a", "c", "b"])
        self.assertEqual(first_two, ranking[:2])
        expected_scores = (0.25 / 11 + 0.75 / 12, 0.75 / 11, 0.25 / 12)
        for result, expected in zip(ranking, expected_scores):
            self.assertAlmostEqual(result.score, expected, places=12)

    def test_ties_are_broken_by_document_id(self):
        lexical = Mock()
        semantic = Mock()
        lexical.search.return_value = self.ranking("ba")
        semantic.search.return_value = self.ranking("ab")
        with (
            patch("semantic_search.retrieval.BM25Index", return_value=lexical),
            patch("semantic_search.retrieval._semantic_index", return_value=semantic),
        ):
            index = create_index("hybrid", self.documents, lexical_weight=0.5, rrf_k=0)
        ranking = index.search("query", k=2)
        self.assertEqual([result.document.id for result in ranking], ["a", "b"])
        self.assertEqual([result.score for result in ranking], [0.75, 0.75])

    def test_lexical_endpoint_does_not_load_semantic_index(self):
        with patch("semantic_search.retrieval._semantic_index") as semantic:
            index = create_index("hybrid", self.documents, lexical_weight=1, rrf_k=0)
        pure_lexical = create_index("bm25", self.documents).search("term", k=len(self.documents))
        ranking = index.search("term", k=len(self.documents))
        self.assertEqual([result.document for result in ranking], [result.document for result in pure_lexical])
        self.assertEqual([result.score for result in ranking], [1, 1 / 2, 1 / 3, 1 / 4])
        semantic.assert_not_called()

    def test_semantic_endpoint_never_searches_lexical_index(self):
        lexical = Mock()
        semantic = Mock()
        semantic.search.return_value = self.ranking("cd")
        with (
            patch("semantic_search.retrieval.BM25Index", return_value=lexical),
            patch("semantic_search.retrieval._semantic_index", return_value=semantic),
        ):
            index = create_index("hybrid", self.documents, lexical_weight=0, rrf_k=0)
        self.assertEqual(index.search("  "), [])
        self.assertEqual(index.search("query", k=0), [])
        semantic.search.assert_not_called()
        ranking = index.search("query", k=len(self.documents))
        self.assertEqual([result.document.id for result in ranking], ["c", "d"])
        self.assertEqual([result.score for result in ranking], [1, 1 / 2])
        lexical.search.assert_not_called()

    def test_invalid_hybrid_parameters(self):
        for name, values in (
            ("lexical_weight", (-0.01, 1.01, math.nan, math.inf)),
            ("rrf_k", (-1, 0.5, True, 10**1000)),
        ):
            for value in values:
                with self.subTest(parameter=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        create_index("hybrid", self.documents, **{name: value})

    def test_lexical_endpoint_is_available_in_cli_without_optional_dependencies(self):
        query = "bicycle brake pads"
        command = [
            sys.executable, "-S", "-m", "semantic_search", "search", "--json", "--method",
            "hybrid", "--lexical-weight", "1", "--rrf-k", "10", "--k1", "0.6",
            "--b", "0.75", "--top-k", "3", query,
        ]
        result = json.loads(subprocess.check_output(command, text=True))
        expected = create_index("bm25", load_corpus(), k1=0.6, b=0.75).search(query, k=3)
        self.assertEqual([item["id"] for item in result], [item.document.id for item in expected])
        self.assertEqual([item["score"] for item in result], [1 / 11, 1 / 12, 1 / 13])


if __name__ == "__main__":
    unittest.main()
