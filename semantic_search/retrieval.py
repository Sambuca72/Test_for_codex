"""Search indexes and shared ranking primitives."""

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from typing import Iterable, Protocol

from .dataset import Document


METHODS = ("tfidf", "bm25", "semantic")
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


@dataclass(frozen=True, slots=True)
class SearchResult:
    document: Document
    score: float


class SearchIndex(Protocol):
    def search(self, query: str, k: int = 10) -> list[SearchResult]: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold().replace("’", "'"))


def _rank(
    documents: tuple[Document, ...], scores: Iterable[tuple[int, float]], k: int
) -> list[SearchResult]:
    ranked = nsmallest(k, scores, key=lambda item: (-item[1], documents[item[0]].id))
    return [SearchResult(documents[position], score) for position, score in ranked]


def _term_counts(
    documents: tuple[Document, ...],
) -> tuple[list[Counter[str]], Counter[str]]:
    counts = [Counter(tokenize(document.content)) for document in documents]
    document_frequencies = Counter(term for document in counts for term in document)
    return counts, document_frequencies


class TfIdfIndex:
    """Cosine similarity over sparse, smoothly weighted term counts."""

    def __init__(self, documents: tuple[Document, ...]) -> None:
        self.documents = documents
        counts, document_frequencies = _term_counts(documents)
        self.idf = {
            term: math.log((1 + len(documents)) / (1 + frequency)) + 1
            for term, frequency in document_frequencies.items()
        }
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for position, document in enumerate(counts):
            for term, weight in self._weights(document).items():
                self.postings[term].append((position, weight))

    def _weights(self, counts: Counter[str]) -> dict[str, float]:
        weights = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {term: weight / norm for term, weight in weights.items()}

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, query_weight in self._weights(Counter(tokenize(query))).items():
            for position, document_weight in self.postings[term]:
                scores[position] += query_weight * document_weight
        return _rank(self.documents, scores.items(), k)


class BM25Index:
    """Okapi BM25 with positive IDF, linear query TF, and preweighted postings."""

    def __init__(self, documents: tuple[Document, ...], *, k1: float = 1.2, b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 < 0:
            raise ValueError("k1 must be finite and non-negative")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between 0 and 1")

        self.documents = documents
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        counts, document_frequencies = _term_counts(documents)
        if not document_frequencies:
            return
        idf = {
            term: math.log1p((len(documents) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }
        lengths = [sum(document.values()) for document in counts]
        average_length = sum(lengths) / len(documents)
        for position, (document, length) in enumerate(zip(counts, lengths)):
            length_norm = 1 - b + b * length / average_length
            # Divide numerator and denominator by k1 + 1 to keep large finite k1 stable.
            document_norm = (k1 / (k1 + 1)) * length_norm
            for term, count in document.items():
                weight = idf[term] * count / (count / (k1 + 1) + document_norm)
                self.postings[term].append((position, weight))

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, query_count in Counter(tokenize(query)).items():
            for position, weight in self.postings.get(term, ()):
                scores[position] += query_count * weight
        return _rank(self.documents, scores.items(), k)


def create_index(method: str, documents: Iterable[Document], **options) -> SearchIndex:
    """Build a reusable search index over the supplied documents."""
    if method not in METHODS:
        raise ValueError(f"unknown search method {method!r}; choose from {', '.join(METHODS)}")
    corpus = tuple(documents)
    if len({document.id for document in corpus}) != len(corpus):
        raise ValueError("document ids must be unique")
    if method == "tfidf":
        return TfIdfIndex(corpus, **options)
    if method == "bm25":
        return BM25Index(corpus, **options)

    from .model import ModelError

    # Newer ONNX Runtime builds read their telemetry opt-out at import time.
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    try:
        from .semantic import SemanticIndex
    except ImportError as error:
        raise ModelError("semantic search requires `python -m pip install '.[semantic]'`") from error
    return SemanticIndex(corpus, **options)
