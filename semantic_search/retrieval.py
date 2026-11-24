"""Search indexes and shared ranking primitives."""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from typing import Iterable, Mapping, Protocol

from .dataset import Document


METHODS = ("tfidf",)
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
    documents: tuple[Document, ...], scores: Mapping[int, float], k: int
) -> list[SearchResult]:
    ranked = nsmallest(k, scores, key=lambda position: (-scores[position], documents[position].id))
    return [SearchResult(documents[position], scores[position]) for position in ranked]


class TfIdfIndex:
    """Cosine similarity over sparse, smoothly weighted term counts."""

    def __init__(self, documents: tuple[Document, ...]) -> None:
        self.documents = documents
        counts = [Counter(tokenize(document.content)) for document in documents]
        document_frequencies = Counter(term for document in counts for term in document)
        self.idf = {
            term: math.log((1 + len(documents)) / (1 + frequency)) + 1
            for term, frequency in document_frequencies.items()
        }
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for position, document in enumerate(counts):
            for term, weight in self._weights(document).items():
                self.postings[term].append((position, weight))

    def _weights(self, counts: Counter) -> dict[str, float]:
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
        return _rank(self.documents, scores, k)


def create_index(method: str, documents: Iterable[Document], **options) -> SearchIndex:
    """Build a reusable search index over the supplied documents."""
    if method not in METHODS:
        raise ValueError(f"unknown search method {method!r}; choose from {', '.join(METHODS)}")
    corpus = tuple(documents)
    if len({document.id for document in corpus}) != len(corpus):
        raise ValueError("document ids must be unique")
    return TfIdfIndex(corpus, **options)
