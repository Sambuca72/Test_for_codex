"""Local document search and reproducible evaluation fixtures."""

from .dataset import Document, Query, load_corpus, load_queries
from .retrieval import METHODS, SearchIndex, SearchResult, create_index

__all__ = [
    "METHODS",
    "Document",
    "Query",
    "SearchIndex",
    "SearchResult",
    "create_index",
    "load_corpus",
    "load_queries",
]
