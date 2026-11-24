"""Document and query formats for the built-in or external JSONL files."""

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable, TypeVar


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True, slots=True)
class Query:
    id: str
    text: str
    split: str
    relevant_ids: frozenset[str]
    kind: str = ""


def _string(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _document(record: dict) -> Document:
    return Document(_string(record, "id"), _string(record, "title"), _string(record, "text"))


def _query(record: dict) -> Query:
    split = _string(record, "split")
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    relevant = record.get("relevant_ids")
    if not isinstance(relevant, list) or not relevant or any(
        not isinstance(value, str) or not value.strip() for value in relevant
    ):
        raise ValueError("relevant_ids must be a non-empty array of non-empty strings")
    if len(set(relevant)) != len(relevant):
        raise ValueError("relevant_ids must be unique")
    kind = _string(record, "kind") if "kind" in record else ""
    return Query(_string(record, "id"), _string(record, "text"), split, frozenset(relevant), kind)


Record = TypeVar("Record", Document, Query)


def _load(
    path: str | Path | None, filename: str, parse: Callable[[dict], Record]
) -> tuple[Record, ...]:
    source = Path(path) if path is not None else files("semantic_search").joinpath("data", filename)
    records = []
    seen = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("each line must be a JSON object")
                record = parse(data)
                if record.id in seen:
                    raise ValueError(f"duplicate id {record.id!r}")
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            seen.add(record.id)
            records.append(record)
    return tuple(records)


def load_corpus(path: str | Path | None = None) -> tuple[Document, ...]:
    """Read documents from a JSONL path or the bundled corpus."""
    return _load(path, "documents.jsonl", _document)


def load_queries(
    path: str | Path | None = None, split: str | None = None
) -> tuple[Query, ...]:
    """Read queries and their relevance judgments, optionally by split."""
    if split is not None and split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    queries = _load(path, "queries.jsonl", _query)
    return queries if split is None else tuple(query for query in queries if query.split == split)
