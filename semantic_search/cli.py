"""Command-line search entry point."""

import argparse
import json

from .dataset import load_corpus
from .retrieval import METHODS, create_index


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search a local document corpus.")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="rank documents for a query")
    search.add_argument("query", help="search text")
    search.add_argument("--method", choices=METHODS, default="tfidf")
    search.add_argument("--corpus", help="path to a JSONL corpus; defaults to the bundled data")
    search.add_argument("--top-k", type=_positive_int, default=10, help="maximum number of results")
    search.add_argument("--json", action="store_true", help="emit the result list as JSON")
    args = parser.parse_args(argv)

    try:
        index = create_index(args.method, load_corpus(args.corpus))
        results = index.search(args.query, k=args.top_k)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps([
            {"id": result.document.id, "title": result.document.title, "score": result.score}
            for result in results
        ], ensure_ascii=False))
    else:
        for rank, result in enumerate(results, 1):
            print(f"{rank:>2}. {result.score:.6f}  {result.document.id}  {result.document.title}")
    return 0
