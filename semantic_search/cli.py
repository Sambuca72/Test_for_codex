"""Command-line search entry point."""

import argparse
import json

from .dataset import load_corpus
from .model import ModelError, prepare_model
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
    search.add_argument("--k1", type=float, help="BM25 term-frequency saturation (default: 1.2)")
    search.add_argument("--b", type=float, help="BM25 length normalization (default: 0.75)")
    search.add_argument("--lexical-weight", type=float, help="BM25 share of hybrid fusion (default: 0.25)")
    search.add_argument("--rrf-k", type=int, help="hybrid fusion rank offset (default: 10)")
    search.add_argument("--model-dir", help="directory containing the prepared offline model assets")
    search.add_argument("--corpus", help="path to a JSONL corpus; defaults to the bundled data")
    search.add_argument("--top-k", type=_positive_int, default=10, help="maximum number of results")
    search.add_argument("--json", action="store_true", help="emit the result list as JSON")
    prepare = commands.add_parser("prepare-model", help="download and verify the pinned MiniLM model")
    prepare.add_argument("--model-dir", help="target directory; defaults to the system user cache")
    args = parser.parse_args(argv)

    if args.command == "prepare-model":
        try:
            print(prepare_model(args.model_dir))
        except (OSError, ModelError) as error:
            parser.error(str(error))
        return 0

    options = {name: value for name, value in {"k1": args.k1, "b": args.b}.items() if value is not None}
    if options and args.method not in {"bm25", "hybrid"}:
        parser.error("--k1 and --b require --method bm25 or hybrid")
    fusion = {
        name: value
        for name, value in {"lexical_weight": args.lexical_weight, "rrf_k": args.rrf_k}.items()
        if value is not None
    }
    if fusion and args.method != "hybrid":
        parser.error("--lexical-weight and --rrf-k require --method hybrid")
    options.update(fusion)
    if args.model_dir is not None:
        if args.method not in {"semantic", "hybrid"}:
            parser.error("--model-dir requires --method semantic or hybrid")
        options["model_dir"] = args.model_dir
    try:
        index = create_index(args.method, load_corpus(args.corpus), **options)
        results = index.search(args.query, k=args.top_k)
    except (OSError, ValueError, ModelError) as error:
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
