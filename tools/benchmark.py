"""Compare local retrieval methods with a common corpus and relevance labels.

Each configuration runs in a new Python process. Mean latency covers a warmed
top-10 search, including query encoding but excluding index construction, I/O
and metrics. Peak RSS is the Linux process high-water mark through the timed
searches, including the interpreter, corpus, libraries, model and index.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from statistics import fmean
from time import perf_counter_ns

from semantic_search import METHODS, create_index, load_corpus, load_queries
from semantic_search.model import ModelError


K = 10
FIELDS = (
    "split", "method", "options", "documents", "queries", "warmup_passes",
    "timed_passes", "precision_at_10", "recall_at_10", "mrr", "mean_search_ms",
    "peak_rss_mib",
)


def query_metrics(top_ids, ranked_ids, relevant_ids):
    hits = len(relevant_ids.intersection(top_ids[:K]))
    reciprocal_rank = next(
        (1 / rank for rank, doc_id in enumerate(ranked_ids, 1) if doc_id in relevant_ids),
        0.0,
    )
    return hits / K, hits / len(relevant_ids), reciprocal_rank


def run(config, args):
    import resource

    documents = load_corpus(args.corpus)
    queries = load_queries(args.queries, split=args.split)
    if not documents or not queries:
        raise ValueError("evaluation requires documents and queries in the chosen split")
    ids = {document.id for document in documents}
    for query in queries:
        if unknown := query.relevant_ids - ids:
            raise ValueError(f"{query.id}: unknown relevant document ids {sorted(unknown)}")

    index = create_index(config["method"], documents, **config["options"])
    for _ in range(args.warmup):
        for query in queries:
            index.search(query.text, k=K)

    timings = []
    for _ in range(args.repeat):
        for query in queries:
            start = perf_counter_ns()
            result = index.search(query.text, k=K)
            timings.append(perf_counter_ns() - start)
            del result
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    scores = []
    for query in queries:
        ranked_ids = [
            result.document.id for result in index.search(query.text, k=len(documents))
        ]
        scores.append(query_metrics(ranked_ids[:K], ranked_ids, query.relevant_ids))
    precision, recall, mrr = map(fmean, zip(*scores))
    return {
        "split": args.split,
        "method": config["method"],
        "options": json.dumps(config["options"], sort_keys=True, separators=(",", ":")),
        "documents": len(documents),
        "queries": len(queries),
        "warmup_passes": args.warmup,
        "timed_passes": args.repeat,
        "precision_at_10": f"{precision:.8f}",
        "recall_at_10": f"{recall:.8f}",
        "mrr": f"{mrr:.8f}",
        "mean_search_ms": f"{fmean(timings) / 1_000_000:.6f}",
        "peak_rss_mib": f"{peak_rss_mib:.6f}",
    }


def parse_config(value):
    try:
        config = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(config, dict) or config.keys() - {"method", "options"}:
        raise argparse.ArgumentTypeError("expected an object with method and optional options")
    if config.get("method") not in METHODS:
        raise argparse.ArgumentTypeError(f"method must be one of {', '.join(METHODS)}")
    options = config.setdefault("options", {})
    if not isinstance(options, dict):
        raise argparse.ArgumentTypeError("options must be a JSON object")
    return config


def run_isolated(config, args):
    command = [
        sys.executable, "-m", "tools.benchmark", "--worker", json.dumps(config),
        "--split", args.split, "--repeat", str(args.repeat), "--warmup", str(args.warmup),
    ]
    for option in ("corpus", "queries"):
        if (path := getattr(args, option)) is not None:
            command.extend((f"--{option}", path))
    environment = os.environ.copy()
    environment.update(
        PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    output = subprocess.check_output(command, env=environment, text=True)
    return json.loads(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--run", type=parse_config, action="append", metavar="JSON")
    parser.add_argument("--corpus", help="optional JSONL document file")
    parser.add_argument("--queries", help="optional JSONL query file")
    parser.add_argument("--repeat", type=int, default=20, help="timed passes over the split")
    parser.add_argument("--warmup", type=int, default=1, help="warm-up passes over the split")
    parser.add_argument("--worker", type=parse_config, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("peak RSS is measured with Linux ru_maxrss")
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be at least 1 and warmup must be nonnegative")
    try:
        if args.worker:
            print(json.dumps(run(args.worker, args)))
            return
        configs = [{"method": method, "options": {}} for method in args.method or ()]
        configs.extend(args.run or ())
        if not configs:
            parser.error("supply at least one --method or --run")
        rows = [run_isolated(config, args) for config in configs]
    except (OSError, TypeError, ValueError, ModelError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    writer = csv.DictWriter(sys.stdout, FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
