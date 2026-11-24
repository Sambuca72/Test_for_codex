"""Verified local assets for the pinned MiniLM CPU model."""

import hashlib
import os
from http.client import HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


_REVISION = "855bcf54d21334332f75db10bdd48970ff0ef547"
_SOURCE = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{_REVISION}"
MODEL_FILENAME = "model_quint8_avx2.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
_ARTIFACTS = {
    MODEL_FILENAME: (
        f"onnx/{MODEL_FILENAME}",
        23_046_789,
        "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
    ),
    TOKENIZER_FILENAME: (
        TOKENIZER_FILENAME,
        466_247,
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    ),
}
_CHUNK_SIZE = 1024 * 1024


class ModelError(RuntimeError):
    """The optional local semantic model cannot be prepared or loaded."""


def _model_directory(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return root / "semantic-search-lab" / f"all-MiniLM-L6-v2-{_REVISION}"


def _valid(path: Path, expected_size: int, expected_hash: str) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != expected_size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest() == expected_hash
    except FileNotFoundError:
        return False


def _download(path: Path, source_path: str, expected_size: int, expected_hash: str) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as target:
            temporary_path = Path(target.name)
            downloaded = 0
            digest = hashlib.sha256()
            with urlopen(f"{_SOURCE}/{source_path}", timeout=30) as response:
                while chunk := response.read(_CHUNK_SIZE):
                    downloaded += len(chunk)
                    if downloaded > expected_size:
                        raise ModelError(f"download exceeded expected size for {path.name}")
                    digest.update(chunk)
                    target.write(chunk)
            if downloaded != expected_size or digest.hexdigest() != expected_hash:
                raise ModelError(f"download failed the size or SHA-256 check for {path.name}")
        temporary_path.replace(path)
    except (OSError, HTTPException) as error:
        raise ModelError(f"could not download {path.name}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_model(model_dir: str | Path | None = None) -> Path:
    """Download the pinned assets if the selected cache lacks verified copies."""
    directory = _model_directory(model_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ModelError(
            f"could not create the model directory {directory}. Use a writable --model-dir: {error}"
        ) from error
    for filename, (source, size, checksum) in _ARTIFACTS.items():
        path = directory / filename
        if not _valid(path, size, checksum):
            _download(path, source, size, checksum)
    return directory


def load_model_files(model_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve and verify assets locally without a network request."""
    directory = _model_directory(model_dir)
    for filename, (_, size, checksum) in _ARTIFACTS.items():
        path = directory / filename
        if not _valid(path, size, checksum):
            raise ModelError(
                f"missing or invalid model asset {path}. "
                "Run `python -m semantic_search prepare-model` with the same --model-dir."
            )
    return directory / MODEL_FILENAME, directory / TOKENIZER_FILENAME
