"""Local MiniLM embeddings, mean pooling, and exact cosine ranking."""

import platform
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from tokenizers import Tokenizer

from .dataset import Document
from .model import ModelError, load_model_files
from .retrieval import SearchResult, _rank


_DIMENSIONS = 384
_MAX_TOKENS = 256
_BATCH_SIZE = 8


def _require_avx2() -> None:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ModelError("the pinned MiniLM model requires an x86-64 CPU with AVX2")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        flags = next((line for line in cpuinfo.read_text().splitlines() if line.startswith("flags")), "")
        if flags and "avx2" not in flags.split():
            raise ModelError("the pinned MiniLM model requires an x86-64 CPU with AVX2")


class MiniLMEncoder:
    def __init__(self, model_dir: str | Path | None) -> None:
        _require_avx2()
        model_path, tokenizer_path = load_model_files(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as error:
            raise ModelError(f"could not load the MiniLM CPU model: {error}") from error
        self.tokenizer.enable_truncation(max_length=_MAX_TOKENS)
        self.tokenizer.enable_padding()

    def encode_batch(self, texts: Sequence[str]) -> NDArray[np.float32]:
        encodings = self.tokenizer.encode_batch(list(texts))
        attention = np.asarray([item.attention_mask for item in encodings], dtype=np.int64)
        inputs = {
            "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
            "attention_mask": attention,
            "token_type_ids": np.asarray([item.type_ids for item in encodings], dtype=np.int64),
        }
        (hidden_states,) = self.session.run(["last_hidden_state"], inputs)
        mask = attention.astype(np.float32)[..., None]
        means = (hidden_states * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        return means / np.maximum(norms, np.finfo(np.float32).eps)


class SemanticIndex:
    """Float32 index for dense exact retrieval with a single CPU encoder."""

    def __init__(self, documents: tuple[Document, ...], *, model_dir: str | Path | None = None) -> None:
        self.documents = documents
        self.encoder = MiniLMEncoder(model_dir) if documents else None
        self.embeddings = np.empty((len(documents), _DIMENSIONS), dtype=np.float32)
        if self.encoder is not None:
            for start in range(0, len(documents), _BATCH_SIZE):
                stop = min(start + _BATCH_SIZE, len(documents))
                texts = [document.content for document in documents[start:stop]]
                self.embeddings[start:stop] = self.encoder.encode_batch(texts)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if self.encoder is None or k <= 0 or not query.strip():
            return []
        (embedding,) = self.encoder.encode_batch([query])
        scores = self.embeddings @ embedding
        np.clip(scores, -1, 1, out=scores)
        return _rank(self.documents, enumerate(map(float, scores)), k)
