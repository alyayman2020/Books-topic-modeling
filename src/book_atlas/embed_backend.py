"""Embedding backends: local (SentenceTransformer, GPU) or OpenAI API."""
from __future__ import annotations
import os, time, random
import numpy as np
from book_atlas.utils import get_logger, l2_normalize

log = get_logger("embed_backend")


def get_embedder(cfg: dict):
    if cfg["embedding"]["backend"] == "openai":
        return _OpenAIEmbedder(cfg)
    return _LocalEmbedder(cfg)


class _LocalEmbedder:
    def __init__(self, cfg):
        from sentence_transformers import SentenceTransformer
        e = cfg["embedding"]
        log.info(f"Local embedder: {e['model']} (max_seq_length={e['max_seq_length']}, dtype={e['dtype']})")
        self.model = SentenceTransformer(e["model"], model_kwargs={"torch_dtype": e["dtype"]})
        self.model.max_seq_length = int(e["max_seq_length"])   # <- the slow-embedding fix
        self.batch_size = int(e["batch_size"])
        self.normalize = bool(e["normalize"])

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                                 show_progress_bar=True, convert_to_numpy=True).astype("float32")


class _OpenAIEmbedder:
    """OpenAI embeddings with token-budgeted batching and retry.

    The embeddings endpoint enforces per-request caps (~300K TOTAL tokens summed
    across inputs, and <=2048 inputs). The '10 longest reviews per book' selection
    biases inputs long, so a fixed 256-item chunk can overflow the token cap.
    We batch by BOTH item count and a conservative token estimate (len/3 chars->tokens,
    with a 250K budget for headroom), and retry transient failures with backoff.
    """

    MAX_ITEMS = 128
    MAX_TOKENS_PER_REQ = 250_000
    MAX_CHARS_PER_TEXT = 8_000     # stays safely under the ~8K-token per-input cap

    def __init__(self, cfg):
        from openai import OpenAI
        e = cfg["embedding"]
        key = os.environ.get(e.get("api_key_env", "OPENAI_API_KEY"), "") or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("embedding.backend=openai but no API key found. Add "
                               f"{e.get('api_key_env', 'OPENAI_API_KEY')}=your-key to the .env "
                               "file in the project root (or set the env var).")
        self.client = OpenAI(api_key=key)
        self.model = e["openai_model"]
        log.info(f"OpenAI embedder: {self.model}")

    def _create(self, batch, max_retries: int = 6):
        attempt = 0
        while True:
            try:
                return self.client.embeddings.create(model=self.model, input=batch)
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                status = getattr(e, "status_code", None)
                retriable = (status in (408, 409, 429, 500, 502, 503, 504)
                             or "rate limit" in msg or "timeout" in msg
                             or "connection" in msg or "overloaded" in msg)
                if attempt < max_retries and retriable:
                    delay = min(60.0, (2.0 ** attempt) + random.random())
                    log.warning(f"embeddings request failed ({e}); retry {attempt + 1}/{max_retries} in {delay:.1f}s")
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise

    def encode(self, texts):
        from tqdm import tqdm
        out = []
        batch, btok = [], 0
        pbar = tqdm(total=len(texts), desc="OpenAI embeddings")

        def flush():
            nonlocal batch, btok
            if not batch:
                return
            resp = self._create(batch)
            out.extend(d.embedding for d in resp.data)
            pbar.update(len(batch))
            batch, btok = [], 0

        for t in texts:
            t = (t or " ")[:self.MAX_CHARS_PER_TEXT]
            approx = max(1, len(t) // 3)   # conservative chars->tokens estimate
            if batch and (len(batch) >= self.MAX_ITEMS or btok + approx > self.MAX_TOKENS_PER_REQ):
                flush()
            batch.append(t)
            btok += approx
        flush()
        pbar.close()
        return l2_normalize(np.asarray(out, dtype="float32"))
