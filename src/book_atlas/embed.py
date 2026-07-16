"""Stage 4 — embed content documents via the configured backend (local or OpenAI)."""
from __future__ import annotations
import numpy as np, polars as pl
from book_atlas.utils import get_logger, load_config, load_params, ensure_dir
from book_atlas.embed_backend import get_embedder

log = get_logger("embed")


def main() -> None:
    cfg = load_config(); prm = load_params()
    proc = cfg["processed_dir"]; art = ensure_dir(cfg["artifacts_dir"])
    df = pl.read_parquet(f"{proc}/documents.parquet")
    ids = df.get_column("book_id").to_list()
    docs = df.get_column("doc_text").to_list()
    log.info(f"Embedding {len(docs):,} content documents ...")
    emb = get_embedder(prm).encode(docs)
    np.save(f"{art}/content_vectors.npy", emb)
    pl.DataFrame({"book_id": ids}).write_parquet(f"{art}/content_ids.parquet")
    log.info(f"content_vectors: {emb.shape} -> {art}/content_vectors.npy")


if __name__ == "__main__":
    main()
