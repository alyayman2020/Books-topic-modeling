"""Stage R — use each book's selected reviews.

1. Cap to N reviews/book (default: 10 longest).
2. Embed them (local or API) and weighted-mean-pool per book -> review_pool.
3. (Optional) Distill via the configured LLM provider -> review_distill.

Distillation is checkpointed: every successful summary is appended to
data/artifacts/review_distill_checkpoint.jsonl, so an interrupted run
(crash, rate limits, Ctrl-C) resumes where it left off instead of re-paying
for thousands of completed calls.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl
from collections import defaultdict
from book_atlas.utils import get_logger, load_config, load_params, ensure_dir
from book_atlas.embed_backend import get_embedder
from book_atlas import llm

log = get_logger("reviews")


def _cap(rv: pl.DataFrame, max_per_book: int, select_by: str) -> pl.DataFrame:
    if not max_per_book or max_per_book <= 0 or select_by == "all":
        return rv
    if select_by == "longest":
        rv = rv.with_columns(pl.col("review_text").str.len_chars().alias("_k")).sort("_k", descending=True)
    elif select_by == "votes":
        rv = rv.sort("n_votes", descending=True, nulls_last=True)
    rv = rv.group_by("book_id", maintain_order=True).head(max_per_book)
    return rv.drop("_k") if "_k" in rv.columns else rv


def _pool(review_emb, book_ids, votes, weight_by_votes):
    idx = defaultdict(list)
    for i, b in enumerate(book_ids):
        idx[b].append(i)
    ids, vecs = [], []
    for b, rows in idx.items():
        w = (1.0 + np.log1p(np.maximum(votes[rows], 0)))[:, None] if weight_by_votes else np.ones((len(rows), 1))
        v = (review_emb[rows] * w).sum(0) / w.sum()
        n = np.linalg.norm(v)
        ids.append(b); vecs.append(v / n if n > 0 else v)
    return ids, np.vstack(vecs).astype("float32")


def _distill(rv: pl.DataFrame, embedder, prm: dict, art: str) -> None:
    cfg_llm = prm["llm"]; r = prm["reviews"]
    if not llm.available(cfg_llm):
        log.warning("Skipping distillation (LLM provider unavailable).")
        return
    counts = rv.group_by("book_id").len().sort("len", descending=True)
    top_ids = counts.head(r["distill_top_n"]).get_column("book_id").to_list()

    # One partition pass instead of a full-frame filter per book (O(N) vs O(N*B)).
    by_book: dict = {}
    for key, sub in rv.partition_by("book_id", as_dict=True).items():
        b = key[0] if isinstance(key, tuple) else key
        by_book[b] = sub

    # Resume support: successful summaries are checkpointed as JSONL.
    ckpt = Path(f"{art}/review_distill_checkpoint.jsonl")
    done: dict[str, str] = {}
    if ckpt.exists():
        with open(ckpt, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    done[o["book_id"]] = o["summary"]
                except Exception:  # noqa: BLE001
                    continue
        log.info(f"Resuming distillation: {len(done):,} books already summarized.")

    log.info(f"Distilling {len(top_ids):,} books via {cfg_llm['provider']}:{cfg_llm['model']} ...")
    with open(ckpt, "a", encoding="utf-8") as ck:
        for i, b in enumerate(top_ids):
            if b not in done:
                sub = by_book.get(b)
                if sub is None:
                    continue
                sub = sub.head(r["distill_max_reviews_in_prompt"])
                joined = "\n\n".join(f"- {t[:600]}" for t in sub.get_column("review_text").to_list())
                prompt = ("Summarize what readers say about this book in 3-4 sentences: themes, tone, and what "
                          "readers liked or disliked. Be specific and neutral.\n\nReviews:\n" + joined)
                try:
                    s = llm.chat(cfg_llm, prompt)
                    done[b] = s
                    ck.write(json.dumps({"book_id": b, "summary": s}) + "\n")
                    ck.flush()
                except Exception as e:  # noqa: BLE001
                    log.warning(f"distill failed for {b}: {e}")
            if (i + 1) % 200 == 0:
                log.info(f"  distilled {i + 1:,}/{len(top_ids):,}")

    ids = [b for b in top_ids if b in done]
    summaries = [done[b] for b in ids]
    if not ids:
        log.warning("No distillation produced.")
        return
    demb = embedder.encode(summaries)
    np.save(f"{art}/review_distill.npy", demb)
    pl.DataFrame({"book_id": ids}).write_parquet(f"{art}/review_distill_ids.parquet")
    pl.DataFrame({"book_id": ids, "summary": summaries}).write_parquet(f"{art}/review_distill_text.parquet")
    log.info(f"review_distill: {demb.shape} -> {art}/review_distill.npy")


def main() -> None:
    cfg = load_config(); prm = load_params(); r = prm["reviews"]; art = ensure_dir(cfg["artifacts_dir"])
    rv = pl.read_parquet(f"{cfg['dataset_dir']}/reviews.parquet")
    log.info(f"Loaded {rv.height:,} reviews for {rv.get_column('book_id').n_unique():,} books.")
    rv = _cap(rv, r["max_reviews_per_book"], r["select_by"])
    log.info(f"After cap ({r['select_by']}, {r['max_reviews_per_book']}/book): {rv.height:,} reviews "
             f"across {rv.get_column('book_id').n_unique():,} books.")

    embedder = get_embedder(prm)
    texts = rv.get_column("review_text").to_list()
    log.info(f"Embedding {len(texts):,} reviews ...")
    remb = embedder.encode(texts)
    ids, pooled = _pool(remb, rv.get_column("book_id").to_numpy(),
                        rv.get_column("n_votes").fill_null(0).to_numpy(), r["weight_by_votes"])
    np.save(f"{art}/review_pool.npy", pooled)
    pl.DataFrame({"book_id": ids}).write_parquet(f"{art}/review_pool_ids.parquet")
    log.info(f"review_pool: {pooled.shape} -> {art}/review_pool.npy")

    if r["distill"]:
        _distill(rv, embedder, prm, art)


if __name__ == "__main__":
    main()
