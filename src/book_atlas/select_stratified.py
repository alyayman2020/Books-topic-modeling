"""Stage 2 — STRATIFIED sample of N books balanced across categories.

Every category (top_genre) gets a roughly equal quota; within each, the most-reviewed
books are taken (so each book has review context). Small categories are taken in full and
the shortfall is topped up from the most-reviewed remainder. Then reviews are filtered to
the selected books.
"""
from __future__ import annotations
import math
import polars as pl
from book_atlas.utils import get_logger, load_config, load_params, ensure_dir

log = get_logger("select")


def main() -> None:
    cfg = load_config(); prm = load_params(); proc = ensure_dir(cfg["processed_dir"])
    s = prm["select"]; N = s["n_books"]; strat = s["stratify_by"]; rank = s["rank_by"]

    books = pl.read_parquet(f"{proc}/books_all.parquet")
    authors = pl.read_parquet(f"{proc}/authors.parquet")
    works = pl.read_parquet(f"{proc}/works.parquet")
    genres = pl.read_parquet(f"{proc}/genres.parquet")

    books = books.join(genres.select(["book_id", "top_genre", "genres_json"]), on="book_id", how="left")
    pool = books.filter(pl.col(strat).is_not_null() & (pl.col(rank).fill_null(0) > 0))
    categories = pool.get_column(strat).unique().to_list()
    per_cat = math.ceil(N / max(len(categories), 1))
    log.info(f"{len(categories)} categories → target ~{per_cat} books/category for {N} total")

    parts = [pool.filter(pl.col(strat) == c).sort(rank, descending=True, nulls_last=True).head(per_cat)
             for c in categories]
    top = pl.concat(parts)

    if top.height > N:                                  # trim overage while keeping balance
        top = top.sample(n=N, seed=42)
    elif top.height < N:                                # top up from the most-reviewed remainder
        have = set(top.get_column("book_id").to_list())
        extra = (pool.filter(~pl.col("book_id").is_in(list(have)))
                     .sort(rank, descending=True, nulls_last=True).head(N - top.height))
        top = pl.concat([top, extra])

    top = (top
           .join(authors.rename({"author_id": "first_author_id", "name": "author_name"})
                        .select(["first_author_id", "author_name"]), on="first_author_id", how="left")
           .join(works, on="work_id", how="left"))
    top.write_parquet(f"{proc}/books_top.parquet")
    dist = top.group_by(strat).len().sort("len", descending=True)
    log.info(f"Selected {top.height:,} books. Category distribution:\n{dist}")

    ids = top.get_column("book_id").to_list()
    log.info(f"Filtering reviews to {len(ids):,} books (streaming) ...")
    reviews = (pl.scan_ndjson(cfg["paths"]["reviews"], ignore_errors=True)
                 .select([pl.col("book_id").cast(pl.Utf8),
                          pl.col("review_text").cast(pl.Utf8),
                          pl.col("rating").cast(pl.Int64, strict=False),
                          pl.col("n_votes").cast(pl.Int64, strict=False)])
                 .filter(pl.col("book_id").is_in(ids))
                 .collect(engine="streaming"))
    reviews = reviews.filter(pl.col("review_text").is_not_null() & (pl.col("review_text").str.len_chars() > 0))
    reviews.write_parquet(f"{proc}/reviews_top.parquet")
    log.info(f"reviews_top: {reviews.height:,} reviews covering "
             f"{reviews.get_column('book_id').n_unique():,} books -> {proc}/reviews_top.parquet")


if __name__ == "__main__":
    main()
