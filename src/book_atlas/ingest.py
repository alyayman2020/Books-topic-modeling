"""Stage 1 — extract metadata tables from the raw Goodreads ndjson files into Parquet.

Polars streaming keeps the ~2GB books file out of memory. Interactions/id-maps are NOT ingested.
"""
from __future__ import annotations
import argparse, json
import polars as pl
from book_atlas.utils import get_logger, load_config, ensure_dir

log = get_logger("ingest")


def ingest_books(path: str, out: str, sample: int | None = None) -> None:
    log.info(f"Scanning books: {path}" + (f"  (sample first {sample})" if sample else ""))
    lf = pl.scan_ndjson(path, ignore_errors=True)
    if sample:
        lf = lf.head(sample)
    lf = lf.select([
        pl.col("book_id").cast(pl.Utf8),
        pl.col("title").cast(pl.Utf8),
        pl.col("description").cast(pl.Utf8),
        pl.col("publisher").cast(pl.Utf8),
        pl.col("language_code").cast(pl.Utf8),
        pl.col("average_rating").cast(pl.Float64, strict=False),
        pl.col("ratings_count").cast(pl.Int64, strict=False),
        pl.col("text_reviews_count").cast(pl.Int64, strict=False),
        pl.col("num_pages").cast(pl.Int64, strict=False),
        pl.col("work_id").cast(pl.Utf8),
        pl.col("series").alias("series_ids"),  # list[str] of series_ids
        pl.col("authors").list.eval(pl.element().struct.field("author_id")).list.first().alias("first_author_id"),
        pl.col("popular_shelves").list.eval(pl.element().struct.field("name")).list.head(10).alias("shelves"),
    ])
    df = lf.collect(engine="streaming")
    df.write_parquet(out)
    log.info(f"books_all: {df.height:,} rows -> {out}")


def ingest_authors(path: str, out: str) -> None:
    (pl.scan_ndjson(path, ignore_errors=True)
       .select([pl.col("author_id").cast(pl.Utf8),
                pl.col("name").cast(pl.Utf8),
                pl.col("average_rating").cast(pl.Float64, strict=False)])
       .collect(engine="streaming").write_parquet(out))
    log.info(f"authors -> {out}")


def ingest_works(path: str, out: str) -> None:
    (pl.scan_ndjson(path, ignore_errors=True)
       .select([pl.col("work_id").cast(pl.Utf8),
                pl.col("original_publication_year").cast(pl.Int64, strict=False).alias("pub_year"),
                pl.col("original_title").cast(pl.Utf8).alias("original_title")])
       .collect(engine="streaming").write_parquet(out))
    log.info(f"works -> {out}")


def ingest_series(path: str, out: str) -> None:
    (pl.scan_ndjson(path, ignore_errors=True)
       .select([pl.col("series_id").cast(pl.Utf8),
                pl.col("title").cast(pl.Utf8).alias("series_title")])
       .collect(engine="streaming").write_parquet(out))
    log.info(f"series -> {out}")


def ingest_genres(path: str, out: str) -> None:
    """genres is a dict of {genre: count} with varying keys -> parse line-by-line for robustness."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            g = o.get("genres", {}) or {}
            top = max(g, key=g.get) if g else None
            rows.append({"book_id": str(o["book_id"]), "top_genre": top, "genres_json": json.dumps(g)})
    pl.DataFrame(rows).write_parquet(out)
    log.info(f"genres: {len(rows):,} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="Only read the first N books (Phase-0 fast pilot).")
    args = ap.parse_args()
    cfg = load_config(); p = cfg["paths"]; proc = ensure_dir(cfg["processed_dir"])
    ingest_books(p["books"], f"{proc}/books_all.parquet", sample=args.sample)
    ingest_authors(p["authors"], f"{proc}/authors.parquet")
    ingest_works(p["works"], f"{proc}/works.parquet")
    try:
        ingest_series(p["series"], f"{proc}/series.parquet")
    except Exception as e:  # noqa: BLE001
        log.warning(f"series ingest skipped ({e}); writing empty series.parquet (series is optional).")
        pl.DataFrame(schema={"series_id": pl.Utf8, "series_title": pl.Utf8}).write_parquet(f"{proc}/series.parquet")
    ingest_genres(p["genres"], f"{proc}/genres.parquet")
    log.info("Ingest complete.")


if __name__ == "__main__":
    main()
