"""Stage 3 — build the per-book content document for the configured embedding backend (reads the frozen bundle)."""
from __future__ import annotations
import polars as pl
from book_atlas.utils import get_logger, load_config, ensure_dir

log = get_logger("build_docs")


def main() -> None:
    cfg = load_config(); proc = ensure_dir(cfg["processed_dir"])
    df = pl.read_parquet(f"{cfg['dataset_dir']}/books.parquet")

    df = df.with_columns([
        pl.col("shelves").list.head(8).list.join(", ").fill_null("").alias("shelves_s"),
        (pl.col("description").fill_null("")
           .str.replace_all(r"<[^>]+>", " ")
           .str.replace_all(r"\s+", " ")
           .str.strip_chars()).alias("desc_clean"),
    ]).with_columns(
        pl.concat_str([
            pl.lit("Title: "), pl.col("title").fill_null(""),
            pl.lit(". Author: "), pl.col("author_name").fill_null(""),
            pl.lit(". Genre: "), pl.col("top_genre").fill_null(""),
            pl.lit(". Reader shelves: "), pl.col("shelves_s"),
            pl.lit(". Description: "), pl.col("desc_clean"),
        ]).alias("doc_text")
    ).select(["book_id", "doc_text"])

    df.write_parquet(f"{proc}/documents.parquet")
    log.info(f"documents: {df.height:,} rows -> {proc}/documents.parquet")


if __name__ == "__main__":
    main()
