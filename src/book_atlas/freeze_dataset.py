"""Stage 2b — FREEZE the selected dataset into a self-contained bundle, then optionally delete the raw files.

After this step NO downstream stage needs the raw Goodreads JSON, so you can safely delete them
to reclaim disk. Deletion is opt-in and only happens AFTER verification passes.

    python -m book_atlas.freeze_dataset                       # build + verify the bundle (safe)
    python -m book_atlas.freeze_dataset --delete-raw          # also delete the huge raw JSON files
    python -m book_atlas.freeze_dataset --delete-raw --clean-intermediate  # also remove working parquets
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from book_atlas.utils import get_logger, load_config, load_params, ensure_dir

log = get_logger("freeze")


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _human(n: float) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-raw", action="store_true",
                    help="Delete raw source JSON files AFTER verification passes.")
    ap.add_argument("--clean-intermediate", action="store_true",
                    help="Also delete working parquets in processed_dir (they are duplicated in the bundle).")
    args = ap.parse_args()

    cfg = load_config(); prm = load_params()
    proc = cfg["processed_dir"]
    bundle = ensure_dir(cfg["dataset_dir"])

    books = pl.read_parquet(f"{proc}/books_top.parquet")
    reviews = pl.read_parquet(f"{proc}/reviews_top.parquet")
    series_fp = f"{proc}/series.parquet"
    series = pl.read_parquet(series_fp) if Path(series_fp).exists() else None

    # ---------------- verify (bundle must be self-contained) ----------------
    problems = []
    if books.height == 0:
        problems.append("books_top is empty")
    if reviews.height == 0:
        problems.append("reviews_top is empty")
    if books.get_column("book_id").null_count() > 0:
        problems.append("null book_id present in books")
    book_ids = books.get_column("book_id").to_list()
    stray = reviews.filter(~pl.col("book_id").is_in(book_ids)).height
    if stray > 0:
        problems.append(f"{stray:,} reviews reference books outside the selected set")

    n_books = books.height
    n_reviews = reviews.height
    n_with_rev = reviews.get_column("book_id").n_unique()
    per_book = n_reviews / max(n_with_rev, 1)

    if problems:
        for p in problems:
            log.error(f"VERIFY FAIL: {p}")
        raise SystemExit("Freeze verification FAILED — raw files were NOT touched. Fix and re-run.")
    log.info("VERIFY PASS — the bundle is self-contained.")

    # ---------------- write bundle ----------------
    books.write_parquet(f"{bundle}/books.parquet")
    reviews.write_parquet(f"{bundle}/reviews.parquet")
    if series is not None:
        series.write_parquet(f"{bundle}/series.parquet")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_books_requested": prm["select"]["n_books"],
        "rank_by": prm["select"]["rank_by"],
        "n_books": n_books,
        "n_reviews": n_reviews,
        "n_books_with_reviews": n_with_rev,
        "avg_reviews_per_book": round(per_book, 2),
        "files": ["books.parquet", "reviews.parquet"] + (["series.parquet"] if series is not None else []),
        "note": "Self-contained dataset for the Semantic Book Atlas. Raw Goodreads JSON not required after this point.",
    }
    with open(f"{bundle}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ---------------- report ----------------
    bundle_bytes = sum(_size(os.path.join(bundle, x)) for x in os.listdir(bundle))
    raw_files = [cfg["paths"][k] for k in ("books", "reviews", "authors", "works", "genres", "series")]
    raw_bytes = sum(_size(p) for p in raw_files)
    log.info("=" * 74)
    log.info(f"Frozen bundle written to: {bundle}/")
    log.info(f"  books.parquet   : {n_books:,} books")
    log.info(f"  reviews.parquet : {n_reviews:,} reviews  (~{per_book:.1f}/book across {n_with_rev:,} books)")
    log.info(f"  bundle size     : {_human(bundle_bytes)}")
    log.info(f"  raw source size : {_human(raw_bytes)}   <-- reclaimable")
    log.info("=" * 74)

    if args.delete_raw:
        log.warning("Deleting raw source JSON files ...")
        for p in raw_files:
            if os.path.exists(p):
                os.remove(p)
                log.warning(f"  deleted {p}")
        log.warning(f"Raw files deleted. Your dataset now lives entirely in {bundle}/")
    else:
        log.info("Raw files kept. Now that the bundle is verified, reclaim disk with:")
        log.info("    python -m book_atlas.freeze_dataset --delete-raw")
        log.info("  (or delete these manually:)")
        for p in raw_files:
            if os.path.exists(p):
                log.info(f"    {p}   [{_human(_size(p))}]")

    if args.clean_intermediate:
        for f in ["books_all.parquet", "authors.parquet", "works.parquet", "genres.parquet",
                  "books_top.parquet", "reviews_top.parquet", "series.parquet"]:
            fp = os.path.join(proc, f)
            if os.path.exists(fp):
                os.remove(fp)
                log.warning(f"  removed intermediate {fp}")


if __name__ == "__main__":
    main()
