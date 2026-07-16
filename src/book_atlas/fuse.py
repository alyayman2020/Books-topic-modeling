"""Stage F — fuse content + pooled-review + distilled-review vectors into one book vector.

book_vector = L2normalize([ w_c * content | w_p * review_pool | w_d * review_distill ])
Missing components (e.g. a book not distilled) contribute a zero block, then the whole vector is renormalized.
"""
from __future__ import annotations
import numpy as np, polars as pl
from pathlib import Path
from book_atlas.utils import get_logger, load_config, load_params, ensure_dir, l2_normalize

log = get_logger("fuse")


def _load(npy: str, ids_parquet: str):
    if not Path(npy).exists():
        return None, {}
    vecs = np.load(npy)
    ids = pl.read_parquet(ids_parquet).get_column("book_id").to_list()
    return vecs, {b: i for i, b in enumerate(ids)}


def main() -> None:
    cfg = load_config(); prm = load_params(); f = prm["fusion"]; art = cfg["artifacts_dir"]

    content = np.load(f"{art}/content_vectors.npy")
    c_ids = pl.read_parquet(f"{art}/content_ids.parquet").get_column("book_id").to_list()
    pool_v, pool_map = _load(f"{art}/review_pool.npy", f"{art}/review_pool_ids.parquet")
    dist_v, dist_map = _load(f"{art}/review_distill.npy", f"{art}/review_distill_ids.parquet")

    content = l2_normalize(content)
    dim = content.shape[1]
    zero = np.zeros(dim, dtype="float32")

    blocks = []
    for i, bid in enumerate(c_ids):
        cv = content[i] * f["w_content"]
        pv = (l2_normalize(pool_v[pool_map[bid]]) * f["w_review_pool"]) if bid in pool_map else zero
        dv = (l2_normalize(dist_v[dist_map[bid]]) * f["w_review_distill"]) if bid in dist_map else zero
        blocks.append(np.concatenate([cv, pv, dv]))
    fused = l2_normalize(np.vstack(blocks).astype("float32"))

    np.save(f"{art}/book_vectors.npy", fused)
    pl.DataFrame({"book_id": c_ids}).write_parquet(f"{art}/book_ids.parquet")
    log.info(f"book_vectors: {fused.shape} (content+pool+distill) -> {art}/book_vectors.npy")
    log.info(f"  books with pooled reviews: {len(pool_map):,} | with distillation: {len(dist_map):,}")


if __name__ == "__main__":
    main()
