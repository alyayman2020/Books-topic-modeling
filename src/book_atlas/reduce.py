"""Stage 6 — dimensionality reduction WITHOUT PCA.

Benchmarks the configured reducers (UMAP, densMAP, optionally PaCMAP) by clustering each
candidate space and scoring it with DBCV, then keeps the best. A separate 2D map is produced
for visualization only.
"""
from __future__ import annotations
import json
import numpy as np
from book_atlas.utils import get_logger, load_config, load_params, ensure_dir

log = get_logger("reduce")


def _reduce(kind: str, X, u: dict, n_components: int):
    if kind in ("umap", "densmap"):
        from umap import UMAP
        return UMAP(n_neighbors=u["n_neighbors"], n_components=n_components, min_dist=u["min_dist"],
                    metric=u["metric"], densmap=(kind == "densmap"),
                    random_state=u["random_state"]).fit_transform(X)
    if kind == "pacmap":
        import pacmap
        return pacmap.PaCMAP(n_components=n_components, n_neighbors=u["n_neighbors"]).fit_transform(X, init="pca")
    raise ValueError(f"unknown reducer: {kind}")


def _dbcv(emb, labels) -> float:
    try:
        from hdbscan.validity import validity_index
        m = labels >= 0
        if m.sum() < 10 or len(set(labels[m])) < 2:
            return -1.0
        return float(validity_index(emb[m].astype("float64"), labels[m]))
    except Exception as e:  # noqa: BLE001
        log.warning(f"DBCV failed: {e}")
        return -1.0


def main() -> None:
    cfg = load_config(); prm = load_params(); art = ensure_dir(cfg["artifacts_dir"])
    r = prm["reduce"]; u = r["umap"]
    X = np.load(f"{art}/book_vectors.npy")
    import hdbscan

    best = None
    for kind in r["reducers"]:
        log.info(f"Reducing with {kind} -> {u['n_components']}D (no PCA) ...")
        emb = _reduce(kind, X, u, u["n_components"])
        cl = hdbscan.HDBSCAN(min_cluster_size=prm["cluster"]["hdbscan"]["min_cluster_size"],
                             gen_min_span_tree=True, core_dist_n_jobs=-1)
        labels = cl.fit_predict(emb)
        score = _dbcv(emb, labels)
        log.info(f"  {kind}: DBCV={score:.4f}  clusters={int((np.unique(labels) >= 0).sum())}  "
                 f"noise={(labels == -1).mean():.1%}")
        if best is None or score > best["score"]:
            best = {"kind": kind, "score": score, "emb": emb}

    log.info(f"Selected reducer: {best['kind']} (DBCV={best['score']:.4f})")
    np.save(f"{art}/book_nd.npy", best["emb"].astype("float32"))
    log.info("Building 2D map for visualization ...")
    np.save(f"{art}/book_2d.npy", _reduce(best["kind"], X, u, 2).astype("float32"))
    json.dump({"reducer": best["kind"], "dbcv": best["score"]},
              open(f"{art}/reduce_choice.json", "w"), indent=2)
    log.info(f"book_nd + book_2d -> {art}/")


if __name__ == "__main__":
    main()
