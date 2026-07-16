"""Stage 7 — HDBSCAN clustering: Optuna-tuned on DBCV, leaf selection, soft outlier reassignment,
hierarchical centroid merge. Highest-quality committed recipe."""
from __future__ import annotations
import numpy as np, polars as pl
from book_atlas.utils import get_logger, load_config, load_params, ensure_dir

log = get_logger("cluster")


def _dbcv(emb, labels) -> float:
    try:
        from hdbscan.validity import validity_index
        m = labels >= 0
        if m.sum() < 10 or len(set(labels[m])) < 2:
            return -1.0
        return float(validity_index(emb[m].astype("float64"), labels[m]))
    except Exception:  # noqa: BLE001
        return -1.0


def _merge(vectors, labels, thr: float):
    uniq = sorted({int(x) for x in labels if x >= 0})
    if len(uniq) < 2:
        return labels
    cents = {}
    for c in uniq:
        v = vectors[labels == c].mean(0)
        n = np.linalg.norm(v)
        cents[c] = v / n if n > 0 else v
    parent = {c: c for c in uniq}
    for i, a in enumerate(uniq):
        if parent[a] != a:
            continue
        for b in uniq[i + 1:]:
            if parent[b] != b:
                continue
            if float(np.dot(cents[a], cents[b])) >= thr:
                parent[b] = a
    merged = np.array([parent.get(int(x), x) if x >= 0 else -1 for x in labels])
    relab = {c: i for i, c in enumerate(sorted({int(x) for x in merged if x >= 0}))}
    return np.array([relab.get(int(x), -1) if x >= 0 else -1 for x in merged])


def main() -> None:
    cfg = load_config(); prm = load_params(); art = ensure_dir(cfg["artifacts_dir"]); c = prm["cluster"]
    import hdbscan, optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = np.load(f"{art}/book_nd.npy")

    # Scale the search space to the dataset around the params.yaml knob.
    # (The old hardcoded 50–600 range was sized for a 250K-book corpus; on 5K
    #  points it forces a handful of giant clusters and ignores hdbscan.min_cluster_size.)
    n = X.shape[0]
    base = int(c.get("hdbscan", {}).get("min_cluster_size", 25))
    mcs_lo = max(5, base // 2)
    mcs_hi = max(mcs_lo + 10, min(base * 12, max(20, n // 10)))
    ms_hi = max(10, min(60, base * 2))
    log.info(f"Optuna search space: min_cluster_size in [{mcs_lo}, {mcs_hi}], "
             f"min_samples in [3, {ms_hi}]  (n={n:,}, base={base})")

    def objective(trial):
        mcs = trial.suggest_int("min_cluster_size", mcs_lo, mcs_hi, log=True)
        ms = trial.suggest_int("min_samples", 3, ms_hi)
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method="leaf",
                             gen_min_span_tree=True, core_dist_n_jobs=-1)
        labels = cl.fit_predict(X)
        if (np.unique(labels) >= 0).sum() < 2:
            return -1.0
        return _dbcv(X, labels) - c["noise_penalty"] * (labels == -1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=c["optuna_trials"])
    best = study.best_params
    log.info(f"Best HDBSCAN params: {best}  (objective={study.best_value:.4f})")

    cl = hdbscan.HDBSCAN(min_cluster_size=best["min_cluster_size"], min_samples=best["min_samples"],
                         cluster_selection_method="leaf", prediction_data=True,
                         gen_min_span_tree=True, core_dist_n_jobs=-1).fit(X)
    labels = cl.labels_.astype(int).copy()
    probs = cl.probabilities_.copy()

    try:  # soft reassignment of noise
        soft = hdbscan.all_points_membership_vectors(cl)
        if getattr(soft, "ndim", 0) == 2 and soft.shape[1] > 0:
            noise = labels == -1
            reassign = noise & (soft.max(1) > c["soft_reassign_threshold"])
            labels[reassign] = soft.argmax(1)[reassign].astype(int)
            log.info(f"Soft-reassigned {int(reassign.sum()):,} / {int(noise.sum()):,} noise points")
    except Exception as e:  # noqa: BLE001
        log.warning(f"soft reassignment skipped: {e}")

    labels = _merge(np.load(f"{art}/book_vectors.npy"), labels, c["merge_cosine_threshold"])
    ids = pl.read_parquet(f"{art}/book_ids.parquet").get_column("book_id").to_list()
    pl.DataFrame({"book_id": ids, "cluster_id": labels.tolist(),
                  "probability": probs.tolist()}).write_parquet(f"{art}/clusters.parquet")
    log.info(f"Final clusters: {len({int(x) for x in labels if x >= 0})}  "
             f"noise: {(np.asarray(labels) == -1).mean():.1%} -> {art}/clusters.parquet")


if __name__ == "__main__":
    main()
