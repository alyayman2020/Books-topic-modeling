"""Stage 9 — evaluation panel: DBCV, silhouette, noise, coherence (c_v), diversity,
LLM-as-judge (configured LLM), and NMI/ARI vs Goodreads genres. Writes reports/metrics.json."""
from __future__ import annotations
import json, os, random
import numpy as np, polars as pl
from book_atlas.utils import get_logger, load_config, load_params, ensure_dir
from book_atlas import llm

log = get_logger("evaluate")


def _coherence(cfg, art) -> float | None:
    lab_fp = f"{art}/cluster_labels.parquet"
    if not os.path.exists(lab_fp):
        return None
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    raw = [k.split(", ")[:10] for k in pl.read_parquet(lab_fp).get_column("keywords").to_list()]
    docs = pl.read_parquet(f"{cfg['processed_dir']}/documents.parquet").get_column("doc_text").to_list()
    rng = random.Random(42)
    texts = [d.lower().split() for d in rng.sample(docs, min(len(docs), 20000))]
    d = Dictionary(texts)
    vocab = set(d.token2id)
    # c-TF-IDF emits bigrams ("space opera") that whitespace-tokenized texts can
    # never contain: gensim silently DROPS them (distorting the score) and raises
    # if a topic is all-bigrams. Keep only in-vocabulary unigrams for c_v.
    topics = [[w for w in t if w in vocab] for t in raw]
    topics = [t for t in topics if len(t) >= 2]
    if len(topics) < 2:
        return None
    return float(CoherenceModel(topics=topics, texts=texts, dictionary=d, coherence="c_v").get_coherence())


def _diversity(art, top_n: int = 10) -> float | None:
    lab_fp = f"{art}/cluster_labels.parquet"
    if not os.path.exists(lab_fp):
        return None
    lists = [k.split(", ")[:top_n] for k in pl.read_parquet(lab_fp).get_column("keywords").to_list()]
    all_words = [w for lst in lists for w in lst]
    return round(len(set(all_words)) / max(len(all_words), 1), 4)


def _judge(art, lj, cfg_llm) -> dict:
    lab = pl.read_parquet(f"{art}/cluster_labels.parquet")
    sample = lab.sample(min(lj["sample_clusters"], lab.height), seed=42)
    ratings = []
    for row in sample.iter_rows(named=True):
        prompt = ("Rate how coherent this book category is on a 1-3 scale "
                  "(1=unrelated, 2=somewhat related, 3=very coherent).\n"
                  f"Name: {row['name']}\nKeywords: {row['keywords']}\nExamples: {row['representatives']}\n"
                  "Answer with ONLY the digit 1, 2, or 3.")
        try:
            r = llm.chat(cfg_llm, prompt)
            digit = next((int(ch) for ch in r if ch in "123"), None)
            if digit:
                ratings.append(digit)
        except Exception:  # noqa: BLE001
            pass
    return {"mean_rating": round(sum(ratings) / len(ratings), 2) if ratings else None, "n_rated": len(ratings)}


def main() -> None:
    cfg = load_config(); prm = load_params(); art = cfg["artifacts_dir"]; rep = ensure_dir(cfg["reports_dir"])
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

    X = np.load(f"{art}/book_nd.npy")
    clusters = pl.read_parquet(f"{art}/clusters.parquet")
    labels = clusters.get_column("cluster_id").to_numpy()
    valid = labels >= 0
    m: dict = {"n_clusters": int((np.unique(labels) >= 0).sum()),
               "noise_fraction": round(float((labels == -1).mean()), 4)}

    try:
        m["silhouette"] = round(float(silhouette_score(
            X[valid], labels[valid], sample_size=min(20000, int(valid.sum())), random_state=42)), 4)
    except Exception as e:  # noqa: BLE001
        m["silhouette"] = None; log.warning(f"silhouette: {e}")
    try:
        from hdbscan.validity import validity_index
        m["dbcv"] = round(float(validity_index(X[valid].astype("float64"), labels[valid])), 4)
    except Exception as e:  # noqa: BLE001
        m["dbcv"] = None; log.warning(f"dbcv: {e}")
    try:
        m["coherence_cv"] = _coherence(cfg, art)
    except Exception as e:  # noqa: BLE001
        m["coherence_cv"] = None; log.warning(f"coherence: {e}")
    m["topic_diversity"] = _diversity(art)

    books = pl.read_parquet(f"{cfg['dataset_dir']}/books.parquet").select(["book_id", "top_genre"])
    j = clusters.join(books, on="book_id").filter((pl.col("cluster_id") >= 0) & pl.col("top_genre").is_not_null())
    if j.height:
        cl = j.get_column("cluster_id").to_list(); gm = j.get_column("top_genre").to_list()
        m["nmi_vs_genre"] = round(float(normalized_mutual_info_score(gm, cl)), 4)
        m["ari_vs_genre"] = round(float(adjusted_rand_score(gm, cl)), 4)

    lj = prm.get("llm_judge", {})
    if lj.get("enabled") and os.path.exists(f"{art}/cluster_labels.parquet") and llm.available(prm["llm"]):
        m["llm_judge"] = _judge(art, lj, prm["llm"])

    json.dump(m, open(f"{rep}/metrics.json", "w"), indent=2)
    log.info("metrics.json:\n" + json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
