"""Stage 8 — cluster labeling grounded in real content.

Feeds gpt-5.4-nano each cluster's representative book titles/descriptions + a few real
review excerpts (richest signal), plus c-TF-IDF keywords as a hint. Falls back to
c-TF-IDF keywords if the LLM is unavailable.
"""
from __future__ import annotations
import re
from collections import defaultdict
import numpy as np, polars as pl
from sklearn.feature_extraction.text import CountVectorizer
from book_atlas.utils import get_logger, load_config, load_params
from book_atlas import llm

log = get_logger("label")


def _ctfidf(docs_per_cluster: list[str], top_k: int) -> list[list[str]]:
    vect = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000, min_df=2)
    tf = vect.fit_transform(docs_per_cluster).toarray().astype(float)
    tf_sum = tf.sum(1, keepdims=True); tf_sum[tf_sum == 0] = 1.0
    ctf = (tf / tf_sum) * np.log(1 + tf.shape[0] / (1 + (tf > 0).sum(0)))
    terms = np.array(vect.get_feature_names_out())
    return [terms[row.argsort()[::-1][:top_k]].tolist() for row in ctf]


def _parse(text: str) -> tuple[str, str]:
    # Strip markdown emphasis first: models often reply "**NAME:** ..." which
    # would otherwise miss the regex and leave "** " glued to the fallback.
    t = (text or "").replace("**", "").replace("__", "").strip()
    name = re.search(r"NAME:\s*(.+)", t)
    desc = re.search(r"DESC:\s*(.+)", t, re.S)
    return (name.group(1).strip() if name else t.split("\n")[0][:60].strip(),
            desc.group(1).strip() if desc else "")


def main() -> None:
    cfg = load_config(); prm = load_params(); art = cfg["artifacts_dir"]; lb = prm["label"]
    docs = pl.read_parquet(f"{cfg['processed_dir']}/documents.parquet")
    clusters = pl.read_parquet(f"{art}/clusters.parquet")
    books = pl.read_parquet(f"{cfg['dataset_dir']}/books.parquet").select(["book_id", "title", "description"])
    reviews = pl.read_parquet(f"{cfg['dataset_dir']}/reviews.parquet").select(["book_id", "review_text"])

    df = docs.join(clusters, on="book_id").filter(pl.col("cluster_id") >= 0)
    grouped = df.group_by("cluster_id").agg(pl.col("doc_text")).sort("cluster_id")
    cluster_ids = grouped.get_column("cluster_id").to_list()
    keywords = _ctfidf([" ".join(x) for x in grouped.get_column("doc_text").to_list()], lb["top_keywords"])

    # representative books = nearest to centroid in content-embedding space
    cv = np.load(f"{art}/content_vectors.npy")
    cids = pl.read_parquet(f"{art}/content_ids.parquet").get_column("book_id").to_list()
    id2row = {b: i for i, b in enumerate(cids)}
    title_map = dict(zip(books.get_column("book_id"), books.get_column("title")))
    desc_map = dict(zip(books.get_column("book_id"), books.get_column("description")))
    rev_map = defaultdict(list)
    for b, t in zip(reviews.get_column("book_id"), reviews.get_column("review_text")):
        if len(rev_map[b]) < 3:
            rev_map[b].append(t)

    members = defaultdict(list)
    for b, cl in zip(clusters.get_column("book_id"), clusters.get_column("cluster_id")):
        if cl >= 0 and b in id2row:
            members[cl].append(b)
    reps = {}
    for cl, bs in members.items():
        rows = [id2row[b] for b in bs]
        cent = cv[rows].mean(0); n = np.linalg.norm(cent); cent = cent / n if n > 0 else cent
        order = np.argsort(cv[rows] @ cent)[::-1][:lb["n_representatives"]]
        reps[cl] = [bs[i] for i in order]

    use_llm = llm.available(prm["llm"])
    if not use_llm:
        log.warning("Labeling with c-TF-IDF fallback (LLM unavailable).")
    rows_out = []
    for i, cid in enumerate(cluster_ids):
        kw = keywords[i]; rep_ids = reps.get(cid, [])
        rep_titles = [title_map.get(b, "") for b in rep_ids]
        name, desc = ", ".join(kw[:3]), ""
        if use_llm:
            # build a content-rich prompt: titles + short descriptions + real review snippets
            snippets = []
            for b in rep_ids:
                d = (desc_map.get(b) or "")[:200]
                rv = " / ".join((r or "")[:160] for r in rev_map.get(b, [])[:1])
                snippets.append(f"- {title_map.get(b,'')}: {d} | readers: {rv}")
                if len(snippets) >= lb["review_snippets_per_cluster"]:
                    break
            prompt = ("You are naming a category of books for a discovery app, based on real content.\n"
                      f"Keyword hints: {', '.join(kw)}\n\nRepresentative books (with descriptions and reader reviews):\n"
                      + "\n".join(snippets) +
                      "\n\nGive a SHORT, specific category name (max 6 words) and a one-sentence description.\n"
                      "Respond EXACTLY as:\nNAME: <name>\nDESC: <description>")
            try:
                name, desc = _parse(llm.chat(prm["llm"], prompt))
            except Exception as e:  # noqa: BLE001
                log.warning(f"label LLM failed for cluster {cid}: {e}")
        rows_out.append({"cluster_id": cid, "name": name, "description": desc,
                         "keywords": ", ".join(kw), "representatives": " | ".join(rep_titles)})
        if (i + 1) % 20 == 0:
            log.info(f"labeled {i + 1}/{len(cluster_ids)}")
    pl.DataFrame(rows_out).write_parquet(f"{art}/cluster_labels.parquet")
    log.info(f"cluster_labels: {len(rows_out)} clusters -> {art}/cluster_labels.parquet")


if __name__ == "__main__":
    main()
