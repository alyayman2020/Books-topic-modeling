"""Semantic Book Atlas — interactive explorer (GPU-free).

Interactive Plotly map of all books + pick-a-book nearest neighbors (NumPy cosine) +
browse by cluster & category. Optional free-text search embeds the query via OpenAI.
Loads PRECOMPUTED artifacts only. Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np, polars as pl, streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from book_atlas.utils import load_config, load_params  # noqa: E402

st.set_page_config(page_title="Semantic Book Atlas", layout="wide")
CFG, PRM = load_config(), load_params()
ART, DS = CFG["artifacts_dir"], CFG["dataset_dir"]


@st.cache_resource
def load_all():
    books = pl.read_parquet(f"{DS}/books.parquet").select(
        ["book_id", "title", "author_name", "top_genre", "average_rating"])
    clusters = pl.read_parquet(f"{ART}/clusters.parquet")
    labels = pl.read_parquet(f"{ART}/cluster_labels.parquet")
    ids = pl.read_parquet(f"{ART}/book_ids.parquet").get_column("book_id").to_list()
    xy = np.load(f"{ART}/book_2d.npy")
    vecs = np.load(f"{ART}/book_vectors.npy").astype("float32")
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    # Content-only vectors for FREE-TEXT search: an OpenAI query embedding is
    # 1536-d, while the fused vectors are 3x1536-d (content|pool|distill), so the
    # query must be matched against the content block, not the fused matrix.
    cvec = np.load(f"{ART}/content_vectors.npy").astype("float32")
    crow = {b: i for i, b in enumerate(
        pl.read_parquet(f"{ART}/content_ids.parquet").get_column("book_id").to_list())}
    cvec = cvec[[crow[b] for b in ids]]
    cvec /= (np.linalg.norm(cvec, axis=1, keepdims=True) + 1e-9)
    return books, clusters, labels, ids, xy, vecs, cvec


books, clusters, labels, ids, xy, vecs, cvec = load_all()
row_of = {b: i for i, b in enumerate(ids)}
id2title = dict(zip(books.get_column("book_id"), books.get_column("title")))
id2author = dict(zip(books.get_column("book_id"), books.get_column("author_name")))
id2cluster = dict(zip(clusters.get_column("book_id"), clusters.get_column("cluster_id")))
label_map = {r["cluster_id"]: r for r in labels.iter_rows(named=True)}


def cname(cid):
    return label_map.get(cid, {}).get("name", f"Cluster {cid}") if cid >= 0 else "Unclustered"


def neighbors(row_idx, k=12):
    sims = vecs @ vecs[row_idx]
    order = np.argsort(sims)[::-1]
    return [(ids[j], float(sims[j])) for j in order if j != row_idx][:k]


st.title("📚 Semantic Book Atlas")
st.caption(f"{len(ids):,} books · balanced across genres · clustered by content + reviews.")

tab_map, tab_find, tab_browse = st.tabs(["🗺️ Map", "🔎 Find similar", "🗂️ Categories"])

with tab_map:
    plot_df = pl.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1],
        "title": [id2title.get(b, "") for b in ids],
        "author": [id2author.get(b, "") for b in ids],
        "category": [cname(id2cluster.get(b, -1)) for b in ids],
    }).to_pandas()
    fig = px.scatter(plot_df, x="x", y="y", color="category",
                     hover_data={"title": True, "author": True, "x": False, "y": False},
                     height=680, opacity=0.75)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend=dict(font=dict(size=9)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tab_find:
    mode = st.radio("Search by", ["Pick a book", "Describe what you want"], horizontal=True)
    query_row = None
    if mode == "Pick a book":
        pick = st.selectbox("Book", ids, format_func=lambda b: f"{id2title.get(b,'')} — {id2author.get(b,'')}")
        query_row = row_of.get(pick)
    else:
        q = st.text_input("e.g. 'atmospheric gothic mysteries with unreliable narrators'")
        if q:
            key = os.environ.get(PRM["embedding"].get("api_key_env", "OPENAI_API_KEY"), "")
            if not key:
                st.warning("Set OPENAI_API_KEY to use free-text search.")
            else:
                from openai import OpenAI
                v = OpenAI(api_key=key).embeddings.create(
                    model=PRM["embedding"]["openai_model"], input=[q]).data[0].embedding
                v = np.asarray(v, dtype="float32"); v /= (np.linalg.norm(v) + 1e-9)
                sims = cvec @ v   # content space (1536-d) — matches the query embedding
                for b, s in [(ids[j], float(sims[j])) for j in np.argsort(sims)[::-1][:12]]:
                    st.markdown(f"**{id2title.get(b,'')}** · {id2author.get(b,'')} · *{cname(id2cluster.get(b,-1))}* · {s:.2f}")
    if query_row is not None:
        cid = id2cluster.get(ids[query_row], -1)
        lab = label_map.get(cid, {})
        st.info(f"🏷️ **{cname(cid)}** — {lab.get('description','')}")
        st.write("**Most similar books:**")
        for b, s in neighbors(query_row):
            st.markdown(f"- **{id2title.get(b,'')}** · {id2author.get(b,'')} · *{cname(id2cluster.get(b,-1))}* · {s:.2f}")

with tab_browse:
    named = labels.sort("cluster_id")
    choice = st.selectbox("Category", named.get_column("cluster_id").to_list(), format_func=cname)
    lab = label_map.get(choice, {})
    st.subheader(cname(choice))
    st.write(lab.get("description", ""))
    st.caption("Keywords: " + lab.get("keywords", ""))
    st.write("**Representative books:** " + lab.get("representatives", ""))
