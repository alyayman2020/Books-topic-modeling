"""Generate the DataMapPlot hero map (PNG + interactive HTML) for the README/demo."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from book_atlas.utils import load_config, ensure_dir  # noqa: E402


def main() -> None:
    cfg = load_config(); art = cfg["artifacts_dir"]
    fig_dir = ensure_dir(f"{cfg['reports_dir']}/figures")
    import datamapplot
    xy = np.load(f"{art}/book_2d.npy")
    clusters = pl.read_parquet(f"{art}/clusters.parquet")
    labels = pl.read_parquet(f"{art}/cluster_labels.parquet")
    id2name = dict(zip(labels.get_column("cluster_id"), labels.get_column("name")))
    ids = pl.read_parquet(f"{cfg['artifacts_dir']}/book_ids.parquet").get_column("book_id").to_list()
    id2cluster = dict(zip(clusters.get_column("book_id"), clusters.get_column("cluster_id")))
    label_arr = np.array([id2name.get(id2cluster.get(b, -1), "Unlabelled") for b in ids])

    fig, _ = datamapplot.create_plot(
        xy, label_arr, title="Semantic Book Atlas",
        sub_title=f"{len(ids):,} books · balanced across genres · clustered by content + reviews")
    fig.savefig(f"{fig_dir}/book_atlas_map.png", dpi=300, bbox_inches="tight")
    print(f"saved {fig_dir}/book_atlas_map.png")
    try:
        datamapplot.create_interactive_plot(xy, label_arr).save(f"{fig_dir}/book_atlas_map.html")
        print(f"saved {fig_dir}/book_atlas_map.html")
    except Exception as e:  # noqa: BLE001
        print(f"interactive map skipped: {e}")


if __name__ == "__main__":
    main()
