"""Fast unit tests (no heavy ML deps required)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from book_atlas.utils import clean_html, l2_normalize  # noqa: E402


def test_clean_html():
    assert clean_html("<b>Hi</b>  there&amp;now") == "Hi there&now"
    assert clean_html(None) == ""


def test_l2_normalize():
    v = np.array([[3.0, 4.0]])
    out = l2_normalize(v)
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-6


def test_build_document():
    from book_atlas.build_documents import main  # import-only smoke check
    assert callable(main)


def test_pool_weighting():
    from book_atlas.reviews import _pool
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    ids, pooled = _pool(emb, np.array(["b", "b"]), np.array([0, 100]), weight_by_votes=True)
    assert ids == ["b"] and pooled.shape == (1, 2)
    # the higher-voted review (second) should dominate -> y component larger than x
    assert pooled[0][1] > pooled[0][0]


def test_parse_markdown_bold():
    from book_atlas.label import _parse
    name, desc = _parse("**NAME:** Cozy Mysteries\n**DESC:** Small-town whodunits.")
    assert name == "Cozy Mysteries" and desc == "Small-town whodunits."


def test_cap_keeps_longest():
    import polars as pl
    from book_atlas.reviews import _cap
    rv = pl.DataFrame({"book_id": ["a", "a", "a"], "review_text": ["xx", "xxxxxx", "xxxx"],
                       "rating": [5, 4, 3], "n_votes": [0, 1, 2]})
    out = _cap(rv, 2, "longest")
    assert sorted(out["review_text"].str.len_chars().to_list()) == [4, 6]
