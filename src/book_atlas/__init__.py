"""Semantic Book Atlas — content + reviews clustering for the most-reviewed Goodreads books.

Environment guards are set here, before any native library (numpy/torch/MKL) is imported,
so `python -m book_atlas.<stage>` runs cleanly on Windows without manual env juggling.
A `.env` file in the project root is loaded first (API keys, etc.):
  * KMP_DUPLICATE_LIB_OK — prevents the OpenMP/libiomp5md.dll double-load crash (numpy+torch).
  * HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE — load the cached model without contacting Hugging Face.
    (Set to "0" in your shell if you still need to DOWNLOAD a model for the first time.)
"""
import os as _os

# Load API keys and settings from a .env file in the project root (or any parent
# directory), so no setx / system-wide env vars are needed. Real environment
# variables still take precedence (override=False). Falls back silently if
# python-dotenv isn't installed.
try:
    from dotenv import load_dotenv as _load_dotenv, find_dotenv as _find_dotenv
    _load_dotenv(_find_dotenv(usecwd=True), override=False)
except ImportError:
    pass

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__version__ = "0.5.0"
