"""Make backend/ importable when the eval scripts are run directly.

The harness reuses the production modules (`fusion`, `bm25_index`, `tokenizer`)
rather than copying them, so what is measured here is the code that is served.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
