"""Make the sibling moshi repo importable for the test suite."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for candidate in (_ROOT / "moshi", _ROOT.parent / "moshi"):
    if (candidate / "moshi").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
