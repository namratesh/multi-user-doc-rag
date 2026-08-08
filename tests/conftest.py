"""Puts `backend/` on sys.path so tests can `import src....` the same way
the app's own non-`api` modules do (see the import-path note in the repo
README) -- without needing PYTHONPATH set externally.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
