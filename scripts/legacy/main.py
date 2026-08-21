"""Compatibility entry point for the former root-level main.py.

Use ``python src/run_one_full_pipeline.py <image-or-id>`` for new workflows.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_one_full_pipeline import main


if __name__ == "__main__":
    main()
