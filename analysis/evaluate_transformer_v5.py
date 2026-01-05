from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    Path(__file__).resolve().parent / "scripts" / "evaluate_transformer_v5.py", run_name="__main__"
)

