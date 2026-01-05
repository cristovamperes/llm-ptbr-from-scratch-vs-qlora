from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    Path(__file__).resolve().parent / "scripts" / "generate_samples_greedy.py", run_name="__main__"
)

