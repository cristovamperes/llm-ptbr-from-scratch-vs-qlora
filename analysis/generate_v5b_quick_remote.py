from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    Path(__file__).resolve().parent / "scripts" / "generate_v5b_quick_remote.py", run_name="__main__"
)

