from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(Path(__file__).resolve().parent / "scripts" / "validate_dataset.py", run_name="__main__")

