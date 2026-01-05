from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def try_get_git_commit(root: Path) -> Optional[str]:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return commit or None
    except Exception:
        return None


def collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", None)
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_count"] = torch.cuda.device_count()
            info["gpu_model"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            info["cuda_available"] = False
    except Exception:
        pass
    for pkg in ("transformers", "datasets", "peft", "bitsandbytes", "accelerate"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", None)
        except Exception:
            continue
    return info


@dataclass(frozen=True)
class RunTimers:
    started_at: str
    ended_at: str
    duration_sec: float


def run_timers(start_ts: float, end_ts: float) -> RunTimers:
    return RunTimers(
        started_at=datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        ended_at=datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        duration_sec=end_ts - start_ts,
    )
