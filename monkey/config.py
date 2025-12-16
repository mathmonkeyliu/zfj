from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MonkeyConfig:
    # --- Core search controls ---
    top_k: int = 1

    # --- Progress reporting ---
    progress_enabled: bool = True
    progress_every_sec: float = 0.5  # throttle prints
    # Progress estimator: est_total_nodes ≈ (top_k*3)^(avg_depth)

    # --- Tree logging / caching ---
    tree_log_depth: int = 40  # plies (MIN action = 1 ply, MAX outcome = 1 ply)

    cache_enabled: bool = True
    cache_dir: Path = Path(".cache") / "monkey"


