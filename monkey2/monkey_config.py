from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MonkeyConfig:
    """
    Monkey2 Config.
    """

    # --- action pruning ---
    # Number of actions to consider at Min nodes
    top_k: int = 3
    
    # When candidates count is below this threshold, increase Top K
    small_candidates_threshold: int = 10
    
    # The increased Top K value
    top_k_when_small_candidates: int = 3

    # --- symmetry pruning ---
    symmetry_enabled: bool = True
    # Decimal places for rounding IG to detect ties
    symmetry_gain_round_ndigits: int = 6
    symmetry_consider_state: bool = True

    # --- search ---
    # Show progress bar
    progress_enabled: bool = True

    # --- progress estimate ---
    progress_depth_hint: int = 14
    
    # Stability
    deterministic: bool = True

