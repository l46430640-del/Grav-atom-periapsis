"""Ranges of the gravitational coupling and black hole spin."""
from __future__ import annotations
import math

def validate_alpha_chi(alpha: float, chi: float) -> None:
    if not math.isfinite(alpha) or not 0.0 < alpha <= 0.3:
        raise ValueError("alpha must lie in (0, 0.3] for this baseline")
    if not math.isfinite(chi) or not 0.0 < chi <= 0.999:
        raise ValueError("chi must lie in (0, 0.999]")
