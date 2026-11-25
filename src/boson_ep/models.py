"""Quantum numbers of a bound scalar state."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class State:
    n: int
    l: int
    m: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        if not 0 <= self.l < self.n:
            raise ValueError("state must satisfy 0 <= l < n")
        if abs(self.m) > self.l:
            raise ValueError("state must satisfy |m| <= l")

    @property
    def label(self) -> str:
        return f"{self.n}{self.l}{self.m:+d}"
