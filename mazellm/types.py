# mazellm/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["up", "down", "left", "right"]


@dataclass(frozen=True)
class Position:
    x: int
    y: int
