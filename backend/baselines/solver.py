# backend/baselines/solver.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from backend.baselines.types import Position  # ✅


@dataclass
class StepResult:
    did_move: bool
    done: bool
    message: str = ""
    new_position: Optional[Position] = None


class Solver(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def next(self, *, maze, robot) -> StepResult:
        raise NotImplementedError
