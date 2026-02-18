# mazellm/solver.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from mazellm.types import Position  # ✅


@dataclass
class StepResult:
    did_move: bool
    done: bool
    message: str = ""
    visited_added_rc: list[tuple[int, int]] = None
    new_position: Optional[Position] = None

    def __post_init__(self):
        if self.visited_added_rc is None:
            self.visited_added_rc = []


class Solver(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name
        self.visited: set[tuple[int, int]] = set()

    def reset(self) -> None:
        self.visited.clear()

    @abstractmethod
    async def next(self, *, maze, robot, logger=None) -> StepResult:
        raise NotImplementedError
