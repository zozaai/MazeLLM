"""Robot state: current position, move history, and fog-of-war memory.

The robot only "knows" a cell's contents (wall/open/end) once it has been
revealed via sense_surroundings() — this is what makes the visualization's
progressive reveal possible on the frontend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .maze import Cell

DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class StepRecord:
    step: int
    sensed: dict[str, str]
    action: str
    position_before: Cell
    position_after: Cell


@dataclass
class Robot:
    position: Cell
    known_cells: dict[Cell, str] = field(default_factory=dict)  # cell -> "open" | "wall" | "end"
    history: list[StepRecord] = field(default_factory=list)

    def record_step(self, sensed: dict[str, str], action: str, new_position: Cell) -> None:
        self.history.append(
            StepRecord(
                step=len(self.history),
                sensed=sensed,
                action=action,
                position_before=self.position,
                position_after=new_position,
            )
        )
        self.position = new_position
