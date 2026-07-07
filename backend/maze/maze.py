"""Static maze representation: grid size, walls, start/end cells."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


Cell = tuple[int, int]  # (x, y)


@dataclass
class Maze:
    width: int
    height: int
    walls: set[Cell] = field(default_factory=set)
    start: Cell = (0, 0)
    end: Cell = (0, 0)

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, cell: Cell) -> bool:
        return cell in self.walls

    def is_walkable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.is_wall(cell)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "walls": [list(c) for c in sorted(self.walls)],
            "start": list(self.start),
            "end": list(self.end),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Maze":
        return cls(
            width=data["width"],
            height=data["height"],
            walls={tuple(c) for c in data.get("walls", [])},
            start=tuple(data["start"]),
            end=tuple(data["end"]),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "Maze":
        return cls.from_dict(json.loads(Path(path).read_text()))
