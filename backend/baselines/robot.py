# backend/baselines/robot.py
from __future__ import annotations

from typing import Dict, TypedDict

from backend.baselines.maze import Maze
from backend.baselines.types import Direction, Position  # ✅ moved here


class MoveResult(TypedDict):
    status: bool
    new_position: Position


class Robot:
    """
    Robot bound to a Maze.

    - Position is (x, y) where:
        x = column index (0..maze.n-1)
        y = row index    (0..maze.m-1)

    Maze uses board[y, x].
    """

    def __init__(self, maze: Maze, position: Position | None = None):
        self.maze: Maze = maze
        self.position: Position = position or Position(x=0, y=0)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.maze.n and 0 <= y < self.maze.m

    def _is_walkable(self, x: int, y: int) -> bool:
        return self._in_bounds(x, y) and (not self.maze.is_barrier(x=x, y=y))

    def move(self, direction: Dict[Direction, int]) -> MoveResult:
        if not direction:
            return {"status": False, "new_position": Position(self.position.x, self.position.y)}

        dir_name, cells = next(iter(direction.items()))

        if dir_name not in ("up", "down", "left", "right"):
            return {"status": False, "new_position": Position(self.position.x, self.position.y)}

        if not isinstance(cells, int) or cells < 0:
            return {"status": False, "new_position": Position(self.position.x, self.position.y)}

        if cells == 0:
            return {"status": True, "new_position": Position(self.position.x, self.position.y)}

        x0, y0 = self.position.x, self.position.y
        x, y = x0, y0

        for _ in range(cells):
            if dir_name == "up":
                y -= 1
            elif dir_name == "down":
                y += 1
            elif dir_name == "left":
                x -= 1
            else:
                x += 1

            if not self._is_walkable(x, y):
                return {"status": False, "new_position": Position(x0, y0)}

        self.position = Position(x=x, y=y)
        return {"status": True, "new_position": Position(x=x, y=y)}
