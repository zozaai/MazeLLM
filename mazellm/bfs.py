# mazellm/bfs.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

from mazellm.maze import Maze
from mazellm.solver import Solver, StepResult
from mazellm.robot import Robot
from mazellm.types import Position, Direction


def _neighbors4(x: int, y: int) -> Iterable[tuple[int, int]]:
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _step_to_move(a: Position, b: Position) -> dict[Direction, int]:
    dx = b.x - a.x
    dy = b.y - a.y
    if dx == 1 and dy == 0:
        return {"right": 1}
    if dx == -1 and dy == 0:
        return {"left": 1}
    if dx == 0 and dy == 1:
        return {"down": 1}
    if dx == 0 and dy == -1:
        return {"up": 1}
    raise ValueError(f"Non-adjacent step: {a} -> {b}")


@dataclass
class BFSSolver(Solver):
    """
    BFS solver that computes the shortest path once, then yields 1 move per tick.
    """

    def __init__(self):
        super().__init__(name="bfs")
        self._path: Optional[list[Position]] = None
        self._i: int = 0

    def _solve_path(self, maze: Maze, start: Position, end: Position) -> list[Position]:
        q: deque[tuple[int, int]] = deque()
        q.append((start.x, start.y))
        parent: dict[tuple[int, int], tuple[int, int] | None] = {(start.x, start.y): None}

        while q:
            x, y = q.popleft()
            if (x, y) == (end.x, end.y):
                break

            for nx, ny in _neighbors4(x, y):
                if not (0 <= nx < maze.n and 0 <= ny < maze.m):
                    continue
                if maze.is_barrier(nx, ny):
                    continue
                if (nx, ny) in parent:
                    continue
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))

        if (end.x, end.y) not in parent:
            raise RuntimeError("No path found from S to E (maze may be disconnected).")

        path_xy: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = (end.x, end.y)
        while cur is not None:
            path_xy.append(cur)
            cur = parent[cur]
        path_xy.reverse()
        return [Position(x=x, y=y) for x, y in path_xy]

    def _ensure_path(self, maze: Maze, robot: Robot) -> None:
        if self._path is not None:
            return

        start = maze.find_cell("S")
        end = maze.find_cell("E")
        self._path = self._solve_path(maze, start, end)
        self._i = 0

        # include initial cell as visited
        self.visited.add((robot.position.y, robot.position.x))

    async def next(self, *, maze: Maze, robot: Robot, logger=None) -> StepResult:
        self._ensure_path(maze, robot)

        # already at end?
        if maze.board[robot.position.y, robot.position.x] == "E":
            return StepResult(did_move=False, done=True, message="✅ Already at goal.")

        assert self._path is not None

        # find current index in path, defensive:
        if self._i < len(self._path) and self._path[self._i] != robot.position:
            # resync by searching
            try:
                self._i = self._path.index(robot.position)
            except ValueError:
                return StepResult(did_move=False, done=False, message="⚠️ Robot is off the BFS path.")

        if self._i >= len(self._path) - 1:
            return StepResult(did_move=False, done=True, message="✅ Reached end.")

        cur = self._path[self._i]
        nxt = self._path[self._i + 1]
        move_cmd = _step_to_move(cur, nxt)

        res = robot.move(move_cmd)
        if not res["status"]:
            return StepResult(did_move=False, done=False, message=f"❌ Move failed: {move_cmd}")

        self._i += 1
        rp = robot.position
        rc = (rp.y, rp.x)
        added = []
        if rc not in self.visited:
            self.visited.add(rc)
            added.append(rc)

        done = maze.board[rp.y, rp.x] == "E"
        dir_name, cells = next(iter(move_cmd.items()))
        return StepResult(
            did_move=True,
            done=done,
            message=f"Move: {dir_name} {cells} -> (x={rp.x}, y={rp.y})",
            visited_added_rc=added,
            new_position=rp,
        )
