# mazellm/astar.py
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable, Optional

from mazellm.maze import Maze
from mazellm.solver import Solver, StepResult
from mazellm.robot import Robot
from mazellm.types import Position, Direction


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


@dataclass(order=True)
class _PrioritizedItem:
    priority: int
    pos: tuple[int, int] = field(compare=False)


def _neighbors4(x: int, y: int) -> Iterable[tuple[int, int]]:
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        yield (x + dx, y + dy)


@dataclass
class AStarSolver(Solver):
    """
    A* solver: computes shortest path using Manhattan heuristic, then yields 1 move per tick.
    """

    def __init__(self):
        super().__init__(name="astar")
        self._path: Optional[list[Position]] = None
        self._i: int = 0

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _solve_path(self, maze: Maze, start: Position, end: Position) -> list[Position]:
        start_xy = (start.x, start.y)
        end_xy = (end.x, end.y)

        pq = [_PrioritizedItem(0, start_xy)]
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start_xy: None}
        g_score = {start_xy: 0}

        while pq:
            current = heapq.heappop(pq).pos
            if current == end_xy:
                break

            for nx, ny in _neighbors4(*current):
                if not (0 <= nx < maze.n and 0 <= ny < maze.m) or maze.is_barrier(nx, ny):
                    continue

                tentative_g = g_score[current] + 1
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    priority = tentative_g + self._heuristic((nx, ny), end_xy)
                    parent[(nx, ny)] = current
                    heapq.heappush(pq, _PrioritizedItem(priority, (nx, ny)))

        if end_xy not in parent:
            raise RuntimeError("No path found from S to E.")

        path: list[Position] = []
        curr = end_xy
        while curr is not None:
            path.append(Position(x=curr[0], y=curr[1]))
            curr = parent[curr]
        path.reverse()
        return path

    def _ensure_path(self, maze: Maze, robot: Robot) -> None:
        if self._path is not None:
            return
        start = maze.find_cell("S")
        end = maze.find_cell("E")
        self._path = self._solve_path(maze, start, end)
        self._i = 0
        self.visited.add((robot.position.y, robot.position.x))

    async def next(self, *, maze: Maze, robot: Robot, logger=None) -> StepResult:
        self._ensure_path(maze, robot)

        if maze.board[robot.position.y, robot.position.x] == "E":
            return StepResult(did_move=False, done=True, message="✅ Already at goal.")

        assert self._path is not None

        if self._i < len(self._path) and self._path[self._i] != robot.position:
            try:
                self._i = self._path.index(robot.position)
            except ValueError:
                return StepResult(did_move=False, done=False, message="⚠️ Robot is off the A* path.")

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
