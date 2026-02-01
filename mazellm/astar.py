# mazellm/astar.py
from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Iterable
from mazellm.maze import Maze
from mazellm.robot import Position

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    pos: tuple[int, int] = field(compare=False)

class AStar:
    """A* Search path finder using Manhattan distance heuristic."""

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            yield (x + dx, y + dy)

    def solve(self, maze: Maze, start: Position, end: Position) -> list[Position]:
        start_xy = (start.x, start.y)
        end_xy = (end.x, end.y)

        pq = [PrioritizedItem(0, start_xy)]
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start_xy: None}
        g_score = {start_xy: 0}

        while pq:
            current = heapq.heappop(pq).pos

            if current == end_xy:
                break

            for nx, ny in self._neighbors(*current):
                if not (0 <= nx < maze.n and 0 <= ny < maze.m) or maze.is_barrier(nx, ny):
                    continue

                tentative_g = g_score[current] + 1
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    priority = tentative_g + self._heuristic((nx, ny), end_xy)
                    parent[(nx, ny)] = current
                    heapq.heappush(pq, PrioritizedItem(priority, (nx, ny)))

        if end_xy not in parent:
            raise RuntimeError("No path found from S to E.")

        path: list[Position] = []
        curr = end_xy
        while curr is not None:
            path.append(Position(x=curr[0], y=curr[1]))
            curr = parent[curr]
        path.reverse()
        return path