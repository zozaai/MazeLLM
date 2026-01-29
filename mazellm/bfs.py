# mazellm/bfs.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from mazellm.maze import Maze
from mazellm.robot import Position


@dataclass
class BFS:
    """Breadth-First Search path finder (shortest path in #steps)."""

    def _neighbors(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        yield (x + 1, y)
        yield (x - 1, y)
        yield (x, y + 1)
        yield (x, y - 1)

    def solve(self, maze: Maze, start: Position, end: Position) -> list[Position]:
        q: deque[tuple[int, int]] = deque()
        q.append((start.x, start.y))

        parent: dict[tuple[int, int], tuple[int, int] | None] = {(start.x, start.y): None}

        while q:
            x, y = q.popleft()
            if (x, y) == (end.x, end.y):
                break

            for nx, ny in self._neighbors(x, y):
                if not (0 <= nx < maze.n and 0 <= ny < maze.m):
                    continue
                if maze.is_barrier(x=nx, y=ny):
                    continue
                if (nx, ny) in parent:
                    continue

                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))

        if (end.x, end.y) not in parent:
            raise RuntimeError("No path found from S to E (maze may be disconnected).")

        # Reconstruct
        path_xy: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = (end.x, end.y)
        while cur is not None:
            path_xy.append(cur)
            cur = parent[cur]
        path_xy.reverse()

        return [Position(x=x, y=y) for x, y in path_xy]
