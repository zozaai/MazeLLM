# mazellm/maze.py
from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np

from mazellm.types import Position  # ✅ from types now


class Maze:
    def __init__(self, cols: int = 5, rows: int = 5, seed: int | None = None):
        self.n = cols
        self.m = rows
        self.board = np.ones((self.m, self.n), dtype=object)
        self._rng = random.Random(seed)

    def _neighbors4(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        yield (x + 1, y)
        yield (x - 1, y)
        yield (x, y + 1)
        yield (x, y - 1)

    def _is_wall_value(self, v: object) -> bool:
        return v == 1

    def _is_walkable_value(self, v: object) -> bool:
        return not self._is_wall_value(v)

    def _bfs_distances_from(self, sx: int, sy: int) -> dict[tuple[int, int], int]:
        if not (0 <= sx < self.n and 0 <= sy < self.m):
            return {}
        if not self._is_walkable_value(self.board[sy, sx]):
            return {}

        q: deque[tuple[int, int]] = deque()
        q.append((sx, sy))
        dist: dict[tuple[int, int], int] = {(sx, sy): 0}

        while q:
            x, y = q.popleft()
            for nx, ny in self._neighbors4(x, y):
                if not (0 <= nx < self.n and 0 <= ny < self.m):
                    continue
                if not self._is_walkable_value(self.board[ny, nx]):
                    continue
                if (nx, ny) in dist:
                    continue
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
        return dist

    def generate_maze(self) -> None:
        self.board[:, :] = 1

        def carve(x: int, y: int) -> None:
            self.board[y, x] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            self._rng.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= ny < self.m and 0 <= nx < self.n and self.board[ny, nx] == 1:
                    self.board[y + dy // 2, x + dx // 2] = 0
                    carve(nx, ny)

        carve(0, 0)

        dist = self._bfs_distances_from(0, 0)
        if not dist:
            self.board[0, 0] = 0
            dist = self._bfs_distances_from(0, 0)

        (ex, ey), _ = max(dist.items(), key=lambda kv: kv[1])

        for y in range(self.m):
            for x in range(self.n):
                if self.board[y, x] in ("S", "E"):
                    self.board[y, x] = 0

        self.board[0, 0] = "S"
        self.board[ey, ex] = "E"

    def is_barrier(self, x: int, y: int) -> bool:
        if 0 <= y < self.m and 0 <= x < self.n:
            return self.board[y, x] == 1
        return True

    # ✅ NEW
    def find_cell(self, value: object) -> Position:
        for y in range(self.m):
            for x in range(self.n):
                if self.board[y, x] == value:
                    return Position(x=x, y=y)
        raise ValueError(f"Could not find {value!r} in maze.board")
