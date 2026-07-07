"""Random solvable maze generation.

Cells are blocked/open (not walls-between-cells), so generation is simple:
pick random non-start/end cells as walls up to the target density, keep the
layout only if a path from start to end still exists, and retry otherwise.
"""
from __future__ import annotations

import random
from collections import deque

from .maze import Maze

MAX_ATTEMPTS = 200


def generate_random_maze(
    width: int,
    height: int,
    wall_density: float = 0.25,
    seed: int | None = None,
) -> Maze:
    """Generate a random NxM maze guaranteed to have a path from start to end.

    Args:
        width: number of columns.
        height: number of rows.
        wall_density: rough fraction of non-start/end cells to block.
        seed: RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    start = (0, 0)
    end = (width - 1, height - 1)
    candidates = [(x, y) for y in range(height) for x in range(width) if (x, y) not in (start, end)]
    target_wall_count = round(wall_density * len(candidates))

    for _ in range(MAX_ATTEMPTS):
        walls = set(rng.sample(candidates, min(target_wall_count, len(candidates))))
        maze = Maze(width=width, height=height, walls=walls, start=start, end=end)
        if is_solvable(maze):
            return maze

    return Maze(width=width, height=height, walls=set(), start=start, end=end)


def is_solvable(maze: Maze) -> bool:
    """BFS from maze.start to maze.end over walkable cells."""
    if maze.start == maze.end:
        return True
    frontier = deque([maze.start])
    visited = {maze.start}
    while frontier:
        x, y = frontier.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            next_cell = (x + dx, y + dy)
            if next_cell in visited or not maze.is_walkable(next_cell):
                continue
            if next_cell == maze.end:
                return True
            visited.add(next_cell)
            frontier.append(next_cell)
    return False
