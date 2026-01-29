# tests/unit/test_dfs.py
import numpy as np

from mazellm.bfs import BFS
from mazellm.dfs import DFS
from mazellm.maze import Maze
from mazellm.robot import Position


def _make_sample_maze() -> Maze:
    board = np.array(
        [
            ["S", 1, 0, 0, 0],
            [0,   1, 0, 1, 0],
            [0,   1, 0, 1, 0],
            [0,   1, 0, 1, 0],
            [0,   0, 0, 1, "E"],
        ],
        dtype=object,
    )
    maze = Maze(cols=5, rows=5, seed=123)
    maze.board = board
    return maze


def _is_walkable(maze: Maze, x: int, y: int) -> bool:
    return (0 <= x < maze.n) and (0 <= y < maze.m) and (not maze.is_barrier(x=x, y=y))


def _assert_path_is_valid(maze: Maze, path: list[Position], start: Position, end: Position) -> None:
    assert len(path) >= 1
    assert path[0] == start
    assert path[-1] == end

    for p in path:
        assert _is_walkable(maze, p.x, p.y)

    for a, b in zip(path, path[1:]):
        manhattan = abs(a.x - b.x) + abs(a.y - b.y)
        assert manhattan == 1


def test_dfs_finds_a_valid_path():
    maze = _make_sample_maze()
    start = Position(x=0, y=0)
    end = Position(x=4, y=4)

    solver = DFS()
    path = solver.solve(maze=maze, start=start, end=end)

    _assert_path_is_valid(maze, path, start, end)


def test_dfs_path_is_not_shorter_than_bfs():
    maze = _make_sample_maze()
    start = Position(x=0, y=0)
    end = Position(x=4, y=4)

    bfs_path = BFS().solve(maze=maze, start=start, end=end)
    dfs_path = DFS().solve(maze=maze, start=start, end=end)

    # DFS is "any path", BFS is shortest => DFS should not beat BFS.
    assert len(dfs_path) >= len(bfs_path)


def test_dfs_raises_if_end_unreachable():
    maze = _make_sample_maze()

    # Wall off the goal region similarly to BFS test
    maze.board[3, 4] = 1  # (x=4,y=3)
    maze.board[4, 3] = 1  # (x=3,y=4)

    start = Position(x=0, y=0)
    end = Position(x=4, y=4)

    solver = DFS()

    try:
        solver.solve(maze=maze, start=start, end=end)
        assert False, "Expected RuntimeError for unreachable end"
    except RuntimeError:
        pass
