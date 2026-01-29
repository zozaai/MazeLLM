# tests/unit/test_bfs.py
import numpy as np

from mazellm.bfs import BFS
from mazellm.maze import Maze
from mazellm.robot import Position


def _make_sample_maze() -> Maze:
    """
    Same sample layout used in test_robot.py (easy to reason about).

    Board is indexed [row, col] == [y, x]
    S at (x=0,y=0), E at (x=4,y=4)
    """
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

    # every node is walkable and every step is 4-neighbor adjacent
    for p in path:
        assert _is_walkable(maze, p.x, p.y)

    for a, b in zip(path, path[1:]):
        manhattan = abs(a.x - b.x) + abs(a.y - b.y)
        assert manhattan == 1


def test_bfs_finds_a_valid_path_and_is_shortest_on_sample_maze():
    maze = _make_sample_maze()
    start = Position(x=0, y=0)
    end = Position(x=4, y=4)

    solver = BFS()
    path = solver.solve(maze=maze, start=start, end=end)

    _assert_path_is_valid(maze, path, start, end)

    # On this specific maze, the shortest path is known.
    # It must go down to row 4 to bypass the wall column, then route via top row and down.
    assert len(path) == 17  # 16 moves


def test_bfs_raises_if_end_unreachable():
    maze = _make_sample_maze()

    # Make E unreachable by turning its only column approach into walls
    # (wall off (4,3) and (3,4), and keep boundaries)
    maze.board[3, 4] = 1  # (x=4,y=3)
    maze.board[4, 3] = 1  # (x=3,y=4)

    start = Position(x=0, y=0)
    end = Position(x=4, y=4)

    solver = BFS()

    try:
        solver.solve(maze=maze, start=start, end=end)
        assert False, "Expected RuntimeError for unreachable end"
    except RuntimeError:
        pass
