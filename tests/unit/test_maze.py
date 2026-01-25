import numpy as np
import pytest

from mazellm.maze import Maze


def test_generate_maze_sets_shape():
    maze = Maze(n=8, m=4, seed=123)
    maze.generate_maze()

    assert maze.board.shape == (4, 8)  # (rows=m, cols=n)


def test_generate_maze_places_start_and_end():
    maze = Maze(n=5, m=5, seed=123)
    maze.generate_maze()

    assert maze.board[0, 0] == "S"
    assert maze.board[maze.m - 1, maze.n - 1] == "E"


def test_is_barrier_out_of_bounds_is_true():
    maze = Maze(n=5, m=5, seed=123)
    maze.generate_maze()

    assert maze.is_barrier(-1, 0) is True
    assert maze.is_barrier(0, -1) is True
    assert maze.is_barrier(maze.n, 0) is True
    assert maze.is_barrier(0, maze.m) is True


def test_is_barrier_respects_board_values():
    maze = Maze(n=5, m=5, seed=123)
    maze.generate_maze()

    # force known cells
    maze.board[2, 3] = 1  # wall
    maze.board[2, 4] = 0  # free

    assert maze.is_barrier(3, 2) is True   # (x=3,y=2) -> board[2,3]
    assert maze.is_barrier(4, 2) is False  # (x=4,y=2) -> board[2,4]


def test_generate_maze_is_reproducible_with_seed():
    m1 = Maze(n=15, m=15, seed=999)
    m2 = Maze(n=15, m=15, seed=999)

    m1.generate_maze()
    m2.generate_maze()

    # board dtype is object, so use array_equal
    assert np.array_equal(m1.board, m2.board)


def test_generate_maze_different_seeds_usually_different():
    m1 = Maze(n=15, m=15, seed=1)
    m2 = Maze(n=15, m=15, seed=2)

    m1.generate_maze()
    m2.generate_maze()

    # It's theoretically possible to match, but extremely unlikely for this generator
    assert not np.array_equal(m1.board, m2.board)
