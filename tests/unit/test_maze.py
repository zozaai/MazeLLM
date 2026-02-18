import numpy as np
from collections import deque

from mazellm.maze import Maze


def _is_walkable(v: object) -> bool:
    # In this project, 1 is a wall; everything else is walkable (0, "S", "E")
    return v != 1


def _reachable(board: np.ndarray, start_rc: tuple[int, int], goal_rc: tuple[int, int]) -> bool:
    sr, sc = start_rc
    gr, gc = goal_rc

    q: deque[tuple[int, int]] = deque([(sr, sc)])
    seen: set[tuple[int, int]] = {(sr, sc)}

    rows, cols = board.shape

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True

        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            if (rr, cc) in seen:
                continue
            if not _is_walkable(board[rr, cc]):
                continue
            seen.add((rr, cc))
            q.append((rr, cc))

    return False


def test_generate_maze_sets_shape():
    maze = Maze(cols=8, rows=4, seed=123)
    maze.generate_maze()
    assert maze.board.shape == (4, 8)  # (rows=m, cols=n)


def test_generate_maze_places_start_and_end_and_end_is_reachable():
    maze = Maze(cols=5, rows=5, seed=123)
    maze.generate_maze()

    # Start is fixed at top-left
    assert maze.board[0, 0] == "S"

    # End can be anywhere now, but must exist exactly once
    end_pos = maze.find_cell("E")
    end_r, end_c = end_pos.y, end_pos.x

    # E should be on a walkable cell (obvious, but makes intent explicit)
    assert maze.board[end_r, end_c] == "E"

    # Most important invariant now: E must be reachable from S
    assert _reachable(maze.board, (0, 0), (end_r, end_c)) is True


def test_is_barrier_out_of_bounds_is_true():
    maze = Maze(cols=5, rows=5, seed=123)
    maze.generate_maze()

    assert maze.is_barrier(-1, 0) is True
    assert maze.is_barrier(0, -1) is True
    assert maze.is_barrier(maze.n, 0) is True
    assert maze.is_barrier(0, maze.m) is True


def test_is_barrier_respects_board_values():
    maze = Maze(cols=5, rows=5, seed=123)
    maze.generate_maze()

    # force known cells
    maze.board[2, 3] = 1  # wall
    maze.board[2, 4] = 0  # free

    assert maze.is_barrier(3, 2) is True   # (x=3,y=2) -> board[2,3]
    assert maze.is_barrier(4, 2) is False  # (x=4,y=2) -> board[2,4]


def test_generate_maze_is_reproducible_with_seed():
    m1 = Maze(cols=15, rows=15, seed=999)
    m2 = Maze(cols=15, rows=15, seed=999)

    m1.generate_maze()
    m2.generate_maze()

    assert np.array_equal(m1.board, m2.board)


def test_generate_maze_different_seeds_usually_different():
    m1 = Maze(cols=15, rows=15, seed=1)
    m2 = Maze(cols=15, rows=15, seed=2)

    m1.generate_maze()
    m2.generate_maze()

    # Extremely unlikely to match
    assert not np.array_equal(m1.board, m2.board)