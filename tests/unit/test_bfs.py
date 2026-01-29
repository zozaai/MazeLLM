# tests/unit/test_bfs.py
import pytest
import numpy as np
from mazellm.bfs import BFS
from mazellm.maze import Maze
from mazellm.robot import Position

def _assert_path_integrity(maze: Maze, path: list[Position], start: Position, end: Position):
    """Common utility to verify path rules."""
    assert path[0] == start, f"Path must start at {start}"
    assert path[-1] == end, f"Path must end at {end}"
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        # Must be 4-connected
        assert abs(a.x - b.x) + abs(a.y - b.y) == 1
        # Must not be a wall
        assert not maze.is_barrier(b.x, b.y), f"Path hits wall at {b}"

def test_bfs_shortest_path_on_ring_maze():
    """Test BFS finds the shortest route when two paths exist."""
    # S 0 0
    # 0 1 0
    # 0 0 E
    # Shortest path is 5 nodes: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) or similar
    board = np.array([
        ["S", 0, 0],
        [0, 1, 0],
        [0, 0, "E"]
    ], dtype=object)
    maze = Maze(cols=3, rows=3)
    maze.board = board
    
    path = BFS().solve(maze, Position(0, 0), Position(2, 2))
    _assert_path_integrity(maze, path, Position(0, 0), Position(2, 2))
    assert len(path) == 5

def test_bfs_single_cell_maze():
    """Edge case: Start and End are the same cell."""
    maze = Maze(cols=1, rows=1)
    maze.board = np.array([["S"]], dtype=object) # Treat S as E
    pos = Position(0, 0)
    path = BFS().solve(maze, pos, pos)
    assert path == [pos]

@pytest.mark.parametrize("seed", [1, 42, 999])
def test_bfs_on_generated_mazes(seed):
    """Ensure BFS always finds a path in solvable generated mazes."""
    maze = Maze(cols=10, rows=10, seed=seed)
    maze.generate_maze()
    
    # Locate S and E from generator
    s_idx = np.argwhere(maze.board == "S")[0]
    e_idx = np.argwhere(maze.board == "E")[0]
    start, end = Position(s_idx[1], s_idx[0]), Position(e_idx[1], e_idx[0])
    
    path = BFS().solve(maze, start, end)
    _assert_path_integrity(maze, path, start, end)

def test_bfs_raises_runtime_error_on_blocked_maze():
    maze = Maze(cols=3, rows=3)
    # Completely encircle 'E' with walls (1)
    # S 0 1
    # 0 0 1
    # 0 1 E
    maze.board = np.array([
        ["S", 0, 1],
        [0, 0, 1],
        [0, 1, "E"]
    ], dtype=object)
    
    solver = BFS()
    start = Position(x=0, y=0)
    end = Position(x=2, y=2)
    
    with pytest.raises(RuntimeError, match="No path found"):
        solver.solve(maze, start, end)