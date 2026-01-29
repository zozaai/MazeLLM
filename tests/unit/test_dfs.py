# tests/unit/test_dfs.py
import pytest
import numpy as np
from mazellm.bfs import BFS
from mazellm.dfs import DFS
from mazellm.maze import Maze
from mazellm.robot import Position

def test_dfs_vs_bfs_optimality():
    """
    In an open 5x5 area, DFS will often take a 'scenic' route 
    whereas BFS must take the shortest.
    """
    maze = Maze(cols=5, rows=5)
    maze.board = np.zeros((5, 5), dtype=object)
    start, end = Position(0, 0), Position(4, 4)
    
    dfs_path = DFS().solve(maze, start, end)
    bfs_path = BFS().solve(maze, start, end)
    
    # BFS is shortest: 9 nodes. DFS is likely longer or equal.
    assert len(bfs_path) == 9
    assert len(dfs_path) >= len(bfs_path)
    # Check DFS path validity
    for i in range(len(dfs_path)-1):
        a, b = dfs_path[i], dfs_path[i+1]
        assert abs(a.x - b.x) + abs(a.y - b.y) == 1

def test_dfs_dead_end_backtracking():
    """Test that DFS can back out of a dead end to find the goal."""
    # S 0 0
    # 1 1 0
    # 0 0 E
    # Path must go around the wall.
    board = np.array([
        ["S", 0, 0],
        [1, 1, 0],
        [0, 0, "E"]
    ], dtype=object)
    maze = Maze(cols=3, rows=3)
    maze.board = board
    
    start, end = Position(0, 0), Position(2, 2)
    path = DFS().solve(maze, start, end)
    assert path[-1] == end
    assert Position(0, 1) not in path # This is a wall anyway

def test_dfs_large_sparse_maze():
    """Verify DFS doesn't hit recursion limits or loops on a 20x20 open grid."""
    maze = Maze(cols=20, rows=20)
    maze.board = np.zeros((20, 20), dtype=object)
    start, end = Position(0, 0), Position(19, 19)
    
    path = DFS().solve(maze, start, end)
    assert path[0] == start
    assert path[-1] == end

def test_dfs_unsolvable_raises():
    maze = Maze(cols=2, rows=2)
    maze.board = np.array([["S", 1], [1, "E"]], dtype=object)
    with pytest.raises(RuntimeError):
        DFS().solve(maze, Position(0, 0), Position(1, 1))