# tests/unit/test_astar.py
import pytest
import numpy as np
from mazellm.astar import AStar
from mazellm.maze import Maze
from mazellm.robot import Position

def test_astar_finds_shortest_path():
    # S 0 0
    # 1 1 0
    # 0 0 E
    # Obstacle forces a specific path
    board = np.array([
        ["S", 0, 0],
        [1, 1, 0],
        [0, 0, "E"]
    ], dtype=object)
    maze = Maze(cols=3, rows=3)
    maze.board = board
    
    path = AStar().solve(maze, Position(0, 0), Position(2, 2))
    assert len(path) == 5
    assert path[-1] == Position(2, 2)

def test_astar_no_path():
    maze = Maze(cols=2, rows=2)
    maze.board = np.array([["S", 1], [1, "E"]], dtype=object)
    with pytest.raises(RuntimeError):
        AStar().solve(maze, Position(0, 0), Position(1, 1))