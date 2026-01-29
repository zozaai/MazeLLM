# tests/unit/test_panels.py
from __future__ import annotations

import numpy as np
import pytest

from mazellm.maze import Maze
from mazellm.visualizer.panels import MazePanel


class FakeTile:
    """Minimal stand-in for a Textual Static widget."""
    def __init__(self):
        self.classes: set[str] = set()

    def set_class(self, enabled: bool, class_name: str) -> None:
        if enabled:
            self.classes.add(class_name)
        else:
            self.classes.discard(class_name)


def _make_maze_from_board(board_2d) -> Maze:
    """
    Helper: build a Maze object with a specific board.
    board_2d is indexed [row][col] like Maze.board.
    """
    rows = len(board_2d)
    cols = len(board_2d[0]) if rows else 0
    # Updated: renamed keyword arguments from n/m to cols/rows
    maze = Maze(cols=cols, rows=rows, seed=0)
    maze.board = np.array(board_2d, dtype=object)
    return maze


def test_render_maze_assigns_classes_start_end_wall_free():
    # 2x2 board:
    # S  0
    # 1  E
    maze = _make_maze_from_board([
        ["S", 0],
        [1, "E"],
    ])

    panel = MazePanel(maze=maze)

    # Inject fake tiles (normally created in compose())
    panel.tiles = [FakeTile() for _ in range(panel.rows * panel.cols)]  # type: ignore
    
    panel.render_maze()

    # (0,0) -> start
    assert "start" in panel.tiles[0].classes
    assert "wall" not in panel.tiles[0].classes
    assert "free" not in panel.tiles[0].classes
    assert "end" not in panel.tiles[0].classes

    # (0,1) -> free
    assert "free" in panel.tiles[1].classes
    assert "start" not in panel.tiles[1].classes
    assert "wall" not in panel.tiles[1].classes
    assert "end" not in panel.tiles[1].classes

    # (1,0) -> wall
    assert "wall" in panel.tiles[2].classes
    assert "free" not in panel.tiles[2].classes
    assert "start" not in panel.tiles[2].classes
    assert "end" not in panel.tiles[2].classes

    # (1,1) -> end
    assert "end" in panel.tiles[3].classes
    assert "start" not in panel.tiles[3].classes
    assert "wall" not in panel.tiles[3].classes
    assert "free" not in panel.tiles[3].classes


def test_render_maze_robot_overlay_is_applied():
    maze = _make_maze_from_board([
        ["S", 0],
        [0, "E"],
    ])

    panel = MazePanel(maze=maze)
    panel.tiles = [FakeTile() for _ in range(panel.rows * panel.cols)] # type: ignore

    # Pretend robot is at (0,1)
    panel.robot_pos = (0, 1)

    panel.render_maze()

    # Robot overlays on top of whatever base class is there
    idx = 0 * panel.cols + 1
    assert "robot" in panel.tiles[idx].classes
    # Base class should still exist too (free in this case)
    assert "free" in panel.tiles[idx].classes


def test_render_maze_clears_previous_classes_before_setting_new():
    # Start with a board that's all free
    maze = _make_maze_from_board([
        [0, 0],
        [0, 0],
    ])

    panel = MazePanel(maze=maze)
    panel.tiles = [FakeTile() for _ in range(panel.rows * panel.cols)]

    panel.render_maze()
    assert all("free" in t.classes for t in panel.tiles)

    # Now mutate board to add a wall and re-render
    panel.maze.board[1, 0] = 1
    panel.render_maze()

    # (1,0) must be wall now, not free
    idx = 1 * panel.cols + 0
    assert "wall" in panel.tiles[idx].classes
    assert "free" not in panel.tiles[idx].classes