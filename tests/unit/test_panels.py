from __future__ import annotations

import numpy as np

from mazellm.maze import Maze
from mazellm.robot import Robot
from mazellm.solver import Solver, StepResult
from mazellm.types import Position
from mazellm.visualizer.panels import SolveMazePanel


class FakeTile:
    """Minimal stand-in for a Textual Static widget."""
    def __init__(self):
        self.classes: set[str] = set()

    def set_class(self, enabled: bool, class_name: str) -> None:
        if enabled:
            self.classes.add(class_name)
        else:
            self.classes.discard(class_name)


class DummySolver(Solver):
    def __init__(self):
        super().__init__(name="dummy")

    async def next(self, *, maze, robot, logger=None) -> StepResult:
        return StepResult(did_move=False, done=False, message="")


def _make_maze_from_board(board_2d) -> Maze:
    rows = len(board_2d)
    cols = len(board_2d[0]) if rows else 0
    maze = Maze(cols=cols, rows=rows, seed=0)
    maze.board = np.array(board_2d, dtype=object)
    return maze


def _make_panel_for_board(board_2d) -> SolveMazePanel:
    maze = _make_maze_from_board(board_2d)
    robot = Robot(maze=maze, position=Position(x=0, y=0))
    solver = DummySolver()
    panel = SolveMazePanel(maze=maze, robot=robot, solver=solver, interval_s=0.0)

    # Inject fake tiles (normally created in compose())
    panel.tiles = [FakeTile() for _ in range(panel.rows * panel.cols)]  # type: ignore
    return panel


def test_render_maze_assigns_classes_start_end_wall_free():
    # 2x2:
    # S  0
    # 1  E
    panel = _make_panel_for_board([
        ["S", 0],
        [1, "E"],
    ])

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
    panel = _make_panel_for_board([
        ["S", 0],
        [0, "E"],
    ])

    # Pretend robot is at (0,1)
    panel.robot_pos = (0, 1)
    panel.render_maze()

    idx = 0 * panel.cols + 1
    assert "robot" in panel.tiles[idx].classes
    assert "free" in panel.tiles[idx].classes


def test_render_maze_clears_previous_classes_before_setting_new():
    panel = _make_panel_for_board([
        [0, 0],
        [0, 0],
    ])

    panel.render_maze()
    assert all("free" in t.classes for t in panel.tiles)

    # mutate board then re-render
    panel.maze.board[1, 0] = 1
    panel.render_maze()

    idx = 1 * panel.cols + 0
    assert "wall" in panel.tiles[idx].classes
    assert "free" not in panel.tiles[idx].classes
