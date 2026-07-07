# tests/unit/test_robot.py
import numpy as np

from backend.baselines.maze import Maze
from backend.baselines.robot import Robot
from backend.baselines.types import Position


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
    # Updated: n -> cols, m -> rows
    maze = Maze(cols=5, rows=5, seed=123)
    maze.board = board
    return maze


def test_move_blocked_by_wall_does_not_change_position():
    maze = _make_sample_maze()
    robot = Robot(maze=maze, position=Position(x=0, y=0))

    res = robot.move({"right": 1})
    assert res["status"] is False
    assert res["new_position"] == Position(x=0, y=0)
    assert robot.position == Position(x=0, y=0)


def test_move_success_updates_position():
    maze = _make_sample_maze()
    robot = Robot(maze=maze, position=Position(x=0, y=0))

    res = robot.move({"down": 4})
    assert res["status"] is True
    assert res["new_position"] == Position(x=0, y=4)
    assert robot.position == Position(x=0, y=4)


def test_move_fails_if_path_crosses_wall_midway():
    maze = _make_sample_maze()
    robot = Robot(maze=maze, position=Position(x=0, y=0))

    assert robot.move({"down": 4})["status"] is True
    assert robot.position == Position(x=0, y=4)

    res = robot.move({"right": 3})  # hits wall at (3,4)
    assert res["status"] is False
    assert res["new_position"] == Position(x=0, y=4)
    assert robot.position == Position(x=0, y=4)


def test_move_out_of_bounds_fails():
    maze = _make_sample_maze()
    robot = Robot(maze=maze, position=Position(x=0, y=0))

    res = robot.move({"up": 1})
    assert res["status"] is False
    assert res["new_position"] == Position(x=0, y=0)
    assert robot.position == Position(x=0, y=0)


def test_move_zero_steps_is_success_noop():
    maze = _make_sample_maze()
    robot = Robot(maze=maze, position=Position(x=0, y=0))

    res = robot.move({"down": 0})
    assert res["status"] is True
    assert res["new_position"] == Position(x=0, y=0)
    assert robot.position == Position(x=0, y=0)