from pathlib import Path

from backend.maze.maze import Maze

SAMPLE_MAZE = Path(__file__).parent.parent / "mazes" / "sample_maze.json"


def test_load_sample_maze():
    maze = Maze.load_json(SAMPLE_MAZE)
    assert maze.width == 8
    assert maze.height == 8
    assert maze.start == (0, 0)
    assert maze.end == (7, 7)
    assert (0, 1) in maze.walls


def test_in_bounds():
    maze = Maze(width=4, height=4)
    assert maze.in_bounds((0, 0))
    assert maze.in_bounds((3, 3))
    assert not maze.in_bounds((4, 0))
    assert not maze.in_bounds((-1, 0))


def test_is_walkable():
    maze = Maze(width=4, height=4, walls={(1, 0)})
    assert maze.is_walkable((0, 0))
    assert not maze.is_walkable((1, 0))
    assert not maze.is_walkable((4, 4))


def test_round_trip_json(tmp_path):
    maze = Maze(width=3, height=3, walls={(1, 1)}, start=(0, 0), end=(2, 2))
    out = tmp_path / "m.json"
    maze.save_json(out)
    loaded = Maze.load_json(out)
    assert loaded == maze
