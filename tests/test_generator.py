from backend.maze.generator import generate_random_maze, is_solvable
from backend.maze.maze import Maze


def test_generated_maze_is_solvable():
    maze = generate_random_maze(width=10, height=10, wall_density=0.3, seed=42)
    assert maze.width == 10
    assert maze.height == 10
    assert is_solvable(maze)


def test_generated_maze_is_deterministic_with_seed():
    a = generate_random_maze(width=6, height=6, wall_density=0.25, seed=7)
    b = generate_random_maze(width=6, height=6, wall_density=0.25, seed=7)
    assert a.walls == b.walls


def test_start_and_end_are_never_walls():
    maze = generate_random_maze(width=5, height=5, wall_density=0.6, seed=1)
    assert maze.start not in maze.walls
    assert maze.end not in maze.walls


def test_is_solvable_detects_blocked_maze():
    maze = Maze(width=3, height=1, walls={(1, 0)}, start=(0, 0), end=(2, 0))
    assert not is_solvable(maze)


def test_is_solvable_detects_open_path():
    maze = Maze(width=3, height=1, walls=set(), start=(0, 0), end=(2, 0))
    assert is_solvable(maze)
