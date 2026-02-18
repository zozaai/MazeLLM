import pytest
import numpy as np

from mazellm.bfs import BFSSolver
from mazellm.maze import Maze
from mazellm.robot import Robot
from mazellm.types import Position


def _assert_walkable_and_adjacent(maze: Maze, positions: list[Position]):
    for i in range(len(positions) - 1):
        a, b = positions[i], positions[i + 1]
        assert abs(a.x - b.x) + abs(a.y - b.y) in (0, 1)  # 0 allowed if a tick didn't move
        assert not maze.is_barrier(b.x, b.y), f"Hit wall at {b}"


async def _run_solver_collect_positions(solver, maze: Maze, robot: Robot, max_ticks: int = 10_000):
    positions = [robot.position]
    for _ in range(max_ticks):
        res = await solver.next(maze=maze, robot=robot, logger=None)
        positions.append(robot.position)
        if res.done:
            break
    return positions


@pytest.mark.asyncio
async def test_bfs_shortest_path_on_ring_maze():
    # S 0 0
    # 0 1 0
    # 0 0 E
    board = np.array(
        [
            ["S", 0, 0],
            [0, 1, 0],
            [0, 0, "E"],
        ],
        dtype=object,
    )
    maze = Maze(cols=3, rows=3)
    maze.board = board

    robot = Robot(maze=maze, position=Position(x=0, y=0))
    solver = BFSSolver()

    positions = await _run_solver_collect_positions(solver, maze, robot, max_ticks=100)
    _assert_walkable_and_adjacent(maze, positions)

    assert positions[-1] == Position(x=2, y=2)

    # old test expected 5 nodes => positions length should be 5
    assert len(positions) == 5


@pytest.mark.asyncio
async def test_bfs_small_maze_start_next_to_end():
    # 1x2: S E
    board = np.array([["S", "E"]], dtype=object)
    maze = Maze(cols=2, rows=1)
    maze.board = board

    robot = Robot(maze=maze, position=Position(x=0, y=0))
    solver = BFSSolver()

    positions = await _run_solver_collect_positions(solver, maze, robot, max_ticks=10)
    assert positions[-1] == Position(x=1, y=0)
    assert len(positions) == 2  # [S, E]


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 42, 999])
async def test_bfs_on_generated_mazes(seed):
    maze = Maze(cols=10, rows=10, seed=seed)
    maze.generate_maze()

    start = maze.find_cell("S")
    robot = Robot(maze=maze, position=Position(x=start.x, y=start.y))
    solver = BFSSolver()

    positions = await _run_solver_collect_positions(solver, maze, robot, max_ticks=10_000)
    _assert_walkable_and_adjacent(maze, positions)

    assert maze.board[positions[-1].y, positions[-1].x] == "E"


@pytest.mark.asyncio
async def test_bfs_raises_runtime_error_on_blocked_maze():
    maze = Maze(cols=3, rows=3)
    maze.board = np.array(
        [
            ["S", 0, 1],
            [0, 0, 1],
            [0, 1, "E"],
        ],
        dtype=object,
    )

    robot = Robot(maze=maze, position=Position(x=0, y=0))
    solver = BFSSolver()

    with pytest.raises(RuntimeError, match="No path found"):
        await solver.next(maze=maze, robot=robot, logger=None)
