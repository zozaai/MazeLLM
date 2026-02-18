import pytest
import numpy as np

from mazellm.astar import AStarSolver
from mazellm.maze import Maze
from mazellm.robot import Robot
from mazellm.types import Position


async def _run_solver_collect_positions(solver, maze: Maze, robot: Robot, max_ticks: int = 10_000):
    positions = [robot.position]
    for _ in range(max_ticks):
        res = await solver.next(maze=maze, robot=robot, logger=None)
        positions.append(robot.position)
        if res.done:
            break
    return positions


@pytest.mark.asyncio
async def test_astar_finds_shortest_path():
    # S 0 0
    # 1 1 0
    # 0 0 E
    # Obstacle forces a specific shortest route
    board = np.array(
        [
            ["S", 0, 0],
            [1, 1, 0],
            [0, 0, "E"],
        ],
        dtype=object,
    )
    maze = Maze(cols=3, rows=3)
    maze.board = board

    robot = Robot(maze=maze, position=Position(x=0, y=0))
    solver = AStarSolver()

    positions = await _run_solver_collect_positions(solver, maze, robot, max_ticks=100)
    assert positions[-1] == Position(x=2, y=2)

    # previous test expected 5 nodes => 4 moves
    # positions includes initial position, so length should be 5
    assert len(positions) == 5


@pytest.mark.asyncio
async def test_astar_no_path():
    maze = Maze(cols=2, rows=2)
    maze.board = np.array([["S", 1], [1, "E"]], dtype=object)

    robot = Robot(maze=maze, position=Position(x=0, y=0))
    solver = AStarSolver()

    # Path planning happens on first next() (ensure_path)
    with pytest.raises(RuntimeError, match="No path found"):
        await solver.next(maze=maze, robot=robot, logger=None)
