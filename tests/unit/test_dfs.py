import pytest
import numpy as np

from mazellm.bfs import BFSSolver
from mazellm.dfs import DFSSolver
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
async def test_dfs_vs_bfs_optimality():
    maze = Maze(cols=5, rows=5)
    maze.board = np.zeros((5, 5), dtype=object)
    maze.board[0, 0] = "S"
    maze.board[4, 4] = "E"

    # DFS
    robot_dfs = Robot(maze=maze, position=Position(x=0, y=0))
    dfs = DFSSolver()
    dfs_positions = await _run_solver_collect_positions(dfs, maze, robot_dfs, max_ticks=10_000)

    # BFS
    robot_bfs = Robot(maze=maze, position=Position(x=0, y=0))
    bfs = BFSSolver()
    bfs_positions = await _run_solver_collect_positions(bfs, maze, robot_bfs, max_ticks=10_000)

    # BFS shortest on open grid: (0,0) -> (4,4) needs 8 moves => 9 nodes
    assert len(bfs_positions) == 9
    assert len(dfs_positions) >= len(bfs_positions)
    assert dfs_positions[-1] == Position(x=4, y=4)


@pytest.mark.asyncio
async def test_dfs_dead_end_backtracking():
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
    dfs = DFSSolver()

    positions = await _run_solver_collect_positions(dfs, maze, robot, max_ticks=1_000)
    assert positions[-1] == Position(x=2, y=2)


@pytest.mark.asyncio
async def test_dfs_large_sparse_maze():
    maze = Maze(cols=20, rows=20)
    maze.board = np.zeros((20, 20), dtype=object)
    maze.board[0, 0] = "S"
    maze.board[19, 19] = "E"

    robot = Robot(maze=maze, position=Position(x=0, y=0))
    dfs = DFSSolver()

    positions = await _run_solver_collect_positions(dfs, maze, robot, max_ticks=100_000)
    assert positions[0] == Position(x=0, y=0)
    assert positions[-1] == Position(x=19, y=19)


@pytest.mark.asyncio
async def test_dfs_unsolvable_raises():
    maze = Maze(cols=2, rows=2)
    maze.board = np.array([["S", 1], [1, "E"]], dtype=object)

    robot = Robot(maze=maze, position=Position(x=0, y=0))
    dfs = DFSSolver()

    with pytest.raises(RuntimeError, match="No path found"):
        await dfs.next(maze=maze, robot=robot, logger=None)
