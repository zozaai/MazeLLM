# mazellm/main.py
from __future__ import annotations

import argparse

from mazellm.maze import Maze
from mazellm.robot import Robot
from mazellm.types import Position
from mazellm.visualizer.panels import SolveMazePanel
from mazellm.bfs import BFSSolver
from mazellm.dfs import DFSSolver
from mazellm.astar import AStarSolver
from mazellm.agent import LLMSolver


def _build_solver(method: str, llm_model: str):
    method = method.lower().strip()
    if method == "bfs":
        return BFSSolver()
    if method == "dfs":
        return DFSSolver()
    if method == "astar":
        return AStarSolver()
    if method == "llm":
        return LLMSolver(model=llm_model)
    raise SystemExit(f"Unknown method: {method!r}. Use one of: bfs, dfs, astar, llm")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MazeLLM: solve maze and visualize step-by-step.")
    p.add_argument("--c", "--cols", dest="cols", type=int, default=15, help="Maze width (columns)")
    p.add_argument("--r", "--rows", dest="rows", type=int, default=15, help="Maze height (rows)")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    p.add_argument("--interval", type=float, default=0.15, help="Seconds between steps")
    p.add_argument("-m", "--method", type=str, default="bfs", help="Solve method: bfs | dfs | astar | llm")
    p.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model name (for -m llm)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.cols <= 0 or args.rows <= 0:
        raise SystemExit("rows and columns must be positive integers.")

    maze = Maze(cols=args.cols, rows=args.rows, seed=args.seed)
    maze.generate_maze()

    start = maze.find_cell("S")
    robot = Robot(maze=maze, position=Position(x=start.x, y=start.y))

    solver = _build_solver(args.method, args.llm_model)

    SolveMazePanel(
        maze=maze,
        robot=robot,
        solver=solver,
        interval_s=args.interval,
    ).run()


if __name__ == "__main__":
    main()
