# mazellm/main.py
from __future__ import annotations

import argparse
from collections.abc import Iterable

from mazellm.maze import Maze
from mazellm.robot import Robot, Position, Direction
from mazellm.visualizer.panels import MazePanel
from mazellm.bfs import BFS
from mazellm.dfs import DFS
from mazellm.astar import AStar

def _find_cell(maze: Maze, value: object) -> Position:
    for y in range(maze.m):
        for x in range(maze.n):
            if maze.board[y, x] == value:
                return Position(x=x, y=y)
    raise ValueError(f"Could not find {value!r} in maze.board")


def _step_to_move(a: Position, b: Position) -> dict[Direction, int]:
    dx = b.x - a.x
    dy = b.y - a.y
    if dx == 1 and dy == 0:
        return {"right": 1}
    if dx == -1 and dy == 0:
        return {"left": 1}
    if dx == 0 and dy == 1:
        return {"down": 1}
    if dx == 0 and dy == -1:
        return {"up": 1}
    raise ValueError(f"Non-adjacent step: {a} -> {b}")


class SolveMazePanel(MazePanel):
    def __init__(
        self,
        maze: Maze,
        robot: Robot,
        path: list[Position],
        interval_s: float = 0.2,
        method: str = "bfs",
        **kwargs,
    ):
        super().__init__(maze=maze, **kwargs)
        self.robot = robot
        self.path = path
        self.interval_s = float(interval_s)
        self.method = method
        self._step_i = 0

    def on_mount(self) -> None:
        super().on_mount()

        # Initial robot placement at start
        self.set_robot_position(self.robot.position.y, self.robot.position.x)
        self.log_info(f"Method: {self.method}")
        self.log_info(f"Start: (x={self.robot.position.x}, y={self.robot.position.y})")
        self.log_info(f"Goal : (x={self.path[-1].x}, y={self.path[-1].y})")
        self.log_info(f"Path length: {len(self.path)}")

        self.set_interval(self.interval_s, self._tick)

    def _tick(self) -> None:
        if self._step_i >= len(self.path) - 1:
            self.log_info("✅ Reached end.")
            return

        cur = self.path[self._step_i]
        nxt = self.path[self._step_i + 1]
        move_cmd = _step_to_move(cur, nxt)

        res = self.robot.move(move_cmd)
        if not res["status"]:
            self.log_info(f"❌ Move failed: {move_cmd} from (x={cur.x}, y={cur.y})")
            return

        self._step_i += 1
        new_pos = self.robot.position

        self.set_robot_position(new_pos.y, new_pos.x)
        dir_name, cells = next(iter(move_cmd.items()))
        self.log_info(f"Move: {dir_name} {cells} -> (x={new_pos.x}, y={new_pos.y})")


def _build_solver(method: str):
    method = method.lower().strip()
    if method == "bfs":
        return BFS()
    if method == "dfs":
        return DFS()
    if method == "astar":  # Add this
        return AStar()
    raise SystemExit(f"Unknown method: {method!r}. Use one of: bfs, dfs, astar")

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MazeLLM: solve maze and visualize step-by-step.")
    p.add_argument("--c", "--cols", dest="cols", type=int, default=15, help="Maze width (columns)")
    p.add_argument("--r", "--rows", dest="rows", type=int, default=15, help="Maze height (rows)")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    p.add_argument("--interval", type=float, default=0.15, help="Seconds between steps")
    p.add_argument("-m", "--method", type=str, default="bfs", help="Solve method: bfs | dfs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.cols <= 0 or args.rows <= 0:
        raise SystemExit("rows and columns must be positive integers.")

    maze = Maze(cols=args.cols, rows=args.rows, seed=args.seed)
    maze.generate_maze()

    start = _find_cell(maze, "S")
    end = _find_cell(maze, "E")

    solver = _build_solver(args.method)
    path = solver.solve(maze=maze, start=start, end=end)

    robot = Robot(maze=maze, position=Position(x=start.x, y=start.y))

    SolveMazePanel(
        maze=maze,
        robot=robot,
        path=path,
        interval_s=args.interval,
        method=args.method.lower().strip(),
    ).run()


if __name__ == "__main__":
    main()
