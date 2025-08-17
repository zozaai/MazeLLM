# main.py
from __future__ import annotations

import argparse
from typing import Optional, Tuple

from mazellm.maze import Maze
from mazellm.robot import Robot
from mazellm.agent import LLMAgent
from mazellm.visualizer.panels import MazePanel  # your UI


class MazeSolverApp(MazePanel):
    """
    App that shows a maze and moves a robot step-by-step using an LLM agent.
    """
    def __init__(self, maze_n: int = 15, maze_m: int = 15, **kwargs):
        # MazePanel renders an n×n grid; pass the larger side for now
        super().__init__(n=max(maze_n, maze_m), **kwargs)
        self.maze_n = maze_n
        self.maze_m = maze_m
        self.maze = Maze(n=self.maze_n, m=self.maze_m)
        self.maze.generate_maze()
        self.robot = Robot()  # Robot with Position(x, y)
        self.agent = LLMAgent()
        self.logs = [f"{self.maze_n}x{self.maze_m} maze"]

    # ---------- helpers ----------
    def _to_row_col(self, pos) -> tuple[int, int]:
        """Convert Robot.position (Position) to (row, col)."""
        if hasattr(pos, "y") and hasattr(pos, "x"):
            return int(pos.y), int(pos.x)
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            return int(pos[0]), int(pos[1])
        return (0, 0)

    def _write_robot_position(self, r: int, c: int) -> None:
        """Write (row, col) into Robot.position (Position)."""
        if hasattr(self.robot.position, "y") and hasattr(self.robot.position, "x"):
            self.robot.position.y = r
            self.robot.position.x = c
        else:
            self.robot.position = (r, c)

    # ---------- lifecycle ----------
    async def on_mount(self) -> None:
        """
        Place the robot and start a non-blocking control loop that calls step().
        """
        super().on_mount()

        r, c = self._to_row_col(self.robot.position)
        self.set_robot_position(r, c)
        self.log_info(f"Current location ({r},{c})")

        # Call _tick every 100 ms (sync), which schedules async step()
        self.set_interval(0.5, self._tick)

    async def step(self) -> None:
        """
        Ask the LLM agent for the next move and visualize it.
        """
        raw_next = await self.agent.next_step(self.maze, self.robot)
        if raw_next is None:
            self.log_info("No next step (done).")
            return

        nr, nc = self._to_row_col(raw_next)
        self.log_info(f"Moving to ({nr},{nc}) ...")
        self._write_robot_position(nr, nc)
        self.set_robot_position(nr, nc)
        self.log_info(f"Current location ({nr},{nc})")

    def _tick(self) -> None:
        """
        Sync timer callback fired every 100 ms. Shows a waiting line, then
        schedules the async `step()` to run.
        """
        self.log_info("Finding next step ...")
        self.call_later(self.step)


def _parse_args():
    p = argparse.ArgumentParser(description="LLM-driven Maze Solver TUI")
    p.add_argument("--n", type=int, default=15, help="Maze rows (n)")
    p.add_argument("--m", type=int, default=15, help="Maze cols (m)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    app = MazeSolverApp(maze_n=args.n, maze_m=args.m)
    app.run()
