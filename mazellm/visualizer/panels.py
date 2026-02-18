# mazellm/visualizer/panels.py
from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Header, Footer

from mazellm.maze import Maze
from mazellm.robot import Robot
from mazellm.solver import Solver


class SolveMazePanel(App):
    """
    One generic panel for ALL solvers (bfs/dfs/astar/llm).

    It runs an async loop:
      - await solver.next(...)
      - paint visited
      - paint robot position
      - stop on done
    """

    CSS_PATH = Path(__file__).with_name("maze_visualizer.tcss")
    BINDINGS = [("q", "quit", "Quit"), ("space", "toggle_pause", "Pause/Resume")]

    def __init__(self, *, maze: Maze, robot: Robot, solver: Solver, interval_s: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.maze = maze
        self.robot = robot
        self.solver = solver
        self.interval_s = float(interval_s)

        self.rows = int(getattr(maze, "m", 1))  # y
        self.cols = int(getattr(maze, "n", 1))  # x

        self.tiles: list[Static] = []
        self.robot_pos: tuple[int, int] | None = None  # (row, col)
        self.info_panel: Static | None = None
        self.logs: list[str] = [f"{self.rows}x{self.cols} maze", f"Solver: {self.solver.name}"]
        self.visited: set[tuple[int, int]] = set()
        self.paused: bool = False
        self._stop: bool = False
        self._task = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Container(id="left-panel"):
                self.info_panel = Static("\n".join(self.logs), id="left-panel-content")
                yield self.info_panel

            with Container(id="right-panel"):
                with Container(id="chess-board"):
                    for r in range(self.rows):
                        for c in range(self.cols):
                            tile = Static(classes="tile")
                            self.tiles.append(tile)
                            yield tile
        yield Footer()

    def on_mount(self) -> None:
        board = self.query_one("#chess-board", Container)
        board.styles.grid_size_rows = self.rows
        board.styles.grid_size_columns = self.cols

        self.render_maze()
        self.set_robot_position(self.robot.position.y, self.robot.position.x)
        self.mark_visited([(self.robot.position.y, self.robot.position.x)])

        self.log_info(f"Mode: {self.solver.name}")
        self.log_info("▶ starting async runner loop")

        self._task = asyncio.create_task(self._runner_loop())

    async def _runner_loop(self) -> None:
        while not self._stop:
            if self.paused:
                await asyncio.sleep(0.05)
                continue

            try:
                # Stop if already at end
                if self.maze.board[self.robot.position.y, self.robot.position.x] == "E":
                    self.log_info("✅ Reached end.")
                    self._stop = True
                    break

                result = await self.solver.next(maze=self.maze, robot=self.robot, logger=self.log_info)

                if result.message:
                    self.log_info(result.message)

                if result.visited_added_rc:
                    self.mark_visited(result.visited_added_rc)

                self.set_robot_position(self.robot.position.y, self.robot.position.x)

                if result.done:
                    self.log_info("✅ Reached end.")
                    self._stop = True
                    break

            except Exception as e:
                self.log_info(f"💥 Runner crashed: {type(e).__name__}: {e}")
                self.log_info(traceback.format_exc())
                self._stop = True
                break

            await asyncio.sleep(self.interval_s)

    def action_quit(self) -> None:
        self._stop = True
        self.exit()

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.log_info("⏸ Paused" if self.paused else "▶ Resumed")

    def log_info(self, message: str) -> None:
        self.logs.append(message)
        if self.info_panel:
            self.info_panel.update("\n".join(self.logs[-20:]))

    def _idx(self, row: int, col: int) -> int:
        return row * self.cols + col

    def render_maze(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                idx = self._idx(r, c)
                cell = self.maze.board[r, c]

                for cls in ("start", "end", "wall", "free", "visited", "visited-old", "robot"):
                    self.tiles[idx].set_class(False, cls)

                if cell == "S":
                    self.tiles[idx].set_class(True, "start")
                elif cell == "E":
                    self.tiles[idx].set_class(True, "end")
                elif cell == 1:
                    self.tiles[idx].set_class(True, "wall")
                else:
                    self.tiles[idx].set_class(True, "free")

                if (r, c) in self.visited and cell != 1:
                    self.tiles[idx].set_class(True, "visited")

        if self.robot_pos is not None:
            rr, cc = self.robot_pos
            if 0 <= rr < self.rows and 0 <= cc < self.cols:
                self.tiles[self._idx(rr, cc)].set_class(True, "robot")

    def set_robot_position(self, row: int, col: int) -> None:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self.log_info(f"Invalid robot position: ({row}, {col})")
            return

        if self.robot_pos is not None:
            old_r, old_c = self.robot_pos
            self.tiles[self._idx(old_r, old_c)].set_class(False, "robot")

        self.robot_pos = (row, col)
        self.tiles[self._idx(row, col)].set_class(True, "robot")

    def mark_visited(self, cells: list[tuple[int, int]]) -> None:
        # downgrade previous visited to visited-old
        for (r, c) in self.visited:
            idx = self._idx(r, c)
            self.tiles[idx].set_class(False, "visited")
            self.tiles[idx].set_class(True, "visited-old")

        # add new visited
        for (r, c) in cells:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.visited.add((r, c))
                idx = self._idx(r, c)
                self.tiles[idx].set_class(True, "visited")

        self.render_maze()
