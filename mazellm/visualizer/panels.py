# mazellm/visualizer/panels.py
from __future__ import annotations

import argparse
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Header, Footer
from pathlib import Path

from mazellm.maze import Maze


class MazePanel(App):
    """
    Visualize a Maze object (maze.board) with a two-panel layout.
      - S (start): green
      - E (end): blue
      - wall (1): black
      - free (0): white
      - robot: red (optional, if set_robot_position is used)
    """

    CSS_PATH = Path(__file__).with_name("maze_visualizer.tcss")
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, maze: Maze, **kwargs):
        super().__init__(**kwargs)
        self.maze = maze
        self.rows = int(getattr(maze, "m", 1))  # y / rows
        self.cols = int(getattr(maze, "n", 1))  # x / cols

        self.tiles: list[Static] = []
        self.robot_pos: tuple[int, int] | None = None  # (row, col)
        self.info_panel: Static | None = None
        self.logs: list[str] = [f"{self.rows}x{self.cols} maze"]

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

    def action_quit(self) -> None:
        self.exit()

    def log_info(self, message: str) -> None:
        self.logs.append(message)
        if self.info_panel:
            self.info_panel.update("\n".join(self.logs[-20:]))

    def _idx(self, row: int, col: int) -> int:
        return row * self.cols + col

    def render_maze(self) -> None:
        """
        Paint all tiles based on maze.board values:
          - "S" -> start
          - "E" -> end
          - 1   -> wall
          - else -> free
        """
        for r in range(self.rows):
            for c in range(self.cols):
                idx = self._idx(r, c)
                cell = self.maze.board[r, c]

                # Clear all known state classes (except base "tile")
                for cls in ("start", "end", "wall", "free", "robot"):
                    self.tiles[idx].set_class(False, cls)

                if cell == "S":
                    self.tiles[idx].set_class(True, "start")
                elif cell == "E":
                    self.tiles[idx].set_class(True, "end")
                elif cell == 1:
                    self.tiles[idx].set_class(True, "wall")
                else:
                    self.tiles[idx].set_class(True, "free")

        # Re-apply robot highlight if set
        if self.robot_pos is not None:
            rr, cc = self.robot_pos
            if 0 <= rr < self.rows and 0 <= cc < self.cols:
                self.tiles[self._idx(rr, cc)].set_class(True, "robot")

    def set_robot_position(self, row: int, col: int) -> None:
        """Highlight robot cell in red (overlays whatever is underneath)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self.log_info(f"Invalid robot position: ({row}, {col})")
            return

        # Remove old robot highlight
        if self.robot_pos is not None:
            old_r, old_c = self.robot_pos
            self.tiles[self._idx(old_r, old_c)].set_class(False, "robot")

        # Apply new
        self.robot_pos = (row, col)
        self.tiles[self._idx(row, col)].set_class(True, "robot")
        self.log_info(f"Current location ({row},{col})")


def _parse_args():
    p = argparse.ArgumentParser(description="MazeLLM board viewer")
    p.add_argument("--n", type=int, default=15, help="Maze width (columns)")
    p.add_argument("--m", type=int, default=15, help="Maze height (rows)")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    maze = Maze(n=args.n, m=args.m, seed=args.seed)
    maze.generate_maze()
    print(maze.board)
    MazePanel(maze=maze).run()
