from __future__ import annotations
import argparse
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Header, Footer


class MazePanel(App):
    """An nxn maze-like board with a two-panel layout."""
    CSS_PATH = "maze_visualizer.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),  # Press q to quit
    ]

    def __init__(self, n: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n = max(1, n)
        self.tiles: list[Static] = []
        self.robot_pos: tuple[int, int] | None = None
        self.info_panel: Static | None = None
        self.logs: list[str] = [f"{self.n}x{self.n} maze"]  # initial message

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Container(id="left-panel"):
                self.info_panel = Static("\n".join(self.logs), id="left-panel-content")
                yield self.info_panel
            with Container(id="right-panel"):
                with Container(id="chess-board"):
                    for r in range(self.n):
                        for c in range(self.n):
                            tile_class = "dark" if (r + c) % 2 else "light"
                            tile = Static(classes=f"tile {tile_class}")
                            self.tiles.append(tile)
                            yield tile
        yield Footer()

    def on_mount(self) -> None:
        board = self.query_one("#chess-board", Container)
        board.styles.grid_size_rows = self.n
        board.styles.grid_size_columns = self.n

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def log_info(self, message: str) -> None:
        """Append a message to the info panel."""
        self.logs.append(message)
        if self.info_panel:
            self.info_panel.update("\n".join(self.logs[-20:]))  # keep last 20 lines

    def set_robot_position(self, row: int, col: int) -> None:
        """Highlight the robot's cell in red and update info panel."""
        if not (0 <= row < self.n and 0 <= col < self.n):
            self.log(f"Invalid robot position: ({row}, {col})")
            return

        # Reset old robot cell to original color
        if self.robot_pos is not None:
            old_r, old_c = self.robot_pos
            old_idx = old_r * self.n + old_c
            base_class = "dark" if (old_r + old_c) % 2 else "light"
            self.tiles[old_idx].set_class(False, "robot")
            self.tiles[old_idx].set_class(True, base_class)

        # Set new robot cell to red
        idx = row * self.n + col
        self.tiles[idx].set_class(True, "robot")
        self.robot_pos = (row, col)

        # Log new location
        self.log_info(f"Current location ({row},{col})")


def _parse_args():
    p = argparse.ArgumentParser(description="nxn maze TUI")
    p.add_argument("--n", type=int, default=5, help="Board size (nxn). Default: 5")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    app = MazePanel(n=args.n)

    path = [(1, 1), (1, 2), (2, 2), (3, 2)]  # list of positions
    delay = 0.1  # seconds (100 ms)

    def run_path(step: int = 0):
        if step < len(path):
            if step > 0:
                app.log_info(f"Moving to {path[step]} ...")

            row, col = path[step]
            app.set_robot_position(row, col)

            if step < len(path) - 1:
                app.set_timer(delay / 2, lambda: app.log_info("Finding next step ..."))

            app.set_timer(delay, lambda: run_path(step + 1))

    app.call_after_refresh(lambda: run_path(0))
    app.run()
