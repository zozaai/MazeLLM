# mazellm/cli.py
from __future__ import annotations
import time
from typing import List, Tuple

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.text import Text

app = typer.Typer(help="CLI chess/maze visualizer (Typer + Rich)")
console = Console()

# Hard-coded tile size (twice the previous 4x2)
TILE_W = 8  # characters wide
TILE_H = 4  # text lines tall


def parse_pos(s: str) -> Tuple[int, int]:
    r, c = s.split(",")
    return int(r), int(c)


def render_board(n: int, robot: Tuple[int, int] | None) -> Panel:
    """Return a Rich Panel containing an n×n board with optional robot cell (red)."""
    t = Table.grid(padding=0)
    for _ in range(n):
        t.add_column(no_wrap=True)

    fill = " " * TILE_W
    block = "\n".join([fill] * TILE_H)  # multi-line cell for visual height

    for r in range(n):
        row_cells = []
        for c in range(n):
            base_dark = (r + c) % 2 == 1
            style = "on #B58863" if base_dark else "on #F0D9B5"
            if robot and (r, c) == robot:
                style = "on red"
            row_cells.append(Text(block, style=style))
        t.add_row(*row_cells)

    return Panel(t, title=f"{n}×{n} board", border_style="cyan")


def render_info(logs: List[str]) -> Panel:
    return Panel("\n".join(logs[-20:]), title="Info", border_style="magenta")


def render_layout(n: int, robot: Tuple[int, int] | None, logs: List[str]):
    left = render_info(logs)
    right = render_board(n, robot)
    return Columns([left, right], equal=False, expand=True)


@app.command()
def board(
    n: int = typer.Option(8, help="Board size n (nxn)"),
    robot: str = typer.Option(None, help='Robot position as "row,col" (0-indexed)'),
):
    """Render a single board frame."""
    robot_pos = parse_pos(robot) if robot else None
    console.print(render_layout(n, robot_pos, [f"{n}x{n} maze"]))


@app.command()
def animate(
    n: int = typer.Option(8, help="Board size n (nxn)"),
    path: List[str] = typer.Argument(..., help='Sequence like: "1,1" "1,2" "2,2"'),
    interval: float = typer.Option(0.1, help="Seconds between steps (0.1 = 100ms)"),
):
    """Animate robot movement along PATH."""
    pts = [parse_pos(p) for p in path]
    logs: List[str] = [f"{n}x{n} maze"]

    with Live(render_layout(n, None, logs), console=console, refresh_per_second=30) as live:
        # initial
        r, c = pts[0]
        logs.append(f"Current location ({r},{c})")
        live.update(render_layout(n, (r, c), logs))

        for nr, nc in pts[1:]:
            logs.append("Finding next step ...")
            live.update(render_layout(n, (r, c), logs))
            time.sleep(interval)

            logs.append(f"Moving to ({nr},{nc}) ...")
            live.update(render_layout(n, (r, c), logs))
            r, c = nr, nc
            logs.append(f"Current location ({r},{c})")
            live.update(render_layout(n, (r, c), logs))


if __name__ == "__main__":
    app()
