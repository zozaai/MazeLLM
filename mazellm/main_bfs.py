# mazellm/main_demo.py
from __future__ import annotations

import argparse
from collections import deque
from typing import Iterable

from mazellm.maze import Maze
from mazellm.robot import Robot, Position, Direction
from mazellm.visualizer.panels import MazePanel


def _find_cell(maze: Maze, value: object) -> Position:
    for y in range(maze.m):
        for x in range(maze.n):
            if maze.board[y, x] == value:
                return Position(x=x, y=y)
    raise ValueError(f"Could not find {value!r} in maze.board")


def _neighbors(x: int, y: int) -> Iterable[tuple[int, int]]:
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _bfs_path(maze: Maze, start: Position, end: Position) -> list[Position]:
    """Shortest path (4-neighborhood) from start to end, inclusive."""
    q: deque[tuple[int, int]] = deque()
    q.append((start.x, start.y))

    parent: dict[tuple[int, int], tuple[int, int] | None] = {(start.x, start.y): None}

    while q:
        x, y = q.popleft()
        if (x, y) == (end.x, end.y):
            break

        for nx, ny in _neighbors(x, y):
            if not (0 <= nx < maze.n and 0 <= ny < maze.m):
                continue
            if maze.is_barrier(x=nx, y=ny):
                continue
            if (nx, ny) in parent:
                continue

            parent[(nx, ny)] = (x, y)
            q.append((nx, ny))

    if (end.x, end.y) not in parent:
        raise RuntimeError("No path found from S to E (maze may be disconnected).")

    # Reconstruct
    path_xy: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = (end.x, end.y)
    while cur is not None:
        path_xy.append(cur)
        cur = parent[cur]
    path_xy.reverse()

    return [Position(x=x, y=y) for x, y in path_xy]


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


class DemoMazePanel(MazePanel):
    def __init__(self, maze: Maze, robot: Robot, path: list[Position], interval_s: float = 0.2, **kwargs):
        super().__init__(maze=maze, **kwargs)
        self.robot = robot
        self.path = path
        self.interval_s = float(interval_s)
        self._step_i = 0  # index into path (current position is path[_step_i])

    def on_mount(self) -> None:
        super().on_mount()

        # Initial robot placement at start
        self.set_robot_position(self.robot.position.y, self.robot.position.x)
        self.log_info(f"Start: (x={self.robot.position.x}, y={self.robot.position.y})")
        self.log_info(f"Goal : (x={self.path[-1].x}, y={self.path[-1].y})")
        self.log_info(f"Path length: {len(self.path)}")

        # Drive the animation
        self.set_interval(self.interval_s, self._tick)

    def _tick(self) -> None:
        # Already at end
        if self._step_i >= len(self.path) - 1:
            self.log_info("✅ Reached end.")
            return

        cur = self.path[self._step_i]
        nxt = self.path[self._step_i + 1]
        move_cmd = _step_to_move(cur, nxt)

        res = self.robot.move(move_cmd)
        if not res["status"]:
            # Should never happen if BFS path is valid, but fail gracefully.
            self.log_info(f"❌ Move failed: {move_cmd} from (x={cur.x}, y={cur.y})")
            return

        self._step_i += 1
        new_pos = self.robot.position

        # Update visuals + left panel logs
        self.set_robot_position(new_pos.y, new_pos.x)
        dir_name, cells = next(iter(move_cmd.items()))
        self.log_info(f"Move: {dir_name} {cells} -> (x={new_pos.x}, y={new_pos.y})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MazeLLM demo: solve maze and visualize robot step-by-step.")
    p.add_argument("--n", type=int, default=15, help="Maze width (columns)")
    p.add_argument("--m", type=int, default=15, help="Maze height (rows)")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    p.add_argument("--interval", type=float, default=0.15, help="Seconds between steps")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n <= 0 or args.m <= 0:
        raise SystemExit("n and m must be positive integers.")

    maze = Maze(n=args.n, m=args.m, seed=args.seed)
    maze.generate_maze()

    start = _find_cell(maze, "S")
    end = _find_cell(maze, "E")

    robot = Robot(maze=maze, position=Position(x=start.x, y=start.y))
    path = _bfs_path(maze, start=start, end=end)

    DemoMazePanel(maze=maze, robot=robot, path=path, interval_s=args.interval).run()


if __name__ == "__main__":
    main()
