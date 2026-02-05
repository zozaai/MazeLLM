# mazellm/main.py
from __future__ import annotations

import argparse
import asyncio
import traceback
from collections.abc import Iterable

from mazellm.maze import Maze
from mazellm.robot import Robot, Position, Direction
from mazellm.visualizer.panels import MazePanel
from mazellm.bfs import BFS
from mazellm.dfs import DFS
from mazellm.astar import AStar
from mazellm.agent import LLMAgent

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


class LLMSolveMazePanel(MazePanel):
    def __init__(
        self,
        maze: Maze,
        robot: Robot,
        agent: LLMAgent,
        interval_s: float = 0.2,
        **kwargs,
    ):
        super().__init__(maze=maze, **kwargs)
        self.robot = robot
        self.agent = agent
        self.interval_s = float(interval_s)
        self._stop = False
        self._task = None

    def on_mount(self) -> None:
        super().on_mount()

        self.set_robot_position(self.robot.position.y, self.robot.position.x)
        self.mark_visited([(self.robot.position.y, self.robot.position.x)])

        self.log_info("Mode: llm")
        self.log_info("Tools: sense(), move(direction,steps), get_state()")
        self.log_info("▶ starting async runner loop")

        # Start ONE loop task (no set_interval, no overlapping tasks)
        import asyncio
        self._task = asyncio.create_task(self._runner_loop())

    async def _runner_loop(self) -> None:
        import asyncio
        import traceback

        while not self._stop:
            try:
                self.log_info("⏱️ tick (runner loop)")

                # If already at end, stop
                if self.maze.board[self.robot.position.y, self.robot.position.x] == "E":
                    self.log_info("✅ Reached end.")
                    self._stop = True
                    break

                self.log_info("➡️ calling agent.run_until_move_or_done()")

                result = await self.agent.run_until_move_or_done(
                    maze=self.maze,
                    robot=self.robot,
                    logger=self.log_info,
                )

                self.log_info(
                    f"⬅️ agent returned: did_move={result.get('did_move')} done={result.get('done')}"
                )

                # paint visited + position
                if result.get("visited_added_rc"):
                    self.mark_visited(result["visited_added_rc"])

                self.set_robot_position(self.robot.position.y, self.robot.position.x)

                if result.get("done"):
                    self.log_info("✅ Reached end.")
                    self._stop = True
                    break

            except Exception as e:
                self.log_info(f"💥 Runner crashed: {type(e).__name__}: {e}")
                self.log_info(traceback.format_exc())
                # stop so it doesn't spin forever
                self._stop = True
                break

            # pace loop
            await asyncio.sleep(self.interval_s)

    def action_quit(self) -> None:
        self._stop = True
        super().action_quit()



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
    p.add_argument("-m", "--method", type=str, default="bfs", help="Solve method: bfs | dfs | astar | llm")
    p.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model name (for -m llm)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.cols <= 0 or args.rows <= 0:
        raise SystemExit("rows and columns must be positive integers.")

    maze = Maze(cols=args.cols, rows=args.rows, seed=args.seed)
    maze.generate_maze()

    start = _find_cell(maze, "S")
    end = _find_cell(maze, "E")

    robot = Robot(maze=maze, position=Position(x=start.x, y=start.y))

    method = args.method.lower().strip()
    if method == "llm":
        agent = LLMAgent(model=args.llm_model)
        LLMSolveMazePanel(
            maze=maze,
            robot=robot,
            agent=agent,
            interval_s=args.interval,
        ).run()
        return

    solver = _build_solver(method)
    path = solver.solve(maze=maze, start=start, end=end)
    SolveMazePanel(
        maze=maze,
        robot=robot,
        path=path,
        interval_s=args.interval,
        method=method,
    ).run()

if __name__ == "__main__":
    main()
