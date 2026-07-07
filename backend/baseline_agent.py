"""Web adapter for the classical baseline solvers (currently A* only).

This is the single integration point that bridges the two otherwise
self-contained stacks:

  * ``backend/maze``      — what the web app and LLM agent use (Cell-based Maze).
  * ``backend/baselines`` — the classical BFS/DFS/A* solvers, with their own
                            NumPy-board Maze/Robot/Position representation.

Neither stack imports the other; this module imports both and translates
between them. It deliberately mirrors ``MazeSolvingAgent``'s ``run_step()`` /
``is_done()`` contract so ``server.py`` can drive either solver with the exact
same loop and emit the exact same WebSocket step events the frontend animates.
"""
from __future__ import annotations

from backend.baselines import AStarSolver, BFSSolver, DFSSolver
from backend.baselines.maze import Maze as BaselineMaze
from backend.baselines.robot import Robot as BaselineRobot
from backend.baselines.types import Position

from .maze.maze import Maze
from .maze.robot import Robot

# Solver name -> baseline Solver class. Adding a classical solver here is all it
# takes to expose it over the WebSocket — server.py and the event contract are
# registry-driven and need no changes (the frontend just needs a matching button).
SOLVERS = {"astar": AStarSolver, "bfs": BFSSolver, "dfs": DFSSolver}

_DELTA_TO_DIRECTION = {
    (0, -1): "up",
    (0, 1): "down",
    (-1, 0): "left",
    (1, 0): "right",
}


def _to_baseline_maze(maze: Maze) -> BaselineMaze:
    """Rebuild the browser's maze as a baseline Maze so the solver runs on the
    identical layout the canvas is showing (1 = wall, 0 = open, "S"/"E" cells)."""
    bmaze = BaselineMaze(cols=maze.width, rows=maze.height)
    for y in range(maze.height):
        for x in range(maze.width):
            bmaze.board[y, x] = 1 if (x, y) in maze.walls else 0
    sx, sy = maze.start
    ex, ey = maze.end
    bmaze.board[sy, sx] = "S"
    bmaze.board[ey, ex] = "E"
    return bmaze


class BaselineSolvingAgent:
    """Runs a classical solver against the web maze, one move per ``run_step``.

    Presents the same interface as ``MazeSolvingAgent`` (``run_step`` returning a
    list of step-event dicts, plus ``is_done``). Unlike the LLM agent it needs
    no provider/API key — the whole solve is local and deterministic.
    """

    def __init__(self, maze: Maze, robot: Robot, solver_name: str = "astar", max_steps: int = 200):
        if solver_name not in SOLVERS:
            raise ValueError(f"unknown baseline solver: {solver_name!r}")
        self.maze = maze
        self.robot = robot
        self.max_steps = max_steps
        self.solver_name = solver_name
        self._bmaze = _to_baseline_maze(maze)
        self._brobot = BaselineRobot(
            maze=self._bmaze, position=Position(x=robot.position[0], y=robot.position[1])
        )
        self._solver = SOLVERS[solver_name]()
        self._sent_plan = False

    def _plan_memory(self) -> str:
        sx, sy = self.maze.start
        ex, ey = self.maze.end
        return (
            f"{self.solver_name.upper()} solver — classical, full-knowledge pathfinding.\n"
            f"Start: ({sx},{sy})   End: ({ex},{ey})\n"
            f"Maze: {self.maze.width}x{self.maze.height}\n"
            "Walks the shortest path it computes over the known layout — no "
            "sensing/fog-of-war, unlike the LLM agent."
        )

    async def run_step(self) -> list[dict]:
        """Advance the solver by one move and return the resulting step events.

        The first call also emits a `memory` event describing the plan, so the
        side panel that shows the LLM's working stays populated for baselines too.
        """
        events: list[dict] = []
        if not self._sent_plan:
            events.append({"type": "memory", "content": self._plan_memory()})
            self._sent_plan = True

        result = await self._solver.next(maze=self._bmaze, robot=self._brobot)

        if result.did_move and result.new_position is not None:
            before = self.robot.position
            after = (result.new_position.x, result.new_position.y)
            direction = _DELTA_TO_DIRECTION.get((after[0] - before[0], after[1] - before[1]), "")
            # Keep the web robot in sync so server.py's done/step-count logic
            # (shared with the LLM path) works unchanged.
            self.robot.record_step({}, direction, after)
            events.append(
                {
                    "type": "move",
                    "direction": direction,
                    "distance_requested": 1,
                    "distance_moved": 1,
                    "success": True,
                    "position_before": list(before),
                    "position_after": list(after),
                    "reasoning": f"{self.solver_name.upper()}: {result.message}",
                }
            )
        return events

    def is_done(self) -> bool:
        return self.robot.position == self.maze.end or len(self.robot.history) >= self.max_steps
