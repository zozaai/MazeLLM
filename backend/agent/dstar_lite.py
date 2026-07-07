"""Fog-of-war solver: **D\\* Lite** (Koenig & Likhachev, 2002).

A *fair* comparison against the LLM — it navigates under the SAME limitation:
it only knows cells that `sense_surroundings` has revealed (`robot.known_cells`),
treats unsensed cells as optimistically open, and heads for the goal along the
shortest route over that known map.

Unlike a plan-from-scratch replanner, D\\* Lite searches *backward from the goal*
and, when sensing turns an assumed-open cell into a wall, **incrementally repairs**
only the affected part of the search instead of recomputing everything. The robot
trajectory it produces is identical to optimistic replanning A\\*; the difference
is that the repair is cheap, so it scales to large grids.

It runs in the agent's own representation and reuses the same tools
(`sense_surroundings` / `move`) and memory rendering as `MazeSolvingAgent`, so it
emits byte-identical `memory` / `sense` / `move` events (same fog reveal), and its
transcripts are a drop-in expert for finetuning.
"""
from __future__ import annotations

import heapq

from ..maze.maze import Cell, Maze
from ..maze.robot import DIRECTIONS, Robot
from .llm_agent import build_memory_message
from .tools import move as move_robot, sense_surroundings

# Fog-of-war solver names dispatched by server.py.
FOG_SOLVERS = {"dstar_lite"}

INF = float("inf")
_DELTA_TO_DIRECTION = {delta: name for name, delta in DIRECTIONS.items()}


class DStarLite:
    """D\\* Lite over a 4-connected grid. Cost 1 between open cells, ∞ into a
    known wall. `g[s]` is the (repaired) cost-to-goal estimate for each cell; the
    robot always steps to the successor minimising `cost + g`."""

    def __init__(self, width: int, height: int, start: Cell, goal: Cell):
        self.w, self.h = width, height
        self.start = start
        self.goal = goal
        self._last = start          # start position at the previous repair (for km)
        self._km = 0.0              # key modifier that absorbs the moving start
        self.g: dict[Cell, float] = {}
        self.rhs: dict[Cell, float] = {}
        self.walls: set[Cell] = set()
        self._pq: list[tuple[float, float, int, Cell]] = []
        self._pq_keys: dict[Cell, tuple[float, float]] = {}  # authoritative key of queued cells
        self._counter = 0
        self.rhs[goal] = 0.0
        self._insert(goal, self._calc_key(goal))
        self.compute_shortest_path()

    # --- value accessors (default to ∞) ---
    def _g(self, s: Cell) -> float:
        return self.g.get(s, INF)

    def _rhs(self, s: Cell) -> float:
        return self.rhs.get(s, INF)

    @staticmethod
    def _h(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _in_bounds(self, s: Cell) -> bool:
        return 0 <= s[0] < self.w and 0 <= s[1] < self.h

    def _neighbors(self, s: Cell):
        x, y = s
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            n = (x + dx, y + dy)
            if self._in_bounds(n):
                yield n

    def _cost(self, u: Cell, v: Cell) -> float:
        return INF if (u in self.walls or v in self.walls) else 1.0

    def _calc_key(self, s: Cell) -> tuple[float, float]:
        k2 = min(self._g(s), self._rhs(s))
        return (k2 + self._h(self.start, s) + self._km, k2)

    # --- priority queue with lazy deletion (entry is stale if its key != _pq_keys[cell]) ---
    def _insert(self, s: Cell, key: tuple[float, float]) -> None:
        self._pq_keys[s] = key
        self._counter += 1
        heapq.heappush(self._pq, (key[0], key[1], self._counter, s))

    def _remove(self, s: Cell) -> None:
        self._pq_keys.pop(s, None)

    def _top(self):
        while self._pq:
            k1, k2, _, s = self._pq[0]
            if self._pq_keys.get(s) == (k1, k2):
                return (k1, k2), s
            heapq.heappop(self._pq)
        return None, None

    def _update_vertex(self, u: Cell) -> None:
        if u != self.goal:
            best = INF
            for n in self._neighbors(u):
                c = self._cost(u, n) + self._g(n)
                if c < best:
                    best = c
            self.rhs[u] = best
        if u in self._pq_keys:
            self._remove(u)
        if self._g(u) != self._rhs(u):
            self._insert(u, self._calc_key(u))

    def compute_shortest_path(self) -> None:
        while True:
            key, u = self._top()
            if key is None:
                break
            if not (key < self._calc_key(self.start) or self._rhs(self.start) != self._g(self.start)):
                break
            k_new = self._calc_key(u)
            if key < k_new:                       # key was out of date — reinsert with fresh key
                self._insert(u, k_new)
            elif self._g(u) > self._rhs(u):        # overconsistent — relax
                self.g[u] = self._rhs(u)
                self._remove(u)
                for n in self._neighbors(u):
                    self._update_vertex(n)
            else:                                  # underconsistent — raise and repropagate
                self.g[u] = INF
                self._update_vertex(u)
                for n in self._neighbors(u):
                    self._update_vertex(n)

    def observe_walls(self, new_walls: list[Cell]) -> None:
        """Incorporate newly discovered walls and repair the search."""
        if not new_walls:
            return
        self._km += self._h(self._last, self.start)
        self._last = self.start
        for w in new_walls:
            self.walls.add(w)
            self._update_vertex(w)
            for n in self._neighbors(w):
                self._update_vertex(n)
        self.compute_shortest_path()

    def greedy_path(self) -> list[Cell]:
        """Reconstruct the current best route from the start by descending g."""
        path = [self.start]
        cur = self.start
        seen = {self.start}
        while cur != self.goal:
            best, best_n = INF, None
            for n in self._neighbors(cur):
                if self._cost(cur, n) == INF:
                    continue
                if self._g(n) < best:
                    best, best_n = self._g(n), n
            if best_n is None or best >= self._g(cur) or best_n in seen:
                break
            path.append(best_n)
            seen.add(best_n)
            cur = best_n
        return path


class DStarLiteAgent:
    """Drives a robot with D\\* Lite under fog-of-war. Mirrors
    ``MazeSolvingAgent``'s ``run_step`` / ``is_done`` contract."""

    def __init__(self, maze: Maze, robot: Robot, solver_name: str = "dstar_lite", max_steps: int = 200):
        if solver_name not in FOG_SOLVERS:
            raise ValueError(f"unknown fog solver: {solver_name!r}")
        self.maze = maze
        self.robot = robot
        self.solver_name = solver_name
        self.max_steps = max_steps
        self._planner = DStarLite(maze.width, maze.height, tuple(robot.position), maze.end)

    async def run_step(self) -> list[dict]:
        """One turn: sense, repair the search with any newly found walls, then
        take the longest straight move along the D\\* Lite route that stays on
        cells sensing just confirmed open."""
        self._planner.start = tuple(self.robot.position)
        events: list[dict] = [{"type": "memory", "content": build_memory_message(self.maze, self.robot)}]

        sensed = sense_surroundings(self.maze, self.robot)
        events.append(
            {
                "type": "sense",
                "position": list(self.robot.position),
                "sensed": sensed,
                "reasoning": "Sensing to map open corridors from here.",
            }
        )

        new_walls = [c for c, st in self.robot.known_cells.items() if st == "wall" and c not in self._planner.walls]
        self._planner.observe_walls(new_walls)

        path = self._planner.greedy_path()
        if len(path) < 2:
            events.append({"type": "error", "message": "no route to goal under the known map"})
            return events

        cur = self.robot.position
        dx, dy = path[1][0] - cur[0], path[1][1] - cur[1]
        direction = _DELTA_TO_DIRECTION[(dx, dy)]

        # follow the route straight in this direction, capped to the run sensing proved open
        straight = 1
        while straight + 1 < len(path) and (
            path[straight + 1][0] - path[straight][0],
            path[straight + 1][1] - path[straight][1],
        ) == (dx, dy):
            straight += 1
        distance = max(1, min(straight, sensed[direction]["distance"]))

        position_before = self.robot.position
        success, _msg, new_position, moved = move_robot(self.maze, self.robot, direction, distance)
        if moved > 0:
            self.robot.record_step(sensed, direction, new_position)
        events.append(
            {
                "type": "move",
                "direction": direction,
                "distance_requested": distance,
                "distance_moved": moved,
                "success": success,
                "position_before": list(position_before),
                "position_after": list(new_position),
                "reasoning": f"D* Lite route heads {direction}; moving {moved} toward the end.",
            }
        )
        return events

    def is_done(self) -> bool:
        return self.robot.position == self.maze.end or len(self.robot.history) >= self.max_steps
