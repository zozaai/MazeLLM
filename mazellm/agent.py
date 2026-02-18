# mazellm/agent.py
from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from mazellm.maze import Maze
from mazellm.solver import Solver, StepResult
from mazellm.robot import Robot
from mazellm.types import Position, Direction


@dataclass
class MoveRecord:
    i: int
    direction: Direction
    steps: int
    from_pos: Position
    to_pos: Position
    ok: bool


class LLMSolver(Solver):
    """
    LLM-driven solver that performs ONE successful move per tick.

    - Still uses tool-calling (sense/move/get_state).
    - Internally applies the move on the robot (consistent with Solver.next contract).
    - Updates self.visited and returns visited_added_rc.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_total_steps: int = 2000,
        max_tool_calls_per_turn: int = 12,
        api_key_env: str = "OPENAI_API_KEY",
        request_timeout_s: float = 20.0,
        also_print: bool = True,
    ):
        super().__init__(name="llm")
        self.model = model
        self.max_total_steps = int(max_total_steps)
        self.max_tool_calls_per_turn = int(max_tool_calls_per_turn)
        self.request_timeout_s = float(request_timeout_s)
        self.also_print = bool(also_print)

        if api_key_env not in os.environ:
            raise RuntimeError(f"{api_key_env} not set in environment.")

        self.client = AsyncOpenAI()

        self.step_i = 0
        self.moves: List[MoveRecord] = []
        self.last_sensor: Optional[Dict[Direction, int]] = None

    # -------------------------
    # Internal helpers
    # -------------------------
    def _at_goal(self, maze: Maze, robot: Robot) -> bool:
        x, y = robot.position.x, robot.position.y
        return (0 <= y < maze.m) and (0 <= x < maze.n) and (maze.board[y, x] == "E")

    def _local_memory_view(self, maze: Maze, robot: Robot, radius: int = 3) -> List[str]:
        rx, ry = robot.position.x, robot.position.y
        lines: List[str] = []
        for y in range(ry - radius, ry + radius + 1):
            row = []
            for x in range(rx - radius, rx + radius + 1):
                if not (0 <= x < maze.n and 0 <= y < maze.m):
                    row.append(" ")
                    continue
                if (x, y) == (rx, ry):
                    row.append("R")
                    continue
                v = maze.board[y, x]
                if v == 1:
                    row.append("#")
                elif v == "E":
                    row.append("E")
                else:
                    row.append("v" if (y, x) in self.visited else ".")
            lines.append("".join(row))
        return lines

    def _state_dict(self, maze: Maze, robot: Robot) -> Dict[str, Any]:
        pos = {"x": robot.position.x, "y": robot.position.y}
        done = self._at_goal(maze, robot)

        last_moves = [
            {
                "i": m.i,
                "dir": m.direction,
                "steps": m.steps,
                "from": {"x": m.from_pos.x, "y": m.from_pos.y},
                "to": {"x": m.to_pos.x, "y": m.to_pos.y},
                "ok": m.ok,
            }
            for m in self.moves[-10:]
        ]

        mem_view = self._local_memory_view(maze, robot, radius=3)

        return {
            "position": pos,
            "done": done,
            "visited_count": len(self.visited),
            "last_sensor": self.last_sensor,
            "last_moves": last_moves,
            "limits": {"step_i": self.step_i, "max_total_steps": self.max_total_steps},
            "memory_view_7x7": mem_view,
        }

    def _visited_cells_for_success_move(
        self, from_pos: Position, direction: Direction, steps: int, to_pos: Position
    ) -> List[tuple[int, int]]:
        cells: List[tuple[int, int]] = []
        x, y = from_pos.x, from_pos.y
        cells.append((x, y))

        dx, dy = 0, 0
        if direction == "up":
            dy = -1
        elif direction == "down":
            dy = 1
        elif direction == "left":
            dx = -1
        else:
            dx = 1

        for _ in range(steps):
            x += dx
            y += dy
            cells.append((x, y))

        if cells[-1] != (to_pos.x, to_pos.y):
            cells[-1] = (to_pos.x, to_pos.y)
        return cells

    def _tool_calls_to_dicts(self, tool_calls: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tc in (tool_calls or []):
            out.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", None),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", "{}") or "{}",
                    },
                }
            )
        return out

    # -------------------------
    # Tools
    # -------------------------
    def tool_sense(self, maze: Maze, robot: Robot) -> Dict[str, Any]:
        sensor = robot.sensor()
        self.last_sensor = sensor
        pos = {"x": robot.position.x, "y": robot.position.y}
        return {"position": pos, "sensor": sensor}

    def tool_move(self, maze: Maze, robot: Robot, direction: Direction, steps: int) -> Dict[str, Any]:
        if self.last_sensor is None:
            self.last_sensor = robot.sensor()

        max_steps = int(self.last_sensor.get(direction, 0))
        if steps < 0:
            return {
                "status": False,
                "error": f"Invalid steps: {steps} (must be >= 0)",
                "new_position": {"x": robot.position.x, "y": robot.position.y},
                "done": self._at_goal(maze, robot),
                "visited_added": [],
            }
        if steps > max_steps:
            return {
                "status": False,
                "error": f"Invalid move: steps={steps} > sensor[{direction}]={max_steps}",
                "new_position": {"x": robot.position.x, "y": robot.position.y},
                "done": self._at_goal(maze, robot),
                "visited_added": [],
            }

        from_pos = Position(x=robot.position.x, y=robot.position.y)
        res = robot.move({direction: int(steps)})
        to_pos = res["new_position"]

        ok = bool(res["status"])
        self.moves.append(
            MoveRecord(
                i=self.step_i,
                direction=direction,
                steps=int(steps),
                from_pos=from_pos,
                to_pos=to_pos,
                ok=ok,
            )
        )

        visited_added_xy: List[tuple[int, int]] = []
        if ok:
            visited_added_xy = self._visited_cells_for_success_move(from_pos, direction, int(steps), to_pos)
            # store as (row, col)
            for (x, y) in visited_added_xy:
                self.visited.add((y, x))

        return {
            "status": ok,
            "from_position": {"x": from_pos.x, "y": from_pos.y},
            "new_position": {"x": to_pos.x, "y": to_pos.y},
            "visited_added": [{"x": x, "y": y} for (x, y) in visited_added_xy],
            "done": self._at_goal(maze, robot),
        }

    def tool_get_state(self, maze: Maze, robot: Robot) -> Dict[str, Any]:
        return self._state_dict(maze, robot)

    # -------------------------
    # Solver API
    # -------------------------
    async def next(self, *, maze: Maze, robot: Robot, logger: Optional[Callable[[str], None]] = None) -> StepResult:
        log_lines: List[str] = []

        def log(s: str) -> None:
            log_lines.append(s)
            if logger:
                logger(s)
            if self.also_print:
                print(s, flush=True)

        # Ensure start cell registered
        self.visited.add((robot.position.y, robot.position.x))

        if self._at_goal(maze, robot):
            return StepResult(did_move=False, done=True, message="✅ Already at goal.", new_position=robot.position)

        if self.step_i >= self.max_total_steps:
            return StepResult(did_move=False, done=False, message="⚠️ Reached max_total_steps. Stopping.", new_position=robot.position)

        # Always sense first
        s0 = self.tool_sense(maze, robot)
        log(f"🧭 pre-sense -> {s0}")

        state_json = json.dumps(self._state_dict(maze, robot), ensure_ascii=False)

        system = (
            "You are controlling a robot in a maze.\n"
            "IMPORTANT: You MUST use tool calls. Do NOT respond with plain text.\n"
            "Goal: reach the End cell (E).\n"
            "Rules:\n"
            "- Use sense() to get allowed steps in each direction.\n"
            "- Choose ONE valid move(direction, steps) per tick.\n"
            "- steps must be <= sensor[direction].\n"
            "- Prefer unvisited paths when possible.\n"
            "- Avoid moving to cells marked visited (v) unless you have no other valid option.\n"
            "- Prefer moves that lead to NEW cells.\n"
            "- Stop after one successful move.\n"
        )

        user = "Current state JSON:\n" f"{state_json}\n\n" "Make progress. Use tools only."

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "sense",
                    "description": "Read robot sensors: how many steps possible in each direction.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move",
                    "description": "Move the robot in a direction for N steps.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                            "steps": {"type": "integer", "minimum": 0, "maximum": 999},
                        },
                        "required": ["direction", "steps"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_state",
                    "description": "Get compact state (position, last moves, last sensor, visited count).",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

        for tool_call_i in range(self.max_tool_calls_per_turn):
            log(f"🌐 OpenAI request start (i={tool_call_i}, model={self.model}, timeout={self.request_timeout_s}s)")
            t0 = time.time()

            try:
                coro = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                resp = await asyncio.wait_for(coro, timeout=self.request_timeout_s)
            except asyncio.TimeoutError:
                return StepResult(did_move=False, done=self._at_goal(maze, robot), message="💥 OpenAI call TIMEOUT.", new_position=robot.position)
            except Exception as e:
                log(f"💥 OpenAI call FAILED: {type(e).__name__}: {e}")
                log(traceback.format_exc())
                return StepResult(did_move=False, done=self._at_goal(maze, robot), message="💥 OpenAI call FAILED.", new_position=robot.position)

            dt = time.time() - t0
            log(f"🌐 OpenAI response received in {dt:.2f}s")

            msg = resp.choices[0].message
            tool_calls_obj = getattr(msg, "tool_calls", None)
            tool_calls = list(tool_calls_obj or [])

            # append assistant message BEFORE tool msgs (protocol)
            assistant_entry: Dict[str, Any] = {"role": "assistant", "content": msg.content}
            if tool_calls:
                assistant_entry["tool_calls"] = self._tool_calls_to_dicts(tool_calls_obj)
            messages.append(assistant_entry)

            if not tool_calls:
                return StepResult(did_move=False, done=self._at_goal(maze, robot), message="⚠️ No tool calls returned.", new_position=robot.position)

            for tc in tool_calls:
                fn = tc.function.name
                args_str = tc.function.arguments or "{}"

                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}

                if fn == "sense":
                    out = self.tool_sense(maze, robot)

                elif fn == "get_state":
                    out = self.tool_get_state(maze, robot)

                elif fn == "move":
                    direction = args.get("direction")
                    steps = args.get("steps")

                    if direction not in ("up", "down", "left", "right"):
                        out = {"status": False, "error": f"Invalid direction: {direction!r}"}
                    else:
                        try:
                            steps_int = int(steps)
                        except Exception:
                            steps_int = -1
                        out = self.tool_move(maze, robot, direction=direction, steps=steps_int)

                    if out.get("status") is True:
                        self.step_i += 1
                        rp = robot.position
                        visited_added_rc = [(v["y"], v["x"]) for v in out.get("visited_added", [])]
                        return StepResult(
                            did_move=True,
                            done=bool(out.get("done")),
                            message=f"LLM move -> {direction} {steps}",
                            visited_added_rc=visited_added_rc,
                            new_position=rp,
                        )

                else:
                    out = {"status": False, "error": f"Unknown tool: {fn}"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(out, ensure_ascii=False),
                    }
                )

        self.step_i += 1
        return StepResult(did_move=False, done=self._at_goal(maze, robot), message="⚠️ No successful move this tick.", new_position=robot.position)
