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
from mazellm.robot import Robot, Direction, Position


@dataclass
class MoveRecord:
    i: int
    direction: Direction
    steps: int
    from_pos: Position
    to_pos: Position
    ok: bool


class LLMAgent:
    """
    LLM-driven agent that can:
      - sense() using Robot.sensor()
      - move(direction, steps) using Robot.move()
      - get_state()

    IMPORTANT:
    - Uses AsyncOpenAI to avoid blocking Textual loop.
    - OpenAI calls are wrapped in a timeout to avoid "stuck forever".
    - Logs aggressively for debugging.
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
        self.model = model
        self.max_total_steps = int(max_total_steps)
        self.max_tool_calls_per_turn = int(max_tool_calls_per_turn)
        self.request_timeout_s = float(request_timeout_s)
        self.also_print = bool(also_print)

        if api_key_env not in os.environ:
            raise RuntimeError(f"{api_key_env} not set in environment.")

        self.client = AsyncOpenAI()

        self.step_i = 0
        self.visited: set[tuple[int, int]] = set()  # (x,y)
        self.moves: List[MoveRecord] = []
        self.last_sensor: Optional[Dict[Direction, int]] = None

    # -------------------------
    # Internal helpers
    # -------------------------
    def _at_goal(self, maze: Maze, robot: Robot) -> bool:
        x, y = robot.position.x, robot.position.y
        return (0 <= y < maze.m) and (0 <= x < maze.n) and (maze.board[y, x] == "E")

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
            "limits": {
                "step_i": self.step_i,
                "max_total_steps": self.max_total_steps,
            },
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

    # -------------------------
    # Tools exposed to the LLM
    # -------------------------
    def tool_sense(self, maze: Maze, robot: Robot) -> Dict[str, Any]:
        sensor = robot.sensor()
        self.last_sensor = sensor
        pos = {"x": robot.position.x, "y": robot.position.y}
        return {"position": pos, "sensor": sensor}

    def tool_move(self, maze: Maze, robot: Robot, direction: Direction, steps: int) -> Dict[str, Any]:
        # Validate using current sensor so model can't hang by repeatedly doing impossible moves
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
            for (x, y) in visited_added_xy:
                self.visited.add((x, y))

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
    # Main decision loop
    # -------------------------
    async def run_until_move_or_done(
        self,
        maze: Maze,
        robot: Robot,
        *,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        log_lines: List[str] = []

        def log(s: str) -> None:
            log_lines.append(s)
            if logger:
                logger(s)
            if self.also_print:
                print(s, flush=True)

        if self._at_goal(maze, robot):
            log("✅ Already at goal.")
            return {"did_move": False, "done": True, "visited_added_rc": [], "log_lines": log_lines}

        if self.step_i >= self.max_total_steps:
            log("⚠️ Reached max_total_steps. Stopping.")
            return {"did_move": False, "done": False, "visited_added_rc": [], "log_lines": log_lines}

        # Always start with a sensor read (keeps last_sensor fresh + gives model a stable state)
        s0 = self.tool_sense(maze, robot)
        log(f"🧭 pre-sense -> {s0}")

        state = self._state_dict(maze, robot)
        state_json = json.dumps(state, ensure_ascii=False)

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
            "- If you return immediately to the previous cell, that is usually bad.\n"
            "Stop after one successful move.\n"
        )

        user = (
            "Current state JSON:\n"
            f"{state_json}\n\n"
            "Make progress. Use tools only."
        )

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

        visited_added_rc: List[tuple[int, int]] = []

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
                log("💥 OpenAI call TIMEOUT.")
                return {"did_move": False, "done": self._at_goal(maze, robot), "visited_added_rc": [], "log_lines": log_lines}
            except Exception as e:
                log(f"💥 OpenAI call FAILED: {type(e).__name__}: {e}")
                log(traceback.format_exc())
                return {"did_move": False, "done": self._at_goal(maze, robot), "visited_added_rc": [], "log_lines": log_lines}

            dt = time.time() - t0
            log(f"🌐 OpenAI response received in {dt:.2f}s")

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            content = (msg.content or "").strip()

            log(f"🧠 LLM content (first 200): {content[:200]!r}")
            log(f"🧰 tool_calls: {0 if not tool_calls else len(tool_calls)}")

            if not tool_calls:
                log("⚠️ No tool calls returned. Ending this tick.")
                break

            for tc in tool_calls:
                fn = tc.function.name
                args_str = tc.function.arguments or "{}"
                log(f"🔧 tool_call raw: {fn}({args_str})")

                try:
                    args = json.loads(args_str)
                except Exception:
                    log("💥 Failed to json.loads(tool arguments). Using empty args.")
                    args = {}

                if fn == "sense":
                    out = self.tool_sense(maze, robot)
                    log(f"↩️ sense -> {out}")

                elif fn == "get_state":
                    out = self.tool_get_state(maze, robot)
                    log(f"↩️ get_state -> {out}")

                elif fn == "move":
                    direction = args.get("direction")
                    steps = args.get("steps")

                    if direction not in ("up", "down", "left", "right"):
                        out = {"status": False, "error": f"Invalid direction from LLM: {direction!r}"}
                        log(f"↩️ move -> {out}")
                    else:
                        try:
                            steps_int = int(steps)
                        except Exception:
                            steps_int = -1

                        out = self.tool_move(maze, robot, direction=direction, steps=steps_int)
                        log(f"↩️ move -> {out}")

                    # Stop after first SUCCESSFUL move
                    if out.get("status") is True:
                        visited_added_rc = [(v["y"], v["x"]) for v in out.get("visited_added", [])]
                        self.step_i += 1
                        return {
                            "did_move": True,
                            "done": bool(out.get("done")),
                            "visited_added_rc": visited_added_rc,
                            "log_lines": log_lines,
                        }

                else:
                    out = {"status": False, "error": f"Unknown tool: {fn}"}
                    log(f"↩️ {out}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(out, ensure_ascii=False),
                    }
                )

        # No successful move this tick
        self.step_i += 1
        return {"did_move": False, "done": self._at_goal(maze, robot), "visited_added_rc": [], "log_lines": log_lines}

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
                    row.append("v" if (x, y) in self.visited else ".")
            lines.append("".join(row))
        return lines
