"""The tool-use loop that lets the LLM drive the robot through the maze."""
from __future__ import annotations

import json

from ..maze.maze import Maze
from ..maze.robot import Robot
from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, move, sense_surroundings

NUDGE_MESSAGE = "Please call sense_surroundings or move to continue."

# Index of the memory message within self.messages — refreshed in place every
# turn (rather than appended) so context stays bounded and the model always
# sees an up-to-date map right after the system prompt.
MEMORY_MESSAGE_INDEX = 1


class MazeSolvingAgent:
    def __init__(self, maze: Maze, robot: Robot, llm_client: LLMClient, max_steps: int = 200):
        self.maze = maze
        self.robot = robot
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_memory_message()},
        ]
        self._pending_sensed: dict[str, dict] = {}

    def _build_memory_message(self) -> str:
        """Render everything known about the maze so far — the robot's
        position, the path it took to get here, and a full-size grid of the
        maze — so a model with weak long-context tracking still has explicit,
        accurate grounding every turn instead of relying on scrollback.

        The grid always spans the maze's true width/height and always marks
        the start and end cells — a robot placed in a room would know the
        room's extent and its target destination even before exploring it,
        the same way the human-facing board always shows the full grid and
        the end flag regardless of fog-of-war. Only wall/open status is
        actually gated on having sensed that cell ('?' until then).
        """
        known = self.robot.known_cells
        visited = [self.maze.start] + [record.position_after for record in self.robot.history]

        def symbol(cell: tuple[int, int]) -> str:
            if cell == self.robot.position:
                return "R"
            if cell == self.maze.start:
                return "S"
            if cell == self.maze.end:
                return "E"
            status = known.get(cell)
            if status == "wall":
                return "#"
            if status == "open":
                return "."
            return "?"

        header = "     " + " ".join(f"{x:>2}" for x in range(self.maze.width))
        rows = [header]
        for y in range(self.maze.height):
            row_cells = " ".join(f"{symbol((x, y)):>2}" for x in range(self.maze.width))
            rows.append(f"y={y:<3}{row_cells}")
        grid = "\n".join(rows)

        path = " -> ".join(f"({x},{y})" for x, y in visited)

        return (
            f"Current position: {tuple(self.robot.position)}\n"
            f"Path so far: {path}\n"
            f"Known map so far (full maze size: {self.maze.width}x{self.maze.height}, rows=y, columns=x):\n"
            "  '?'=unexplored   '.'=open   '#'=wall\n"
            "  'S'=start   'E'=end   'R'=you now\n"
            f"{grid}"
        )

    async def run_step(self) -> list[dict]:
        """Run one LLM turn and execute whatever tool calls it makes.

        Returns a list of step events, always starting with a "memory" event
        showing exactly what was fed to the LLM for this turn, e.g.
        {"type": "memory", "content": "Current position: ...\\nPath so far: ...\\n..."},
        followed by zero or more of:
        {"type": "sense", "position": [x, y],
         "sensed": {"up": {"distance": 2, "blocked_by": "wall", "end_at": None}, ...},
         "reasoning": "..."} or
        {"type": "move", "direction": "up", "distance_requested": 3, "distance_moved": 2,
         "success": True, "position_before": [x, y], "position_after": [x, y],
         "reasoning": "..."}.
        """
        memory_content = self._build_memory_message()
        self.messages[MEMORY_MESSAGE_INDEX]["content"] = memory_content
        response = await self.llm_client.chat(self.messages, TOOL_SCHEMAS)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        assistant_message: dict = {"role": "assistant", "content": message.content}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in tool_calls
            ]
        self.messages.append(assistant_message)

        events: list[dict] = [{"type": "memory", "content": memory_content}]
        for call in tool_calls:
            args = json.loads(call.function.arguments or "{}")
            event, result = self._execute_tool_call(call.function.name, args, message.content)
            events.append(event)
            self.messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )

        if not tool_calls:
            self.messages.append({"role": "user", "content": NUDGE_MESSAGE})

        return events

    def _execute_tool_call(self, name: str, args: dict, reasoning: str | None) -> tuple[dict, dict]:
        if name == "sense_surroundings":
            sensed = sense_surroundings(self.maze, self.robot)
            self._pending_sensed = sensed
            event = {
                "type": "sense",
                "position": list(self.robot.position),
                "sensed": sensed,
                "reasoning": reasoning,
            }
            return event, sensed

        if name == "move":
            direction = args.get("direction", "")
            distance = int(args.get("distance", 1))
            position_before = self.robot.position
            success, msg, new_position, moved = move(self.maze, self.robot, direction, distance)
            if moved > 0:
                self.robot.record_step(self._pending_sensed, direction, new_position)
            event = {
                "type": "move",
                "direction": direction,
                "distance_requested": distance,
                "distance_moved": moved,
                "success": success,
                "position_before": list(position_before),
                "position_after": list(new_position),
                "reasoning": reasoning,
            }
            return event, {"success": success, "message": msg, "distance_moved": moved}

        event = {"type": "error", "message": f"unknown tool: {name}", "reasoning": reasoning}
        return event, {"error": f"unknown tool: {name}"}

    def is_done(self) -> bool:
        return self.robot.position == self.maze.end or len(self.robot.history) >= self.max_steps
