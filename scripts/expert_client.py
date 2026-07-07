"""An expert stand-in for the LLM, used only for dataset generation. (Generic —
not tied to any target model; model-specific formatting lives under
`scripts/<model>/`.)

`ExpertLLMClient` has the same async `chat(messages, tools)` interface as the
real `LLMClient`, so it drops straight into `MazeSolvingAgent` — but instead of
calling a model it decides each tool call from a **D* Lite** planner under
fog-of-war. Each turn it records a *neutral* decision (system prompt, memory
grid, tool name + args, rationale); a per-model formatter turns those into the
model's chat format.

Key design point: it alternates **one tool call per turn** — sense, then move,
then sense, … — so each *move* decision is conditioned on the map *after* the
preceding sense revealed the surroundings. That teaches "sense the unknown
before you step into it" rather than guessing a move from a map full of `?`.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from backend.agent.dstar_lite import DStarLite
from backend.agent.llm_agent import MEMORY_MESSAGE_INDEX
from backend.maze.robot import DIRECTIONS

_DELTA_TO_DIRECTION = {delta: name for name, delta in DIRECTIONS.items()}
_REL_X = {1: "east", -1: "west", 0: ""}
_REL_Y = {1: "south", -1: "north", 0: ""}


def _relative(pos, goal) -> str:
    parts = [_REL_Y[(goal[1] > pos[1]) - (goal[1] < pos[1])],
             _REL_X[(goal[0] > pos[0]) - (goal[0] < pos[0])]]
    return "-".join(p for p in parts if p) or "here"


class ExpertLLMClient:
    def __init__(self, maze, robot, rationale: bool = True):
        self.maze = maze
        self.robot = robot
        self.rationale = rationale
        self.examples: list[dict] = []  # neutral turns: {system, memory, name, args, rationale, meta}
        self._planner: DStarLite | None = None
        self._need_sense = True
        self._turn = 0

    async def chat(self, messages, tools):
        turn = self._turn
        self._turn += 1
        pos = tuple(self.robot.position)
        if self._planner is None:
            self._planner = DStarLite(self.maze.width, self.maze.height, pos, self.maze.end)

        if self._need_sense:
            self._need_sense = False
            name, args, action = "sense_surroundings", {}, "sense"
            rationale = (
                f"I just arrived at {pos} and haven't scanned here yet — "
                "sensing all four directions before I commit to a move."
            )
        else:
            direction, distance = self._plan_move(pos)
            if direction is None:  # optimistic map has no route yet — scan again (rare)
                self._need_sense = False
                name, args, action = "sense_surroundings", {}, "sense"
                rationale = "Re-scanning to update the known map."
            else:
                self._need_sense = True
                name, args, action = "move", {"direction": direction, "distance": distance}, "move"
                rationale = (
                    f"The exit is to the {_relative(pos, self.maze.end)}. {direction.capitalize()} is "
                    f"open toward it on the shortest known route, so I move {direction} {distance}."
                )

        self.examples.append(
            {
                "system": messages[0]["content"],
                "memory": messages[MEMORY_MESSAGE_INDEX]["content"],
                "name": name,
                "args": args,
                "rationale": rationale,
                "meta": {"turn": turn, "action": action, "pos": list(pos)},
            }
        )
        # Response the agent consumes to actually execute the tool (always tool_calls
        # shape — this is internal agent plumbing, independent of the dataset format).
        call = SimpleNamespace(
            id=f"call_{turn}", type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        message = SimpleNamespace(content=rationale if self.rationale else None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def _plan_move(self, pos):
        p = self._planner
        p.start = pos
        new_walls = [c for c, st in self.robot.known_cells.items() if st == "wall" and c not in p.walls]
        p.observe_walls(new_walls)
        path = p.greedy_path()
        if len(path) < 2:
            return None, None
        dx, dy = path[1][0] - pos[0], path[1][1] - pos[1]
        direction = _DELTA_TO_DIRECTION[(dx, dy)]
        straight = 1
        while straight + 1 < len(path) and (
            path[straight + 1][0] - path[straight][0],
            path[straight + 1][1] - path[straight][1],
        ) == (dx, dy):
            straight += 1
        run, c = 0, pos
        while True:
            nc = (c[0] + dx, c[1] + dy)
            if self.robot.known_cells.get(nc) in ("open", "end"):
                run += 1
                c = nc
            else:
                break
        return direction, max(1, min(straight, run))
