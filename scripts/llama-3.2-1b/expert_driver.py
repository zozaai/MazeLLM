"""D* Lite expert driver for the Llama-3.2-1B maze SFT dataset.

Self-contained for this target model (nothing shared with other model dirs):
an `ExpertLLMClient` that plugs into the real `MazeSolvingAgent` in place of an
actual LLM, deciding each tool call from a D* Lite planner under fog-of-war; a
sweep over random mazes to collect neutral per-decision turns; duplicate
capping; and the JSONL writer.

Identical to `scripts/llama-3.2-3b/expert_driver.py` — duplicated rather than
shared per this repo's per-target-model convention (see that dir's README).
The neutral turns this produces are model-family agnostic; only the formatter
(`llama_format.py`) is chat-template-specific, and Llama-3.2-1B-Instruct and
-3B-Instruct ship byte-identical `chat_template.jinja` (verified by diffing
both tokenizers' `chat_template`), so the same format applies unchanged.

Key design point: the expert takes **one action per turn, alternating
sense -> move -> sense -> ...**, so each *move* decision is conditioned on the
map *after* the preceding sense revealed the surroundings — it teaches "sense
the unknown before you step into it" rather than guessing a move from a map
full of '?'.
"""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

from backend.agent.dstar_lite import DStarLite
from backend.agent.llm_agent import MEMORY_MESSAGE_INDEX, MazeSolvingAgent
from backend.maze.generator import generate_random_maze
from backend.maze.robot import DIRECTIONS, Robot

_DELTA_TO_DIRECTION = {delta: name for name, delta in DIRECTIONS.items()}
_REL_X = {1: "east", -1: "west", 0: ""}
_REL_Y = {1: "south", -1: "north", 0: ""}


def _relative(pos, goal) -> str:
    parts = [_REL_Y[(goal[1] > pos[1]) - (goal[1] < pos[1])],
             _REL_X[(goal[0] > pos[0]) - (goal[0] < pos[0])]]
    return "-".join(p for p in parts if p) or "here"


class ExpertLLMClient:
    """Same async `chat(messages, tools)` interface as the real `LLMClient`, so
    it drops straight into `MazeSolvingAgent` — but decides each tool call from
    a D* Lite planner instead of calling a model. Records a *neutral* decision
    per turn (system prompt, memory grid, tool name + args, rationale); the
    Llama-specific formatter (`llama_format.py`) renders those into Llama's
    chat/tool-call format."""

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


async def solve_maze(w, h, density, seed, rationale):
    """Run the expert over one maze; return (reached_goal, neutral_turns)."""
    maze = generate_random_maze(w, h, density, seed=seed)
    robot = Robot(position=maze.start)
    client = ExpertLLMClient(maze, robot, rationale=rationale)
    max_steps = 4 * w * h
    agent = MazeSolvingAgent(maze, robot, client, max_steps=max_steps)
    guard = 10 * w * h
    turns = 0
    while not agent.is_done() and turns < guard:
        await agent.run_step()
        turns += 1
    reached = tuple(robot.position) == maze.end
    for t in client.examples:
        t["meta"].update(size=[w, h], density=density, seed=seed)
    return reached, client.examples


async def run_sweep(sizes, per_size, density, seed, rationale, log=print):
    """Sweep square maze sizes x per_size mazes; return (neutral_turns, stats)."""
    turns, solved, failed, idx = [], 0, 0, 0
    for n in sizes:
        for _ in range(per_size):
            reached, exs = await solve_maze(n, n, density, seed + idx, rationale)
            idx += 1
            if reached:
                solved += 1
                turns.extend(exs)
            else:
                failed += 1
                log(f"  ! n={n} seed={seed + idx - 1} did not reach the goal — skipped")
    return turns, {"solved": solved, "failed": failed}


def cap_duplicates(turns, cap=0):
    """Optionally cap how many copies of each identical decision are kept.

    For behavior cloning you generally WANT example frequency to track how often
    a state is visited (every episode starts by sensing at the start cell, so
    that decision is legitimately common). So the default (cap=0) keeps all
    duplicates. A positive `cap` only trims the extreme repeats (e.g. the
    500-per-size start-sense) to reduce bloat without erasing the signal.
    Keyed on (memory, tool name, args), which fully determines an example."""
    if not cap:
        return turns
    counts, out = {}, []
    for t in turns:
        key = json.dumps([t["memory"], t["name"], t["args"]], sort_keys=True)
        h = hashlib.md5(key.encode()).hexdigest()
        counts[h] = counts.get(h, 0) + 1
        if counts[h] <= cap:
            out.append(t)
    return out


def write_dataset(turns, format_example, tools, out_path, include_rationale=True, preview=0):
    """Write JSONL (one formatted example per turn) plus a sibling
    `tool_schemas.json`."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        for t in turns:
            f.write(json.dumps(format_example(t, include_rationale=include_rationale)) + "\n")
    tools_path = os.path.join(out_dir, "tool_schemas.json")
    with open(tools_path, "w") as f:
        json.dump(tools, f, indent=2)

    for t in turns[:preview]:
        ex = json.loads(json.dumps(format_example(t, include_rationale=include_rationale)))
        mem = ex["messages"][1]["content"]
        ex["messages"][1]["content"] = mem[:120] + f"\n... [{len(mem)} chars total]"
        print("\n--- sample example ---")
        print(json.dumps(ex, indent=2))
    return tools_path
