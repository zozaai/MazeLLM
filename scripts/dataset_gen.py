"""Generic dataset engine (model-agnostic).

Drives the D* Lite expert (via `ExpertLLMClient` + the real `MazeSolvingAgent`)
over many random mazes, producing *neutral* per-decision turns. A per-model
script under `scripts/<model>/` supplies a `format_example` function and tool
schema to turn those neutral turns into that model's chat format, and calls
`write_dataset` to emit the `.jsonl`.

Nothing here is specific to any target model.
"""
from __future__ import annotations

import hashlib
import json
import os

from backend.agent.llm_agent import MazeSolvingAgent
from backend.maze.generator import generate_random_maze
from backend.maze.robot import Robot

from expert_client import ExpertLLMClient


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
    """Sweep square maze sizes × per_size mazes; return (neutral_turns, stats)."""
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
    `tool_schemas.json`. `format_example(turn, include_rationale)` and `tools`
    come from the target model's formatter."""
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
        ex["messages"][1]["content"] = mem[:120] + f"\n… [{len(mem)} chars total]"
        print("\n--- sample example ---")
        print(json.dumps(ex, indent=2))
    return tools_path
