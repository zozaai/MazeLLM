#!/usr/bin/env python3
"""Evaluate one or more Ollama-served models (e.g. base vs fine-tuned
Llama-3.2-1B) against the D* Lite reference, over mazes none of them trained
on.

Drives the *real* `MazeSolvingAgent` + `LLMClient` against a live Ollama
server (same code path the actual app uses over `/ws/solve`) — this is not a
simulation, it makes real requests. Point `--models` at as many Ollama tags as
you want to compare in one run (e.g. the stock `llama3.2:1b` next to your
fine-tuned `llama-3.2-1b-maze`).

IMPORTANT: use mazes the fine-tune never saw. If you generated training data
with `generate.py --sizes 5,8,10 --seed 0`, evaluate with a disjoint size
(e.g. `--sizes 7,9`) and/or a seed base far outside that sweep's range (the
default `--seed 1000000` here is deliberately far from `generate.py`'s
default `--seed 0` sweep, but only YOU know the actual training run's
sizes/seed range — double check before trusting the numbers).

Usage:
    python scripts/llama-3.2-1b/eval.py \\
        --models llama3.2:1b,llama-3.2-1b-maze \\
        --sizes 7,9 --per-size 20
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.agent.dstar_lite import DStarLiteAgent  # noqa: E402
from backend.agent.llm_agent import MazeSolvingAgent  # noqa: E402
from backend.agent.llm_client import LLMClient  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.maze.generator import generate_random_maze  # noqa: E402
from backend.maze.robot import Robot  # noqa: E402


async def dstar_reference(w, h, density, seed):
    """Cells the D* Lite expert travels on this maze — the fog-of-war yardstick
    a "perfect" model would match (ratio 1.00)."""
    maze = generate_random_maze(w, h, density, seed=seed)
    robot = Robot(position=maze.start)
    agent = DStarLiteAgent(maze, robot, max_steps=4 * w * h)
    cells = 0
    while not agent.is_done():
        for ev in await agent.run_step():
            if ev["type"] == "move":
                cells += ev["distance_moved"]
    return cells if tuple(robot.position) == maze.end else None


async def run_model(settings, w, h, density, seed, max_steps):
    """Run one model over one maze; return a per-maze metrics dict."""
    maze = generate_random_maze(w, h, density, seed=seed)
    robot = Robot(position=maze.start)
    client = LLMClient(settings)
    agent = MazeSolvingAgent(maze, robot, client, max_steps=max_steps)

    cells = blocked = revisits = invalid = turns = 0
    visited = {tuple(robot.position)}
    guard = 10 * w * h
    try:
        while not agent.is_done() and turns < guard:
            events = await agent.run_step()
            turns += 1
            if len(events) == 1:  # only the "memory" event — no tool call was parsed this turn
                invalid += 1
            for ev in events:
                if ev["type"] == "move":
                    cells += ev["distance_moved"]
                    if not ev["success"]:
                        blocked += 1
                    pos = tuple(ev["position_after"])
                    if pos in visited:
                        revisits += 1
                    visited.add(pos)
    except Exception as exc:  # noqa: BLE001 — one bad maze/response shouldn't kill the sweep
        return {"solved": False, "cells": cells, "blocked": blocked, "revisits": revisits,
                "invalid": invalid, "turns": turns, "error": str(exc)}

    solved = tuple(robot.position) == maze.end
    return {"solved": solved, "cells": cells, "blocked": blocked, "revisits": revisits,
            "invalid": invalid, "turns": turns, "error": None}


def summarize(label, results):
    n = len(results)
    solved = [r for r in results if r["solved"]]
    sr = len(solved) / n if n else 0.0
    ratios = [r["cells"] / r["dstar_cells"] for r in solved if r.get("dstar_cells")]
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731

    print(f"\n===== {label} =====")
    print(f"  mazes evaluated     : {n}")
    print(f"  success rate        : {sr:.1%}  ({len(solved)}/{n} reached the goal)")
    print(f"  path length / opt.  : {avg(ratios):.2f}x   (1.00 = matches D* Lite; solved mazes only)")
    print(f"  blocked-move tries  : {avg([r['blocked'] for r in results]):.2f} per maze")
    print(f"  revisited cells     : {avg([r['revisits'] for r in results]):.2f} per maze")
    print(f"  invalid/unparsed    : {avg([r['invalid'] for r in results]):.2f} per maze  (no tool call parsed)")
    print(f"  turns               : {avg([r['turns'] for r in results]):.1f} per maze")
    errors = [r["error"] for r in results if r.get("error")]
    if errors:
        print(f"  errors              : {len(errors)}/{n}  (e.g. {errors[0]!r})")
    return {"success_rate": sr, "path_ratio": avg(ratios),
            "blocked": avg([r["blocked"] for r in results]),
            "revisits": avg([r["revisits"] for r in results])}


async def _run(args):
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    mazes = [(n, n, args.density, args.seed + i)
              for n in sizes for i in range(args.per_size)]

    print(f"Reference: computing D* Lite optimal path length for {len(mazes)} held-out mazes...")
    dstar_cells = {}
    for (w, h, density, seed) in mazes:
        dstar_cells[(w, h, seed)] = await dstar_reference(w, h, density, seed)

    for model in models:
        settings = Settings(
            llm_provider="ollama", llm_base_url=args.base_url, llm_api_key="",
            llm_model=model, llm_site_url="", llm_site_name="", host="", port=0,
        )
        results = []
        for (w, h, density, seed) in mazes:
            r = await run_model(settings, w, h, density, seed, args.max_steps)
            r["dstar_cells"] = dstar_cells[(w, h, seed)]
            results.append(r)
        summarize(model, results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True,
                     help="comma-separated Ollama model tags to compare, e.g. llama3.2:1b,llama-3.2-1b-maze")
    ap.add_argument("--base-url", default="http://localhost:11434/v1", help="Ollama's OpenAI-compatible endpoint")
    ap.add_argument("--sizes", default="7,9", help="comma-separated square maze sizes — use sizes NOT in training")
    ap.add_argument("--per-size", type=int, default=20, help="held-out mazes to evaluate per size")
    ap.add_argument("--density", type=float, default=0.25, help="wall density (0..0.8)")
    ap.add_argument("--seed", type=int, default=1_000_000,
                     help="base RNG seed — keep far outside generate.py's training seed range")
    ap.add_argument("--max-steps", type=int, default=100)
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
