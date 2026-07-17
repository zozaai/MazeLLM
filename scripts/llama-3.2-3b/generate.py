#!/usr/bin/env python3
"""Generate a maze-solving SFT dataset for `unsloth/Llama-3.2-3B-Instruct`.

Usage:
    python scripts/llama-3.2-3b/generate.py --sizes 5,6,7,8,9,10 --per-size 1000
    python scripts/llama-3.2-3b/generate.py --sizes 6 --per-size 3 --preview 2

Reserve at least one maze size (or a distinct seed range) that this command
never touches — `eval.py` needs mazes the model never trained on. See the
README for the recommended held-out split.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from expert_driver import cap_duplicates, run_sweep, write_dataset  # noqa: E402
from llama_format import format_example, tool_schemas  # noqa: E402

MODEL = "llama-3.2-3b"
DEFAULT_OUT = os.path.join(ROOT, "data", MODEL, "maze_sft.jsonl")


async def _run(args):
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    turns, stats = await run_sweep(sizes, args.per_size, args.density, args.seed, not args.no_rationale)
    raw = len(turns)
    turns = cap_duplicates(turns, args.dup_cap)

    tools_path = write_dataset(
        turns, format_example, tool_schemas(), args.out,
        include_rationale=not args.no_rationale, preview=args.preview,
    )

    acts = collections.Counter(t["meta"]["action"] for t in turns)
    print(f"\nModel: {MODEL}")
    print(f"Mazes: {stats['solved']} solved, {stats['failed']} skipped "
          f"({', '.join(f'{n}x{n}' for n in sizes)}, {args.per_size} each)")
    dd = f"  (capped at {args.dup_cap}/dup, from {raw})" if args.dup_cap else ""
    print(f"Examples: {len(turns)}{dd}  (sense={acts['sense']}, move={acts['move']})")
    print(f"Wrote {args.out}")
    print(f"Wrote {tools_path}")


def main():
    ap = argparse.ArgumentParser(description=f"Generate a maze-solving SFT dataset for {MODEL}.")
    ap.add_argument("--sizes", default="5,8,10", help="comma-separated square maze sizes")
    ap.add_argument("--per-size", type=int, default=100, help="mazes to generate per size")
    ap.add_argument("--density", type=float, default=0.25, help="wall density (0..0.8)")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed (per-maze seed = base + index)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSONL path")
    ap.add_argument("--no-rationale", action="store_true",
                     help="omit meta.rationale (has no effect on the rendered prompt either way — "
                          "Llama's template drops assistant content when tool_calls is set)")
    ap.add_argument("--dup-cap", type=int, default=0,
                    help="cap identical examples to N copies (0 = keep all; natural frequency is best for BC)")
    ap.add_argument("--preview", type=int, default=0, help="print the first N generated examples")
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
