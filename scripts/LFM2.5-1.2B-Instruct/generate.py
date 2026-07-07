#!/usr/bin/env python3
"""Generate a maze-solving SFT dataset for `unsloth/LFM2.5-1.2B-Instruct`.

Thin model-specific entrypoint: it uses the generic engine in
`scripts/dataset_gen.py` and the LFM2 formatter in `lfm2_format.py`, and writes
to `data/LFM2.5-1.2B-Instruct/`.

Usage:
    python scripts/LFM2.5-1.2B-Instruct/generate.py --sizes 5,8,10 --per-size 500
    python scripts/LFM2.5-1.2B-Instruct/generate.py --sizes 6 --per-size 3 --preview 2
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
for _p in (ROOT, SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset_gen import dedup, run_sweep, write_dataset  # noqa: E402
from lfm2_format import format_example, tool_schemas  # noqa: E402

MODEL = "LFM2.5-1.2B-Instruct"
DEFAULT_OUT = os.path.join(ROOT, "data", MODEL, "maze_sft.jsonl")


async def _run(args):
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    turns, stats = await run_sweep(sizes, args.per_size, args.density, args.seed, not args.no_rationale)
    raw = len(turns)
    if not args.no_dedup:
        turns = dedup(turns)

    tools_path = write_dataset(
        turns, format_example, tool_schemas(), args.out,
        include_rationale=not args.no_rationale, preview=args.preview,
    )

    acts = collections.Counter(t["meta"]["action"] for t in turns)
    print(f"\nModel: {MODEL}")
    print(f"Mazes: {stats['solved']} solved, {stats['failed']} skipped "
          f"({', '.join(f'{n}x{n}' for n in sizes)}, {args.per_size} each)")
    dd = "" if args.no_dedup else f"  (deduped from {raw}, dropped {raw - len(turns)})"
    print(f"Examples: {len(turns)}{dd}  (sense={acts['sense']}, move={acts['move']})")
    print(f"Wrote {args.out}")
    print(f"Wrote {tools_path}")


def main():
    ap = argparse.ArgumentParser(description=f"Generate a maze-solving SFT dataset for {MODEL}.")
    ap.add_argument("--sizes", default="5,8,10", help="comma-separated square maze sizes, e.g. '5,8,10'")
    ap.add_argument("--per-size", type=int, default=100, help="mazes to generate per size")
    ap.add_argument("--density", type=float, default=0.25, help="wall density (0..0.8)")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed (per-maze seed = base + index)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSONL path")
    ap.add_argument("--no-rationale", action="store_true", help="omit the natural-language reasoning")
    ap.add_argument("--no-dedup", action="store_true", help="keep exact-duplicate examples")
    ap.add_argument("--preview", type=int, default=0, help="print the first N generated examples")
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
