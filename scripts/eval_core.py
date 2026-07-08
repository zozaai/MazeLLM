"""Generic evaluation helpers (model-agnostic).

Provides the D* Lite reference (fog-of-war optimal path length for a maze) and
aggregation/printing of run metrics. A model-specific evaluator drives its model
over the same mazes and feeds per-maze results here.
"""
from __future__ import annotations

from backend.agent.dstar_lite import DStarLiteAgent
from backend.maze.generator import generate_random_maze
from backend.maze.robot import Robot


async def dstar_reference(w, h, density, seed):
    """Cells the D* Lite expert travels on this maze (the fog-of-war yardstick)."""
    maze = generate_random_maze(w, h, density, seed=seed)
    robot = Robot(position=maze.start)
    agent = DStarLiteAgent(maze, robot, max_steps=4 * w * h)
    cells = 0
    while not agent.is_done():
        for ev in await agent.run_step():
            if ev["type"] == "move":
                cells += ev["distance_moved"]
    return cells if tuple(robot.position) == maze.end else None


def summarize(label, results):
    """results: list of per-maze dicts with keys
    solved, cells, dstar_cells, blocked, revisits, invalid, turns."""
    n = len(results)
    solved = [r for r in results if r["solved"]]
    sr = len(solved) / n if n else 0.0
    ratios = [r["cells"] / r["dstar_cells"] for r in solved if r.get("dstar_cells")]
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731

    print(f"\n===== {label} =====")
    print(f"  mazes evaluated     : {n}")
    print(f"  success rate        : {sr:.1%}  ({len(solved)}/{n} reached the goal)")
    print(f"  path length / opt.  : {avg(ratios):.2f}x   (1.00 = matches D* Lite; solved mazes only)")
    print(f"  blocked-move tries  : {avg([r['blocked'] for r in results]):.2f} per maze   (moves into a wall)")
    print(f"  revisited cells     : {avg([r['revisits'] for r in results]):.2f} per maze")
    print(f"  invalid/unparsed    : {avg([r['invalid'] for r in results]):.2f} per maze")
    print(f"  turns               : {avg([r['turns'] for r in results]):.1f} per maze")
    return {"success_rate": sr, "path_ratio": avg(ratios),
            "blocked": avg([r["blocked"] for r in results]),
            "revisits": avg([r["revisits"] for r in results])}
