# backend/baselines/__init__.py
"""Classical pathfinding baselines (BFS/DFS/A*), ported from the original
MazeLLM CLI/TUI project. Kept as a self-contained subsystem (own Maze/Robot/
Position representation) so it doesn't need to share backend.maze's API —
its purpose is to provide ground-truth/optimal-path comparisons against the
LLM agent's move efficiency, not to run inside the same request loop.
"""
from backend.baselines.solver import Solver, StepResult
from backend.baselines.bfs import BFSSolver
from backend.baselines.dfs import DFSSolver
from backend.baselines.astar import AStarSolver

__all__ = [
    "Solver",
    "StepResult",
    "BFSSolver",
    "DFSSolver",
    "AStarSolver",
]
