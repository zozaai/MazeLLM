# agent.py
import random
from typing import Tuple, Optional


class LLMAgent:
    def __init__(self):
        pass

    async def next_step(self, maze, robot) -> Optional[Tuple[int, int]]:
        """
        Return a random location within maze bounds.
        For now this ignores walls/goals — just demo movement.
        """
        rows = getattr(maze, "n", None) or getattr(maze, "rows", None) or 5
        cols = getattr(maze, "m", None) or getattr(maze, "cols", None) or rows
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        return r, c
