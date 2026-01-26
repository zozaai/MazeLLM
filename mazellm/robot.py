# mazellm/robot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, TypedDict

from mazellm.maze import Maze


Direction = Literal["up", "down", "left", "right"]


@dataclass
class Position:
    x: int
    y: int


class MoveResult(TypedDict):
    status: bool
    new_position: Position


class Robot:
    """
    Robot bound to a Maze.

    - Position is (x, y) where:
        x = column index (0..maze.n-1)
        y = row index    (0..maze.m-1)

    Maze uses board[y, x].
    """

    def __init__(self, maze: Maze, position: Position | None = None):
        self.maze: Maze = maze
        self.position: Position = position or Position(x=0, y=0)

    # --------------------------
    # Helpers
    # --------------------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.maze.n and 0 <= y < self.maze.m

    def _is_walkable(self, x: int, y: int) -> bool:
        """Walkable means: inside bounds and NOT a barrier (1)."""
        return self._in_bounds(x, y) and (not self.maze.is_barrier(x=x, y=y))

    # --------------------------
    # Public API
    # --------------------------
    def sensor(self) -> Dict[Direction, int]:
        """
        Return how many cells the robot can move (max) in each direction
        before hitting a wall/barrier or the maze boundary.

        Example:
          {"up": 0, "down": 3, "left": 1, "right": 7}
        """
        x, y = self.position.x, self.position.y

        distances: Dict[Direction, int] = {"up": 0, "down": 0, "left": 0, "right": 0}

        # Up: decreasing y
        steps = 0
        yy = y - 1
        while yy >= 0 and self._is_walkable(x, yy):
            steps += 1
            yy -= 1
        distances["up"] = steps

        # Down: increasing y
        steps = 0
        yy = y + 1
        while yy < self.maze.m and self._is_walkable(x, yy):
            steps += 1
            yy += 1
        distances["down"] = steps

        # Left: decreasing x
        steps = 0
        xx = x - 1
        while xx >= 0 and self._is_walkable(xx, y):
            steps += 1
            xx -= 1
        distances["left"] = steps

        # Right: increasing x
        steps = 0
        xx = x + 1
        while xx < self.maze.n and self._is_walkable(xx, y):
            steps += 1
            xx += 1
        distances["right"] = steps

        return distances

    def move(self, direction: Dict[Direction, int]) -> MoveResult:
        """
        Attempt to move the robot in a given direction for N cells.

        Input example:
          {"left": 3}
          {"down": 1}

        Rules:
        - You can move only through walkable cells (maze.is_barrier == False).
        - If any intermediate cell is blocked OR out of bounds -> fail, no movement.
        - On success, updates robot.position and returns the new Position.

        Returns:
          {"status": True/False, "new_position": Position(...)}
        """
        if not direction:
            return {"status": False, "new_position": Position(self.position.x, self.position.y)}

        dir_name, cells = next(iter(direction.items()))

        if dir_name not in ("up", "down", "left", "right"):
            return {"status": False, "new_position": Position(self.position.x, self.position.y)}

        if not isinstance(cells, int) or cells < 0:
            return {"status": False, "new_position": Position(self.position.x, self.position.y)}

        # No-op
        if cells == 0:
            return {"status": True, "new_position": Position(self.position.x, self.position.y)}

        x0, y0 = self.position.x, self.position.y
        x, y = x0, y0

        for _ in range(cells):
            if dir_name == "up":
                y -= 1
            elif dir_name == "down":
                y += 1
            elif dir_name == "left":
                x -= 1
            else:  # right
                x += 1

            if not self._is_walkable(x, y):
                return {"status": False, "new_position": Position(x0, y0)}

        self.position = Position(x=x, y=y)
        return {"status": True, "new_position": Position(x=x, y=y)}


if __name__ == "__main__":
    import numpy as np

    board = np.array(
        [
            ["S", 1, 0, 0, 0],
            [0,   1, 0, 1, 0],
            [0,   1, 0, 1, 0],
            [0,   1, 0, 1, 0],
            [0,   0, 0, 1, "E"],
        ],
        dtype=object,
    )

    maze = Maze(n=5, m=5, seed=123)
    maze.board = board

    robot = Robot(maze=maze, position=Position(x=0, y=0))  # on "S"

    print("Board:")
    print(maze.board)
    print("\nRobot start:", robot.position)

    print("\nSensor distances:", robot.sensor())

    print("\nTry move right 1 (should fail):")
    print(robot.move({"right": 1}))
    print("Robot now:", robot.position)

    print("\nTry move down 4 (should succeed):")
    print(robot.move({"down": 4}))
    print("Robot now:", robot.position)

    print("\nTry move right 3 from (0,4) (should fail at wall):")
    print(robot.move({"right": 3}))
    print("Robot now:", robot.position)
