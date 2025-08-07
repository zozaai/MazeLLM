class RobotSensor:
    """
    Provides robot's current state including position and local environment.
    """

    def __init__(self, maze: list[list[object]], start_position: tuple[int, int]):
        self.maze = maze
        self.position = start_position

    def get_state(self) -> dict:
        """
        Returns the robot's current state: position and surroundings.
        """
        x, y = self.position
        surroundings = self._get_surroundings(x, y)
        return {
            "position": (x, y),
            "surroundings": surroundings
        }

    def set_position(self, new_position: tuple[int, int]) -> None:
        """
        Updates the robot's current position.
        """
        self.position = new_position

    def _get_surroundings(self, x: int, y: int) -> dict:
        """
        Returns values of neighboring cells in each direction.
        """
        directions = {
            "up":    (x - 1, y),
            "down":  (x + 1, y),
            "left":  (x, y - 1),
            "right": (x, y + 1)
        }

        result = {}
        for direction, (dx, dy) in directions.items():
            if 0 <= dx < len(self.maze) and 0 <= dy < len(self.maze[0]):
                result[direction] = self.maze[dx][dy]
            else:
                result[direction] = "wall"  # Outside bounds is wall

        return result
