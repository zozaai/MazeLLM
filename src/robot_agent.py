class RobotSensor:
    """
    Robot sensor that detects the robot's position and immediate surroundings in the maze.
    """

    def __init__(self, maze, start_position):
        """
        maze: 2D list (0 = free cell, 1 = wall)
        start_position: tuple (x, y)
        """
        self.maze = maze
        self.position = start_position

    def get_position(self):
        """Return current robot position."""
        return self.position

    def get_surroundings(self):
        """Return surroundings in up/down/left/right directions."""
        x, y = self.position
        directions = {
            "up":    (x - 1, y),
            "down":  (x + 1, y),
            "left":  (x, y - 1),
            "right": (x, y + 1)
        }
        surroundings = {}
        for direction, (dx, dy) in directions.items():
            if 0 <= dx < len(self.maze) and 0 <= dy < len(self.maze[0]):
                surroundings[direction] = self.maze[dx][dy]  # 0 = free, 1 = wall
            else:
                surroundings[direction] = 1  # Out of bounds = wall
        return surroundings


class RobotMove:
    """
    Handles movement of the robot in the maze.
    """

    def __init__(self, sensor: RobotSensor):
        self.sensor = sensor

    def move(self, direction):
        """Move robot in given direction if possible."""
        moves = {
            "up":    (-1, 0),
            "down":  (1, 0),
            "left":  (0, -1),
            "right": (0, 1)
        }

        if direction not in moves:
            return False  # Invalid direction

        dx, dy = moves[direction]
        x, y = self.sensor.get_position()
        new_x, new_y = x + dx, y + dy

        if not (0 <= new_x < len(self.sensor.maze) and 0 <= new_y < len(self.sensor.maze[0])):
            return False  # Out of bounds

        if self.sensor.maze[new_x][new_y] == 1:
            return False  # Hit wall

        # Update position
        self.sensor.position = (new_x, new_y)
        return True
