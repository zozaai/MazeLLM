# maze.py
import random
import numpy as np

class Maze():
    def __init__(self, n: int = 5, m: int = 5):
        self.n = n # in x direction
        self.m = m # in y direction
        self.board = np.ones((self.n, self.m), dtype=object)

    def generate_maze(self):
        def carve(x, y):
            self.board[y, x] = 0 # Use (y, x) for numpy array indexing
            directions = [[0, 2], [2, 0], [0, -2], [-2, 0]]

            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if 0 <= ny < self.m and 0 <= nx < self.n and self.board[ny, nx] == 1:
                    self.board[y + dy // 2, x + dx // 2] = 0
                    carve(nx, ny)

        # Start carving from a valid position
        carve(1, 1)
        # Place Start and End on walkable tiles
        self.board[1, 1] = "S"
        self.board[self.m - 2, self.n - 2] = "E"

    def is_barrier(self, x: int, y: int):
        """Checks if a cell at (x, y) is a barrier (wall)."""
        if 0 <= y < self.m and 0 <= x < self.n:
            return self.board[y, x] == 1
        return True # Out of bounds is considered a barrier

if __name__ == "__main__":
    maze = Maze()
    maze.generate_maze()
    print(maze.board)