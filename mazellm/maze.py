import random
import numpy as np


class Maze():
    def __init__(self, n: int = 5, m: int = 5):
        self.n = n
        self.m = m
        self.board = np.ones((self.n, self.m), dtype=object)

    def generate_maze(self):
        def carve(x, y):
            self.board[x, y] = 0
            directions = [[0,2], [2,0], [0,-2], [-2,0]]

            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.n and 0 <= ny < self.m and self.board[nx, ny] == 1:
                    self.board[x + dx//2, y + dy//2 ] = 0
                    carve(nx,ny)


        carve(1,1)
        self.board[0,0] = "S"
        self.board[self.n -2, self.m - 2] = "E"

if __name__ == "__main__":
    maze = Maze()    
    maze.generate_maze()
    print(maze.board)

