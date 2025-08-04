import random
import numpy as np


def generate_maze(n=5, m=5):
    maze = np.ones((n, m), dtype=object)

    def carve(x, y):
        maze[x, y] = 0
        directions = [[0,2], [2,0], [0,-2], [-2,0]]

        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < n and 0 <= ny < m and maze[nx, ny] == 1:
                maze[x + dx//2, y + dy//2 ] = 0
                carve(nx,ny)


    carve(1,1)
    maze[0,0] = "S"
    maze[n -2, m - 2] = "E"

    return maze

if __name__ == "__main__":
    maze = generate_maze()
    print(maze)

