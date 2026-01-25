# maze.py
import argparse
import random
import numpy as np

"""
# default 5x5
python -m mazellm.maze

# rectangular maze
python -m mazellm.maze -n 8 -m 4 

# reproducible maze
python -m mazellm.maze -n 10 -m 10 --seed 123

"""


class Maze:
    def __init__(self, n: int = 5, m: int = 5, seed: int | None = None):
        self.n = n  # x direction (columns)
        self.m = m  # y direction (rows)
        self.board = np.ones((self.m, self.n), dtype=object)
        self._rng = random.Random(seed)

    def generate_maze(self):
        def carve(x: int, y: int):
            self.board[y, x] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            self._rng.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (
                    0 <= ny < self.m
                    and 0 <= nx < self.n
                    and self.board[ny, nx] == 1
                ):
                    self.board[y + dy // 2, x + dx // 2] = 0
                    carve(nx, ny)

        carve(0, 0)

        # Start and End
        self.board[0, 0] = "S"
        self.board[self.m - 1, self.n - 1] = "E"

    def is_barrier(self, x: int, y: int) -> bool:
        if 0 <= y < self.m and 0 <= x < self.n:
            return self.board[y, x] == 1
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a randomized maze using recursive backtracking.")
    parser.add_argument("-n", "--n", type=int, default=5, help="maze width (x / columns)")
    parser.add_argument("-m", "--m", type=int, default=5, help="maze height (y / rows)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.n <= 0 or args.m <= 0:
        raise SystemExit("n and m must be positive integers.")

    maze = Maze(n=args.n, m=args.m, seed=args.seed)
    maze.generate_maze()

    print(maze.board)


if __name__ == "__main__":
    main()
