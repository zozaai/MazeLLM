import random

def generate_maze(n=5, m=5):
    # Placeholder: generate a blank maze grid
    return [[' ' for _ in range(m)] for _ in range(n)]

if __name__ == "__main__":
    maze = generate_maze()
    for row in maze:
        print("".join(row))
