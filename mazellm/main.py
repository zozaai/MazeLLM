# main.py
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.binding import Binding

from mazellm.maze import Maze
from mazellm.robot import Robot
from mazellm.agent import LLMAgent
from mazellm.visualizer.panels import InfoPanel, MazePanel


class MazeSolverApp(App):
    """A Textual app to solve mazes, using CSS Grid for visualization."""

    CSS_PATH = "visualizer/maze_visualizer.tcss"

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
    ]

    def __init__(self, maze_n: int = 15, maze_m: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.maze_n = maze_n
        self.maze_m = maze_m
        self.maze = Maze(n=self.maze_n, m=self.maze_m)
        self.maze.generate_maze()
        self.robot = Robot()
        self.agent = LLMAgent()

        self.info = InfoPanel()
        self.maze_v = MazePanel(maze=self.maze, n=self.maze_n, m=self.maze_m)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield self.info
            yield self.maze_v

    async def on_mount(self) -> None:
        self.set_interval(0.5, self.step)

    async def step(self) -> None:
        new_pos = (1, 1)
        sensor = (4, 5, 6)
        action = "move right"

        self.info.update(position=new_pos, sensor=sensor, last_move=action)
        
        # --- THIS IS THE CORRECTED LINE ---
        # Pass the argument with the name 'new_position' to match the method's definition
        self.maze_v.update(new_position=new_pos)

        if new_pos == (self.maze.n - 2, self.maze.m - 2):
            self.exit()

if __name__ == "__main__":
    MazeSolverApp().run()