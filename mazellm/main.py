# main.py
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.binding import Binding

from mazellm.maze import Maze
from mazellm.robot import Robot
from mazellm.agent import LLMAgent
from mazellm.visualizer.panels import InfoPanel, MazePanel


class MazeSolverApp(App):
    """A Textual app to solve mazes with an LLM agent."""

    # Add this BINDINGS list to your App
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
    ]

    CSS = """
    Horizontal { height: 100%; }
    InfoPanel { width: 30%; border: solid green; padding:1; }
    MazePanel { width: 70%; border: solid cyan; padding:1; }
    """

    def __init__(self, maze_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.maze = Maze(n=10, m=10) # Using a larger maze
        self.maze.generate_maze() # Generate the maze layout
        self.robot = Robot()
        self.agent = LLMAgent()

        self.info = InfoPanel()
        self.maze_v = MazePanel(self.maze)

    def compose(self) -> ComposeResult:
        # left/right split
        with Horizontal():
            yield self.info
            yield self.maze_v

    async def on_mount(self) -> None:
        # call self.step() every 0.5s
        self.set_interval(0.5, self.step)

    async def step(self) -> None:
        # # 1. sense
        # sensor = self.robot.read_sensor()
        # # 2. update internal map
        # self.maze.update(self.robot.position, sensor)
        # # 3. ask LLM
        # action = await self.agent.decide(self.robot.position, sensor)
        # # 4. move
        # new_pos = self.robot.move_robot(action)

        new_pos = (1, 1)
        sensor = (4, 5, 6)
        action = "move right"

        # 5. update UI
        self.info.update(position=new_pos, sensor=sensor, last_move=action)
        self.maze_v.update(position=new_pos)

        # 6. check goal
        if new_pos == (self.maze.n - 1, self.maze.m - 1):
            self.exit()  # or show success message


if __name__ == "__main__":
    MazeSolverApp().run()