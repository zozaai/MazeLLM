# main.py
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from visualizer.panels import InfoPanel, MazePanel

from maze import Maze
from robot import Robot
from llm_agent import LLMAgent

from mazellm import maze


class MazeSolverApp(App):
    CSS = """
    Horizontal { height: 100%; }
    InfoPanel { width: 30%; border: solid green; padding:1; }
    MazePanel { width: 70%; border: solid cyan; padding:1; }
    """


    def __init__(self, maze_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.maze   = Maze(size=maze_size)
        self.robot  = Robot()
        self.agent  = LLMAgent()

        self.info   = InfoPanel()
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
        # 1. sense
        sensor = self.robot.read_sensor()
        # 2. update internal map
        self.maze.update(self.robot.position, sensor)
        # 3. ask LLM
        action = await self.agent.decide(self.robot.position, sensor)
        # 4. move
        new_pos = self.robot.move_robot(action)
        # 5. update UI
        self.info.update(position=new_pos, sensor=sensor, last_move=action)
        self.maze_v.update(position=new_pos)

        # 6. check goal
        if new_pos == (self.maze.size - 1, self.maze.size - 1):
            self.exit()  # or show success message

if __name__ == "__main__":
    MazeSolverApp().run()
