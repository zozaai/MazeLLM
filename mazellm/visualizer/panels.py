# visualizer/panels.py

from textual.widget import Widget
from textual.reactive import reactive

class InfoPanel(Widget):
    position = reactive((0, 0))
    sensor   = reactive({})
    last_move= reactive(None)

    def update(self, position, sensor, last_move):
        self.position, self.sensor, self.last_move = position, sensor, last_move

    def render(self):
        return (
            f"Pos: {self.position}\n"
            f"Sensor: {self.sensor}\n"
            f"Last: {self.last_move}\n"
        )

class MazePanel(Widget):
    def __init__(self, maze, **kwargs):
        super().__init__(**kwargs)
        self.maze = maze
        self.position = (0,0)

    def update(self, position):
        self.position = position

    def render(self):
        rows = []
        for y in range(self.maze.size):
            row = []
            for x in range(self.maze.size):
                if (x,y) == self.position:
                    row.append("🤖")
                elif self.maze.is_barrier(x,y):
                    row.append("■")
                else:
                    row.append("·")
            rows.append("".join(row))
        return "\n".join(rows)
