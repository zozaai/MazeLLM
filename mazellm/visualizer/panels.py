# visualizer/panels.py
from textual.app import ComposeResult
from textual.containers import Container
from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static

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

class MazePanel(Container):
    """A widget to display the maze using a CSS Grid."""

    def __init__(self, **kwargs) -> None:
        # Pop our custom arguments from the keyword arguments dict.
        self.maze = kwargs.pop("maze")
        self.n = kwargs.pop("n")
        self.m = kwargs.pop("m")

        # Now, kwargs only contains arguments meant for the parent class.
        super().__init__(**kwargs)

        # Initialize our custom attributes.
        self.robot_position = None

    def compose(self) -> ComposeResult:
        """Create the grid of Static widgets."""
        with Container(id="maze-grid"):
            for y in range(self.m):
                for x in range(self.n):
                    yield Static(id=f"cell-{y}-{x}")

    def on_mount(self) -> None:
        """Set up the grid styles and initial state."""
        grid = self.query_one("#maze-grid")
        grid.styles.grid_size = f"{self.n} {self.m}"

        for y in range(self.m):
            for x in range(self.n):
                widget = self.query_one(f"#cell-{y}-{x}", Static)
                cell_value = self.maze.board[y, x]

                if cell_value == 1:
                    widget.add_class("blocked-cell")
                elif cell_value == "S":
                    widget.add_class("start-cell")
                    widget.update("S")
                elif cell_value == "E":
                    widget.add_class("end-cell")
                    widget.update("E")
                else:
                    widget.add_class("free-cell")
                    widget.update("·")

    def update(self, new_position: tuple[int, int]) -> None:
        """Move the robot to a new position."""
        if self.robot_position:
            y, x = self.robot_position[1], self.robot_position[0]
            widget = self.query_one(f"#cell-{y}-{x}", Static)
            cell_value = self.maze.board[y, x]
            if cell_value == "S":
                widget.update("S")
            elif cell_value == "E":
                widget.update("E")
            else:
                widget.update("·")

        y, x = new_position[1], new_position[0]
        self.query_one(f"#cell-{y}-{x}", Static).update("🤖")
        self.robot_position = new_position