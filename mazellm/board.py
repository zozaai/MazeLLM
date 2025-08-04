from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Header, Footer

class ChessBoardApp(App):
    """A 5x5 chess board app with a two-panel layout."""

    CSS_PATH = "chess_board_panels.tcss"

    def compose(self) -> ComposeResult:
        """Create the UI with left and right panels."""
        yield Header()
        with Horizontal(id="main-container"):
            # Left panel
            with Container(id="left-panel"):
                yield Static("Left Panel", id="left-panel-content")
            # Right panel containing the chess board
            with Container(id="right-panel"):
                with Container(id="chess-board"):
                    for i in range(25):
                        yield Static()
        yield Footer()

if __name__ == "__main__":
    app = ChessBoardApp()
    app.run()