import asyncio
import json
from types import SimpleNamespace

from backend.agent.llm_agent import MazeSolvingAgent
from backend.maze.maze import Maze
from backend.maze.robot import Robot


class ScriptedLLMClient:
    """Fake LLM client that plays back a fixed script of tool calls, mimicking
    the shape of an OpenAI chat.completions response — no network involved."""

    def __init__(self, script):
        self.script = list(script)
        self.calls_made = 0

    async def chat(self, messages, tools):
        name, args = self.script[self.calls_made]
        self.calls_made += 1
        tool_call = SimpleNamespace(
            id=f"call_{self.calls_made}",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        message = SimpleNamespace(content=f"deciding to {name}", tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_agent(maze, script, max_steps=10):
    robot = Robot(position=maze.start)
    agent = MazeSolvingAgent(maze, robot, ScriptedLLMClient(script), max_steps=max_steps)
    return agent, robot


def run_until_done(agent):
    async def run():
        events = []
        while not agent.is_done():
            events.extend(await agent.run_step())
        return events

    return asyncio.run(run())


def tool_events(events):
    """Every run_step() call prepends a {"type": "memory", ...} event showing
    what was fed to the LLM that turn — filter it out when a test only cares
    about the resulting sense/move tool events."""
    return [e for e in events if e["type"] != "memory"]


def test_run_step_always_starts_with_a_memory_event():
    maze = Maze(width=2, height=1, start=(0, 0), end=(1, 0))
    agent, robot = make_agent(maze, [("sense_surroundings", {})])

    events = asyncio.run(agent.run_step())

    assert events[0]["type"] == "memory"
    assert "Current position: (0, 0)" in events[0]["content"]


def test_sense_reports_distance_and_end_at():
    maze = Maze(width=4, height=1, start=(0, 0), end=(3, 0))
    agent, robot = make_agent(maze, [("sense_surroundings", {})])

    events = tool_events(asyncio.run(agent.run_step()))

    sensed = events[0]["sensed"]
    assert sensed["right"] == {"distance": 3, "blocked_by": "boundary", "end_at": 3}
    assert sensed["left"] == {"distance": 0, "blocked_by": "boundary", "end_at": None}
    assert sensed["up"] == {"distance": 0, "blocked_by": "boundary", "end_at": None}


def test_sense_stops_at_wall():
    maze = Maze(width=5, height=1, walls={(2, 0)}, start=(0, 0), end=(4, 0))
    agent, robot = make_agent(maze, [("sense_surroundings", {})])

    events = tool_events(asyncio.run(agent.run_step()))

    assert events[0]["sensed"]["right"] == {"distance": 1, "blocked_by": "wall", "end_at": None}


def test_move_with_distance_reaches_end_in_one_call():
    maze = Maze(width=4, height=1, start=(0, 0), end=(3, 0))
    agent, robot = make_agent(maze, [("move", {"direction": "right", "distance": 3})])

    events = tool_events(run_until_done(agent))

    assert robot.position == maze.end
    assert events[0]["distance_requested"] == 3
    assert events[0]["distance_moved"] == 3
    assert events[0]["success"] is True
    assert events[0]["position_after"] == [3, 0]
    assert len(robot.history) == 1  # one move() call, regardless of distance covered


def test_move_marks_every_traversed_cell_as_known():
    # Regression test: move() must record cells it actually passes through
    # as known/open — successfully walking a cell proves it's open just as
    # much as sensing it, and the LLM's map shouldn't show it as "?" after.
    maze = Maze(width=5, height=1, start=(0, 0), end=(4, 0))
    agent, robot = make_agent(maze, [("move", {"direction": "right", "distance": 3})])

    asyncio.run(agent.run_step())

    assert robot.known_cells[(1, 0)] == "open"
    assert robot.known_cells[(2, 0)] == "open"
    assert robot.known_cells[(3, 0)] == "open"
    memory = agent._build_memory_message()
    y_row = next(line for line in memory.splitlines() if line.startswith("y=0"))
    assert "?" not in y_row.split()[1:4]  # x=1,2,3 all known open now, not unexplored


def test_move_stops_early_at_wall_and_reports_partial_distance():
    maze = Maze(width=5, height=1, walls={(2, 0)}, start=(0, 0), end=(4, 0))
    agent, robot = make_agent(maze, [("move", {"direction": "right", "distance": 4})])

    events = tool_events(asyncio.run(agent.run_step()))

    assert events[0]["success"] is True
    assert events[0]["distance_requested"] == 4
    assert events[0]["distance_moved"] == 1
    assert events[0]["position_after"] == [1, 0]
    assert robot.position == (1, 0)


def test_move_defaults_to_distance_one():
    maze = Maze(width=2, height=1, start=(0, 0), end=(1, 0))
    agent, robot = make_agent(maze, [("move", {"direction": "right"})])

    events = tool_events(run_until_done(agent))

    assert events[0]["distance_requested"] == 1
    assert events[0]["distance_moved"] == 1
    assert robot.position == maze.end


def test_blocked_move_does_not_advance_robot():
    maze = Maze(width=2, height=1, start=(0, 0), end=(1, 0))
    agent, robot = make_agent(maze, [("move", {"direction": "left"})])  # out of bounds from (0, 0)

    events = tool_events(asyncio.run(agent.run_step()))

    assert events[0]["type"] == "move"
    assert events[0]["success"] is False
    assert events[0]["distance_moved"] == 0
    assert robot.position == (0, 0)
    assert robot.history == []


def test_messages_accumulate_tool_round_trip():
    maze = Maze(width=2, height=1, start=(0, 0), end=(1, 0))
    agent, _ = make_agent(maze, [("sense_surroundings", {})])

    asyncio.run(agent.run_step())

    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert agent.messages[2]["tool_calls"][0]["function"]["name"] == "sense_surroundings"


def grid_symbols(memory: str) -> set[str]:
    """Extract just the rendered cell symbols from a memory message's map,
    ignoring the legend text (which always literally contains "'#'=wall"
    etc. regardless of what's actually been discovered)."""
    symbols: set[str] = set()
    for line in memory.splitlines():
        if line.startswith("y="):
            symbols.update(line.split()[1:])
    return symbols


def test_memory_message_shows_full_maze_extent_and_end_always():
    maze = Maze(width=3, height=1, walls={(1, 0)}, start=(0, 0), end=(2, 0))
    agent, robot = make_agent(maze, [("sense_surroundings", {})])

    # Even before any sensing, the map spans the full maze width/height and
    # marks the end cell — matching what the human-facing board already
    # shows (full grid + end flag, regardless of fog-of-war). Only wall/open
    # status stays hidden until actually sensed.
    initial_memory = agent.messages[1]["content"]
    assert "Current position: (0, 0)" in initial_memory
    assert "full maze size: 3x1" in initial_memory
    assert grid_symbols(initial_memory) == {"R", "?", "E"}

    asyncio.run(agent.run_step())

    # messages[1] only refreshes at the *start* of run_step (for the next
    # call) — query the builder directly to see the effect of this turn's sense.
    updated_memory = agent._build_memory_message()
    assert grid_symbols(updated_memory) == {"R", "#", "E"}  # wall now known; end still shown
    assert "Path so far: (0,0)" in updated_memory


def test_memory_message_covers_full_grid_for_non_square_maze():
    maze = Maze(width=4, height=6, start=(0, 0), end=(3, 5))
    agent, robot = make_agent(maze, [("move", {"direction": "right", "distance": 1})])

    asyncio.run(agent.run_step())

    memory = agent._build_memory_message()
    assert "full maze size: 4x6" in memory
    y_rows = [line for line in memory.splitlines() if line.startswith("y=")]
    assert len(y_rows) == 6  # every row, not just the explored bounding box
    assert y_rows[-1].startswith("y=5")
    assert len(y_rows[0].split()) == 5  # "y=0" token plus all 4 column symbols


def test_memory_message_tracks_visited_path():
    maze = Maze(width=3, height=1, start=(0, 0), end=(2, 0))
    agent, robot = make_agent(maze, [("move", {"direction": "right", "distance": 2})])

    run_until_done(agent)

    # self.messages[1] only refreshes at the *start* of each run_step (so the
    # next call has fresh data) — since the maze was solved within the final
    # call, query the builder directly to see state as of right now.
    memory = agent._build_memory_message()
    assert "Path so far: (0,0) -> (2,0)" in memory
    assert "Current position: (2, 0)" in memory
