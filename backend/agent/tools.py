"""Tool schemas exposed to the LLM, and their execution against the maze/robot.

Only two tools exist: the robot can scan how far it could travel in each
direction, or move up to that many cells in a straight line. It never gets
direct access to the maze grid.
"""
from __future__ import annotations

import re

from ..maze.maze import Cell, Maze
from ..maze.robot import DIRECTIONS, Robot

# Fallback parser for models/servers that return the tool call as text in the
# assistant `content` instead of structured `tool_calls` — e.g. LFM2 served
# through Unsloth, which emits `<|tool_call_start|>[move(direction="down",
# distance=4)]<|tool_call_end|>` and cannot map it to OpenAI tool_calls. Anchored
# to the known tool names so it never misfires on prose like "at (0,0)".
_TEXT_CALL_RE = re.compile(r"\b(sense_surroundings|move)\s*\(([^)]*)\)")
_TEXT_ARG_RE = re.compile(r"""(\w+)\s*=\s*("[^"]*"|'[^']*'|[^,]+)""")


def parse_text_tool_call(content: str | None) -> tuple[str, dict] | None:
    """Extract (name, args) for the first tool call embedded in free text, or None."""
    if not content:
        return None
    match = _TEXT_CALL_RE.search(content)
    if not match:
        return None
    name, argstr = match.group(1), match.group(2).strip()
    args: dict = {}
    for key, raw in _TEXT_ARG_RE.findall(argstr):
        raw = raw.strip()
        if raw[:1] in "\"'":
            args[key] = raw[1:-1]
        else:
            try:
                args[key] = int(raw)
            except ValueError:
                args[key] = raw
    return name, args

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "sense_surroundings",
            "description": (
                "Check how far the robot could move in each direction "
                "(up/down/left/right) before hitting a wall or the maze "
                "boundary. For each direction, returns 'distance' (number of "
                "open cells it could move through — 0 means immediately "
                "blocked), 'blocked_by' ('wall' or 'boundary'), and 'end_at' "
                "(distance to the end cell if it lies within that stretch, "
                "otherwise null)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": (
                "Move the robot in a straight line in the given direction, "
                "up to 'distance' cells (default 1). It moves as far as "
                "possible toward the requested distance, stopping early if "
                "it hits a wall or the boundary; the result reports how far "
                "it actually moved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": list(DIRECTIONS.keys()),
                    },
                    "distance": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                        "description": "Number of cells to attempt to move.",
                    },
                },
                "required": ["direction"],
            },
        },
    },
]


def sense_surroundings(maze: Maze, robot: Robot) -> dict[str, dict]:
    """Scan each direction from the robot's position until a wall or the
    boundary stops it, recording every cell passed through along the way."""
    result: dict[str, dict] = {}
    for name, (dx, dy) in DIRECTIONS.items():
        distance = 0
        end_at = None
        cell = robot.position
        blocked_by = "boundary"
        while True:
            next_cell = (cell[0] + dx, cell[1] + dy)
            if not maze.in_bounds(next_cell):
                blocked_by = "boundary"
                break
            if maze.is_wall(next_cell):
                robot.known_cells[next_cell] = "wall"
                blocked_by = "wall"
                break
            distance += 1
            if next_cell == maze.end:
                robot.known_cells[next_cell] = "end"
                if end_at is None:
                    end_at = distance
            else:
                robot.known_cells[next_cell] = "open"
            cell = next_cell
        result[name] = {"distance": distance, "blocked_by": blocked_by, "end_at": end_at}
    return result


def move(maze: Maze, robot: Robot, direction: str, distance: int = 1) -> tuple[bool, str, Cell, int]:
    """Attempt to move up to `distance` cells in the given direction.

    Returns (success, message, resulting_position, distance_moved). Moves as
    far as possible toward the requested distance, stopping early at a wall
    or the boundary. success is True as long as at least one cell was moved.

    Every cell actually passed through is recorded as known/open (or "end")
    in robot.known_cells — successfully walking through a cell proves it's
    open just as much as sensing it would, and the map shown to the LLM
    should reflect that instead of leaving traversed cells as "?".
    """
    if direction not in DIRECTIONS:
        return False, f"unknown direction: {direction}", robot.position, 0
    if distance < 1:
        return False, "distance must be at least 1", robot.position, 0

    dx, dy = DIRECTIONS[direction]
    position = robot.position
    moved = 0
    while moved < distance:
        next_cell = (position[0] + dx, position[1] + dy)
        if not maze.is_walkable(next_cell) and next_cell != maze.end:
            break
        position = next_cell
        moved += 1
        robot.known_cells[position] = "end" if position == maze.end else "open"

    if moved == 0:
        return False, f"blocked: cannot move {direction}", robot.position, 0
    if moved < distance:
        return True, f"moved {moved} of {distance} requested cells before hitting a wall", position, moved
    return True, "ok", position, moved
