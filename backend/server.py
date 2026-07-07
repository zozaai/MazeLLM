"""FastAPI app: generates mazes and runs the maze-solving agent over WebSocket.

POST /maze/generate:
  -> {"width": 4, "height": 4, "wall_density": 0.25, "seed": null}
  <- {"width": ..., "height": ..., "walls": [[x,y], ...], "start": [x,y], "end": [x,y]}
     The full layout is returned (not fog-of-war'd) so the UI can render the
     true wall layout up front; the robot itself still only learns cells via
     sense_surroundings during a solve.

Client protocol on /ws/solve:
  -> client sends {"maze": <maze dict from /maze/generate, or a filename
     string under mazes/>, "max_steps": 100, "solver": "llm"} immediately
     after connecting; defaults to sample_maze.json and the LLM solver if
     nothing is sent. "solver" selects the strategy: "llm" (tool-use agent) or
     a fog-of-war classical solver in dstar_lite.FOG_SOLVERS (currently
     "dstar_lite") — both work under the same sensed-only view of the maze.
  <- server sends one {"type": "init", width, height, walls, start, end}
     event, then per turn a {"type": "memory", "content": "..."} event
     (what the solver "knows" this turn) followed by one event per action it
     took ({"type": "sense", ...} / {"type": "move", ...}), then a final
     {"type": "done", ...} (or {"type": "error", ...} on failure). Baseline
     solvers emit only "memory" (once) and "move" events.
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent.dstar_lite import FOG_SOLVERS, DStarLiteAgent
from .agent.llm_agent import MazeSolvingAgent
from .agent.llm_client import LLMClient
from .agent.tools import TOOL_SCHEMAS
from .config import load_settings
from .maze.generator import generate_random_maze
from .maze.maze import Maze
from .maze.robot import Robot

settings = load_settings()
app = FastAPI(title="mazellm")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAZES_DIR = Path(__file__).parent.parent / "mazes"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/llm/info")
def llm_info() -> dict:
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
    }


@app.post("/llm/check")
async def llm_check() -> dict:
    """Send one minimal tool-enabled request to the configured LLM, so
    connectivity and tool-calling support can both be verified from the UI
    without running a full maze solve."""
    llm_client = LLMClient(settings)
    try:
        response = await llm_client.chat(
            [{"role": "user", "content": "Reply with a short greeting."}],
            TOOL_SCHEMAS,
        )
        content = response.choices[0].message.content
        return {"ok": True, "message": content or "(model responded with a tool call, no text)"}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


class GenerateMazeRequest(BaseModel):
    width: int = Field(default=4, ge=2, le=30)
    height: int = Field(default=4, ge=2, le=30)
    wall_density: float = Field(default=0.25, ge=0, le=0.8)
    seed: int | None = None


@app.post("/maze/generate")
def generate_maze(req: GenerateMazeRequest) -> dict:
    maze = generate_random_maze(req.width, req.height, req.wall_density, req.seed)
    return maze.to_dict()


def _load_maze(maze_param) -> Maze:
    if isinstance(maze_param, dict):
        return Maze.from_dict(maze_param)
    return Maze.load_json(MAZES_DIR / maze_param)


@app.websocket("/ws/solve")
async def solve(websocket: WebSocket) -> None:
    await websocket.accept()

    params: dict = {}
    try:
        params = await websocket.receive_json()
    except Exception:
        pass  # no config sent — fall back to defaults

    max_steps = int(params.get("max_steps", 100))
    solver = str(params.get("solver", "llm")).lower()

    try:
        maze = _load_maze(params.get("maze", "sample_maze.json"))
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        await websocket.send_json({"type": "error", "message": f"invalid maze: {exc}"})
        await websocket.close()
        return

    robot = Robot(position=maze.start)
    if solver in FOG_SOLVERS:  # classical solver under the LLM's fog-of-war limitation
        agent = DStarLiteAgent(maze, robot, solver, max_steps=max_steps)
    else:  # default: LLM tool-use agent (also covers an unknown/omitted solver)
        llm_client = LLMClient(settings)
        agent = MazeSolvingAgent(maze, robot, llm_client, max_steps=max_steps)

    await websocket.send_json({"type": "init", **maze.to_dict()})

    max_turns = max_steps * 5  # backstop against a solver stalling without ever completing a move
    turns = 0
    try:
        while not agent.is_done() and turns < max_turns:
            for event in await agent.run_step():
                await websocket.send_json(event)
            turns += 1

        await websocket.send_json(
            {
                "type": "done",
                "success": robot.position == maze.end,
                "steps": len(robot.history),
                "turns": turns,
            }
        )
    except WebSocketDisconnect:
        return
    except Exception as exc:  # LLM/API failures, malformed tool args, etc.
        await websocket.send_json({"type": "error", "message": str(exc)})
