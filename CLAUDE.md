# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MazeLLM: an LLM tool-use agent drives a robot through a maze it can only
sense locally, benchmarked against a classical **D\* Lite** solver running
under the *identical* fog-of-war limitation (it only knows cells it has
sensed) — a fair, apples-to-apples comparison. A FastAPI backend streams the
solve step-by-step over a WebSocket to a vanilla JS/canvas frontend.

## Commands

```bash
cp .env.example .env                        # fill in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
pip install -e ".[dev]"
uvicorn backend.server:app --reload         # run the backend (http://localhost:8000)
pytest                                       # run tests (offline, no API key needed)
pytest tests/test_llm_agent.py::test_name -v # run a single test
pytest -m integration                        # run tests that call a real LLM (excluded by default)
```

Open `frontend/index.html` directly in a browser (or serve it statically); it
talks to `http://localhost:8000` by default, overridable via `?http=...&ws=...`
query params.

Tests default to `-m "not integration"` (set in `pyproject.toml`), so `pytest`
alone never needs network/API access — the LLM agent tests drive it with a
scripted fake client (see `ScriptedLLMClient` in `tests/test_llm_agent.py`).
CI (`.github/workflows/ci.yml`) runs the same unit tests with coverage on PRs.

## Architecture

- **`backend/maze/`** — maze representation (`maze.py`), JSON load/save,
  random generation (`generator.py`), and robot state (`robot.py`: position +
  `known_cells` fog-of-war memory, move history).
- **`backend/agent/`** — the two solvers plus shared plumbing:
  - `llm_agent.py` — `MazeSolvingAgent`, the tool-use loop. `build_memory_message`
    renders the full-size grid (walls `#`, open `.`, unexplored `?`, always
    marking start/end) plus path-so-far; it's refreshed in place each turn
    (not appended) so context stays bounded. Shared by both solvers so their
    transcripts are format-identical.
  - `llm_client.py` — wraps any OpenAI-compatible `chat.completions` endpoint;
    swapping providers is a `.env` change, not a code change.
  - `dstar_lite.py` — `DStarLiteAgent`/`FOG_SOLVERS`: a D\* Lite (Koenig &
    Likhachev) solver that plans backward from the goal over the *known* map
    (unsensed cells treated as optimistically open) and incrementally repairs
    the plan when sensing reveals a wall, rather than replanning from scratch.
    It drives the real robot through the same `sense_surroundings`/`move`
    tools as the LLM agent, so it emits byte-identical `memory`/`sense`/`move`
    events — its transcripts double as an expert source for finetuning.
  - `tools.py` — the only two tools the LLM/solvers get: `sense_surroundings`
    (scan each direction until a wall/boundary, recording every cell passed)
    and `move` (straight-line move, stopping early on a wall). Includes a
    regex fallback (`parse_text_tool_call`) for servers that return the tool
    call as plain text in `content` instead of structured `tool_calls` (e.g.
    LFM2 via Unsloth).
  - `prompts.py` — the system prompt.
- **`backend/server.py`** — FastAPI app. `POST /maze/generate` returns the
  full true layout (not fog-of-war'd) so the UI can render it; the robot only
  *learns* cells via `sense_surroundings` during a solve. `GET /llm/info` and
  `POST /llm/check` report/smoke-test the configured provider. The solve runs
  over `/ws/solve`: client sends `{"maze": ..., "max_steps": 100, "solver":
  "llm"|"dstar_lite"}`, server streams `init` → per turn (`memory` then one
  `sense`/`move` per action) → `done`/`error`.
- **`frontend/`** — vanilla HTML/CSS/JS (no build step). Canvas rendering,
  fog-of-war reveal animation, live decision log, solver picker.
- **`mazes/sample_maze.json`** — the default maze when the client doesn't
  send its own generated layout.
- **`scripts/`** — offline SFT dataset generation from D\* Lite expert
  trajectories (see `scripts/README.md`): generic core (`expert_client.py`
  fakes an LLM client that decides via D\* Lite; `dataset_gen.py` runs the
  maze sweep and writes JSONL; `eval_core.py` is the D\* Lite reference +
  metrics for eval) vs. per-target-model formatter/entrypoint under
  `scripts/<model>/`, writing to `data/<model>/` (gitignored). Add a new
  target model by dropping in `scripts/<model>/<model>_format.py` +
  `generate.py` without touching the generic core.

## Switching LLM providers

Set in `.env` only — `LLM_BASE_URL` / `LLM_MODEL` (OpenAI, local vLLM,
Ollama, or a Hugging Face Inference Endpoint all work since they're
OpenAI-compatible `chat.completions` endpoints).
