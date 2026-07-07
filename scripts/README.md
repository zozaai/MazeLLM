# Dataset generation

Generate supervised-finetuning (SFT) datasets that teach an LLM to solve the
maze reliably, using the **D\* Lite** solver as the expert. Runs offline (no API
key, no browser) and writes `.jsonl`.

## Layout: generic core vs per-model

The maze/expert engine is model-agnostic and lives here; anything specific to a
target model (its chat/tool-call format, its finetuning recipe) lives under
`scripts/<model>/`, and its outputs under `data/<model>/`.

```
scripts/
  expert_client.py          # generic: D* Lite expert as a fake LLM client; records neutral decisions
  dataset_gen.py            # generic: run the sweep, dedup, write JSONL via a model formatter
  LFM2.5-1.2B-Instruct/     # model-specific: LFM2 formatter + entrypoint + finetuning notes
    lfm2_format.py
    generate.py
    README.md
data/
  LFM2.5-1.2B-Instruct/     # generated dataset (gitignored)
    maze_sft.jsonl
    tool_schemas.json
```

## How it works

`dataset_gen.py` drives the real `MazeSolvingAgent`, swapping the LLM for
`ExpertLLMClient`, which decides each tool call from a D\* Lite planner under
fog-of-war. The agent runs unchanged, so each recorded turn is a faithful
`(state → expert action)` pair.

The expert takes **one action per turn, alternating sense → move → sense …**, so
each *move* example is conditioned on the map *after* the preceding sense — it
teaches "sense the unknown before you step into it," the habit the base model
lacks. Each turn is captured as a **neutral** record (system prompt, memory grid,
tool name + args, rationale); the per-model formatter renders it into that
model's chat format. Exact-duplicate decisions are deduped (early-turn states
repeat across mazes because the start cell is fixed).

Each `.jsonl` line is one decision:

```json
{"messages": [{system}, {user: memory grid}, {assistant: action}],
 "meta": {"size":[8,8],"density":0.25,"seed":42,"turn":3,"action":"move","pos":[2,5]}}
```

## Run

```bash
python scripts/LFM2.5-1.2B-Instruct/generate.py --sizes 5,8,10 --per-size 500
```

See `scripts/LFM2.5-1.2B-Instruct/README.md` for that model's format and
finetuning recipe.

## Add another target model

1. Create `scripts/<model>/lfm2_format.py`-equivalent exposing `format_example(turn,
   include_rationale)` (neutral turn → `{"messages": [...], "meta": ...}`) and
   `tool_schemas()` in that model's expected shape.
2. Create `scripts/<model>/generate.py` that imports `dataset_gen`
   (`run_sweep`, `dedup`, `write_dataset`) and your formatter, writing to
   `data/<model>/`.

The generic core (`expert_client.py`, `dataset_gen.py`) stays untouched.

## Next steps (not yet built)

- **Eval harness** — score a finetuned model vs D\* Lite on unseen mazes:
  success rate, path-length ratio, blocked-move attempts, revisited cells.
- **DAgger** — run the finetuned model, query D\* Lite at the states it actually
  reaches, add those corrections. The real fix for looping/repeats.
