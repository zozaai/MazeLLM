# Dataset + finetuning: `unsloth/Llama-3.2-3B-Instruct`

Self-contained pipeline for this one target model — nothing shared with other
model dirs (there aren't any others right now). Generates a maze-solving SFT
dataset from a D\* Lite expert, finetunes Llama-3.2-3B-Instruct with Unsloth
LoRA, exports a GGUF for Ollama, and evaluates base vs fine-tuned against the
D\* Lite reference.

```
scripts/llama-3.2-3b/
  expert_driver.py   # D* Lite fake-LLM expert + maze sweep + dup capping + JSONL writer
  llama_format.py    # neutral expert decision -> Llama tool-call chat format
  generate.py         # CLI entrypoint -> data/llama-3.2-3b/{maze_sft.jsonl,tool_schemas.json}
  train.py            # Unsloth LoRA finetune -> LoRA adapter + GGUF + Ollama Modelfile
  eval.py             # base vs fine-tuned (both served via Ollama) vs D* Lite reference
data/llama-3.2-3b/    # generated dataset (gitignored)
  maze_sft.jsonl
  tool_schemas.json
```

## How the dataset is built

`expert_driver.py` drives the real `MazeSolvingAgent` (the same class the
backend uses), swapping the LLM for `ExpertLLMClient`, which decides each tool
call from a D\* Lite planner under fog-of-war. The agent runs unchanged, so
each recorded turn is a faithful `(state → expert action)` pair — and because
the expert takes **one action per turn, alternating sense → move → sense →
…**, each *move* example is conditioned on the map *after* the preceding
sense, teaching "sense the unknown before you step into it" rather than
guessing from a map full of `?`.

Each neutral turn (system prompt, memory grid, tool name + args, rationale) is
rendered by `llama_format.py` into one JSONL line: one **decision**, not one
episode. Duplicates are **kept by default** — for behavior cloning, example
frequency should track how often a state is visited (every episode starts by
sensing at the start cell, so that decision is legitimately common; deduping
it away biases the model against sensing). Use `--dup-cap N` only to trim
extreme repeats.

## Generate

```bash
python scripts/llama-3.2-3b/generate.py --sizes 5,6,8,10 --per-size 1000
# tiny sanity run:
python scripts/llama-3.2-3b/generate.py --sizes 6 --per-size 3 --preview 1
```

**Reserve at least one maze size (or a seed range far outside the training
sweep) for `eval.py`** — e.g. train on `5,6,8,10` and evaluate on `7,9`. The
model must never see the eval mazes.

Writes `data/llama-3.2-3b/maze_sft.jsonl` and `tool_schemas.json` (`data/` is
gitignored).

## Format (verified against Llama-3.2-3B-Instruct's real `chat_template.jinja`)

This wasn't assumed from docs — the template was fetched from
`unsloth/Llama-3.2-3B-Instruct` and read directly. Three things it does that
are easy to get wrong by guessing:

1. **Tools render as-is.** Each tool is dumped with `tojson` verbatim, so the
   existing OpenAI-style `{"type": "function", "function": {...}}` schema
   already in `backend/agent/tools.py` (`data/llama-3.2-3b/tool_schemas.json`)
   works unmodified.
2. **`tool_calls` renders as single-line JSON**, `{"name": ..., "parameters":
   ...}`, reading `tool_calls[0].function.{name,arguments}` — `arguments`
   **must be a dict**, not a JSON string (a string renders as an escaped blob
   instead of an object; same gotcha as most tool-calling chat templates).
   The template also only supports **one** tool call per assistant turn
   (raises otherwise), which already matches this project's one-call-per-turn
   design.
3. **Assistant `content` is dropped whenever `tool_calls` is set.** Unlike
   Qwen, there's no rendered slot for a rationale next to the call — so
   `llama_format.py` sets `content: null` and keeps the rationale only in
   `meta.rationale`, for dataset inspection/debugging. `--no-rationale` on
   `generate.py` has no effect on the rendered prompt either way; it only
   toggles whether `meta.rationale` is populated.

A rendered training row (confirmed by actually loading the tokenizer and
calling `apply_chat_template`, not just reading the template source):

```
<|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call...
{"type": "function", "function": {"name": "move", ...}}
{"type": "function", "function": {"name": "sense_surroundings", ...}}

Current position: (0, 0)
Known map so far ...
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"name": "move", "parameters": {"direction": "down", "distance": 5}}<|eot_id|>
```

Note `tools_in_user_message` (the default) injects the tool listing into the
**first user message**, not the system message. Since each training example
is one independent turn (system + one memory-grid user message + one
assistant call), that first user message is always the memory grid — which
also matches the real serving loop: every later turn's tool result uses
`role: "tool"`, never `role: "user"`, so the memory message is *always* the
one and only "first user message" the template sees, turn after turn. Turn-
level (not full-episode) examples are the right shape for this template.

## Finetune (Unsloth)

Requires a CUDA GPU. Install the finetuning stack first — it's a separate
extra from the main app's dependencies since it pulls in torch/unsloth/trl:

```bash
pip install -e ".[finetune]"
```

If `unsloth` fails to install or import against your existing torch/CUDA
build, follow its official install matrix at
https://github.com/unslothai/unsloth#installation — the pinned torch/xformers
versions are CUDA-version-specific and change often enough that hardcoding a
command here would go stale.

```bash
python scripts/llama-3.2-3b/train.py \
    --data data/llama-3.2-3b/maze_sft.jsonl \
    --tools data/llama-3.2-3b/tool_schemas.json \
    --epochs 3
```

Base model loads in **full precision (bf16) by default**, not 4-bit — a 3B
model leaves plenty of VRAM headroom on most training GPUs, and skipping
quantization avoids the small but real per-step cost of dequantizing 4-bit
weights on every matmul. Pass `--load-in-4bit` to opt into QLoRA-style 4-bit
if you're VRAM-constrained; that trades some speed for memory, not the other
way around — quantization is a memory optimization, not a speed one.

`train.py`'s imports are deferred into `main()` so `--help` still works
without any of the above installed. Recipe adapted from the `Production-Ready-Instruction-Finetuning-
of-Meta-Llama-3.2-3B-Instruct-Project` reference (LoRA r=16/alpha=16/
dropout=0, `train_on_responses_only` masking, `adamw_8bit` + linear
scheduler), with bugs from that reference (and from targeting a much newer
`trl` than it was written against) fixed:

- `lora_dropout` is actually passed to `get_peft_model` (the reference reads
  it from YAML but never forwards it — silently ignored despite the config
  saying 0).
- Export goes straight to GGUF via Unsloth's `save_pretrained_gguf`
  (merge+quantize+convert in one call) instead of a manual
  `PeftModel(...).merge_and_upload()` — that method name is a typo for
  `merge_and_unload` in the reference and crashes as written.
- The dataset is stripped down to a single `text` column (`messages`/`meta`
  dropped) before being handed to `SFTTrainer`. Newer `trl` (0.19+) treats
  any dataset with a `messages` column as "conversational" and **silently
  re-derives `input_ids` straight from `messages` via its own internal
  `apply_chat_template` call** — bypassing the precomputed `text` field
  entirely, and critically without our `tools` (it only reads a *per-example*
  `tools` column, which doesn't exist). Left in place, this can make
  `train_on_responses_only`'s marker search find zero matches and mask the
  *entire* dataset to `-100` (`ZeroDivisionError: All labels ... are -100`).
  Dropping `messages` forces the `dataset_text_field="text"` path, which is
  the one actually verified against the real chat template (see
  `llama_format.py`). Uses `SFTConfig` + `processing_class=` (the current
  `trl` API) rather than the `TrainingArguments`/`tokenizer=` kwargs the
  reference project's older `trl` version accepted.

Scale `--epochs` to your dataset size, not a fixed step count — the reference
repos' `max_steps=60`/`max_steps=300` were tuned for datasets orders of
magnitude larger or smaller than a maze sweep will produce. Keep response-only
masking on regardless of scale (see `train_on_responses_only` above).

## Serving (Ollama)

`train.py` exports a GGUF + Modelfile to `--gguf-out` and prints the exact
`ollama create`/`ollama run` commands. To compare base vs fine-tuned:

```bash
ollama pull hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M   # base, for comparison
cd scripts/llama-3.2-3b/outputs/maze-gguf
ollama create llama-3.2-3b-maze -f Modelfile
```

Then point `backend`'s `.env` at either tag (`LLM_BASE_URL=http://localhost:11434/v1`,
`LLM_MODEL=<tag>`) — no backend code changes needed either way, since
`LLMClient` already talks to any OpenAI-compatible endpoint.

## Evaluate

```bash
python scripts/llama-3.2-3b/eval.py \
    --models hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M,llama-3.2-3b-maze \
    --sizes 7,9 --per-size 20
```

Drives the *real* `MazeSolvingAgent` + `LLMClient` against a live Ollama
server for each model tag (not a simulation), over maze sizes/seeds you
choose — use ones outside `generate.py`'s training sweep. Reports, per model:
success rate, path length vs the D\* Lite optimum (1.00x = matches it),
blocked-move attempts, revisited cells, and invalid/unparsed tool-call turns.
