# Dataset + finetuning: `unsloth/Llama-3.2-1B-Instruct`

Self-contained pipeline for this one target model — nothing shared with other
model dirs (same convention as `scripts/llama-3.2-3b/`; code is duplicated,
not imported, per that dir's README). Generates a maze-solving SFT dataset
from a D\* Lite expert, finetunes Llama-3.2-1B-Instruct with Unsloth LoRA,
exports a GGUF for Ollama, and evaluates base vs fine-tuned against the D\*
Lite reference.

```
scripts/llama-3.2-1b/
  expert_driver.py   # D* Lite fake-LLM expert + maze sweep + dup capping + JSONL writer
  llama_format.py    # neutral expert decision -> Llama tool-call chat format
  generate.py         # CLI entrypoint -> data/llama-3.2-1b/{maze_sft.jsonl,tool_schemas.json}
  train.py            # Unsloth LoRA finetune -> LoRA adapter + GGUF + Ollama Modelfile
  eval.py             # base vs fine-tuned (both served via Ollama) vs D* Lite reference
data/llama-3.2-1b/    # generated dataset (gitignored)
  maze_sft.jsonl
  tool_schemas.json
```

Identical in every respect to `scripts/llama-3.2-3b/` except the base model
and output paths: Llama-3.2-1B-Instruct and -3B-Instruct ship **byte-identical
`chat_template.jinja`** (diffed both tokenizers' `chat_template` directly, not
assumed), so the expert driver, dataset format, and eval harness all carry
over unchanged. See that dir's README for the full write-up of the dataset
design and format gotchas — this one only calls out what differs.

## Why 1B in addition to 3B

Same dataset-generation and training recipe, smaller model: faster to
iterate on (fewer parameters to update, smaller checkpoints, lower VRAM
pressure), useful as a cheaper baseline to compare against 3B — does the
extra capacity actually buy a better fog-of-war maze solver, or does 1B get
you most of the way there?

## Generate

```bash
python scripts/llama-3.2-1b/generate.py --sizes 5,6,8,10 --per-size 1000
# tiny sanity run:
python scripts/llama-3.2-1b/generate.py --sizes 6 --per-size 3 --preview 1
```

**Reserve at least one maze size (or a seed range far outside the training
sweep) for `eval.py`** — e.g. train on `5,6,8,10` and evaluate on `7,9`. The
model must never see the eval mazes. If you're generating both the 1B and 3B
datasets, keep the same held-out split for both so the two models' eval
numbers are directly comparable.

Writes `data/llama-3.2-1b/maze_sft.jsonl` and `tool_schemas.json` (`data/` is
gitignored).

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
python scripts/llama-3.2-1b/train.py \
    --data data/llama-3.2-1b/maze_sft.jsonl \
    --tools data/llama-3.2-1b/tool_schemas.json \
    --epochs 3
```

Same defaults as the 3B recipe (LoRA r=16/alpha=16/dropout=0,
`train_on_responses_only` masking, `adamw_8bit` + linear scheduler, full bf16
precision by default — pass `--load-in-4bit` only if you're VRAM-constrained,
which is even less likely at 1B than at 3B).

## Serving (Ollama)

`train.py` exports a GGUF + Modelfile to `--gguf-out` and prints the exact
`ollama create`/`ollama run` commands. To compare base vs fine-tuned:

```bash
ollama pull hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M   # base, for comparison
cd scripts/llama-3.2-1b/outputs/maze-gguf
ollama create llama-3.2-1b-maze -f Modelfile
```

Then point `backend`'s `.env` at either tag (`LLM_BASE_URL=http://localhost:11434/v1`,
`LLM_MODEL=<tag>`) — no backend code changes needed either way, since
`LLMClient` already talks to any OpenAI-compatible endpoint.

## Evaluate

```bash
python scripts/llama-3.2-1b/eval.py \
    --models hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M,llama-3.2-1b-maze \
    --sizes 7,9 --per-size 20
```

Drives the *real* `MazeSolvingAgent` + `LLMClient` against a live Ollama
server for each model tag (not a simulation), over maze sizes/seeds you
choose — use ones outside `generate.py`'s training sweep. Reports, per model:
success rate, path length vs the D\* Lite optimum (1.00x = matches it),
blocked-move attempts, revisited cells, and invalid/unparsed tool-call turns.

To compare directly against the 3B model, run this same maze sizes/seeds
sweep against `llama-3.2-3b-maze` too (`scripts/llama-3.2-3b/eval.py` or this
same `eval.py` — the harness is identical) and diff the summaries.
