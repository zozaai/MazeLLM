# Dataset + finetuning: `unsloth/Qwen3.5-4B`

Model-specific pieces for Qwen3.5. The generic maze/expert engine lives in
`scripts/` (`dataset_gen.py`, `expert_client.py`); this folder adds Qwen's
output format.

- `qwen_format.py` — neutral expert decision → Qwen chat example + tool schema.
- `generate.py` — entrypoint; writes to `data/qwen3.5/`.

## Generate

```bash
python scripts/qwen3.5/generate.py --sizes 5,6,7,8,9,10 --per-size 1000
# tiny sanity run:
python scripts/qwen3.5/generate.py --sizes 6 --per-size 3 --preview 2
```

Outputs `data/qwen3.5/maze_sft.jsonl` and `tool_schemas.json`. `data/` is
gitignored. Duplicates are kept by default (natural frequency is best for
behavior cloning); use `--dup-cap N` only to trim extreme repeats.

## Format (verified against Qwen3.5-4B's chat template)

Unlike LFM2, Qwen's template renders **structured** `tool_calls`, so each example
keeps the OpenAI-style call:

```json
{"messages": [
  {"role": "system", "content": "You are a robot … (prose tool description)"},
  {"role": "user", "content": "Current position … Known map …"},
  {"role": "assistant", "content": "The exit is south-east …",
   "tool_calls": [{"id": "call_1", "type": "function",
                   "function": {"name": "move", "arguments": {"direction": "down", "distance": 4}}}]}]}
```

Gotchas the format handles for you:
- `arguments` is a **dict**, not a JSON string (a string makes Qwen's template
  drop the parameters).
- Tools are the OpenAI `{type, function}` schema (`data/qwen3.5/tool_schemas.json`).
- Qwen is a thinking model: the template auto-adds an (empty) `<think></think>`
  and renders the assistant content as reasoning *before* the `<tool_call>`.

A rendered training row (Qwen XML tool-call format):

```
<|im_start|>assistant
<think>

</think>

The exit is south-east …

<tool_call>
<function=move>
<parameter=direction>
down
</parameter>
<parameter=distance>
4
</parameter>
</function>
</tool_call><|im_end|>
```

## Finetune (Unsloth)

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
import json

model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen3.5-4B", load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

tools = json.load(open("data/qwen3.5/tool_schemas.json"))
ds = load_dataset("json", data_files="data/qwen3.5/maze_sft.jsonl", split="train")

def fmt(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tools=tools, tokenize=False)}

ds = ds.map(fmt)
SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text").train()
```

Keep **Assistant completions only** on and use **≥1 epoch / a few thousand
steps** (not 30) — see the top-level `scripts/README.md` notes.

## Serving (Ollama) & the app

You run `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` via Ollama. Qwen emits tool calls
as the `<tool_call><function=…>` XML above. **Open question to check at eval
time:** whether Ollama parses that into OpenAI `tool_calls` (so the web app reads
`message.tool_calls` directly) or returns it as text. If it comes back as text,
the app needs a Qwen-XML parser (the existing `parse_text_tool_call` only handles
the `name(args)` form). We'll confirm which when you first Solve in the browser.
