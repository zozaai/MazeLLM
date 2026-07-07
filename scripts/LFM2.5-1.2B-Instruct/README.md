# Dataset + finetuning: `unsloth/LFM2.5-1.2B-Instruct`

Model-specific pieces for LFM2.5. The generic maze/expert engine lives in
`scripts/` (`dataset_gen.py`, `expert_client.py`); this folder only adds LFM2's
output format.

- `lfm2_format.py` — turns neutral expert decisions into LFM2's native chat
  format and the flat tool schema.
- `generate.py` — entrypoint; writes to `data/LFM2.5-1.2B-Instruct/`.

## Generate

```bash
python scripts/LFM2.5-1.2B-Instruct/generate.py --sizes 5,8,10 --per-size 500
# tiny sanity run:
python scripts/LFM2.5-1.2B-Instruct/generate.py --sizes 6 --per-size 3 --preview 2
```

Outputs `data/LFM2.5-1.2B-Instruct/maze_sft.jsonl` and `tool_schemas.json`
(flat). `data/` is gitignored.

## Why this format (verified)

Checked against LFM2.5's `chat_template.jinja` and LiquidAI's tool-use docs:

- The template renders **only `message["content"]`** — it never reads
  `tool_calls`. So tool calls live in the assistant *content* as
  `<|tool_call_start|>[name(args)]<|tool_call_end|>`.
- Tools go in the **system** turn as `List of tools: [...]` using a **flat**
  schema (`{name, description, parameters}`), not OpenAI's `{type, function}`.

A rendered training row:

```
<|im_start|>system
You are a robot placed inside a maze. …
List of tools: [{"name": "sense_surroundings", …}, {"name": "move", …}]<|im_end|>
<|im_start|>user
Current position: (0, 0) … Known map …<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[move(direction="down", distance=4)]<|tool_call_end|>
The exit is to the south-east. …<|im_end|>
```

## Finetune (Unsloth)

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
import json

model, tok = FastLanguageModel.from_pretrained("unsloth/LFM2.5-1.2B-Instruct", load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

tools = json.load(open("data/LFM2.5-1.2B-Instruct/tool_schemas.json"))   # flat schema
ds = load_dataset("json", data_files="data/LFM2.5-1.2B-Instruct/maze_sft.jsonl", split="train")

def fmt(ex):
    # assistant content already holds <|tool_call_start|>…; the template renders it
    # verbatim and adds "List of tools:" to the system turn.
    return {"text": tok.apply_chat_template(ex["messages"], tools=tools, tokenize=False)}

ds = ds.map(fmt)
SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text").train()
```
