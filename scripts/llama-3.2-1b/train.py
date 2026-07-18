#!/usr/bin/env python3
"""Finetune `unsloth/Llama-3.2-1B-Instruct` on the maze SFT dataset with
Unsloth LoRA, then export a GGUF + Ollama Modelfile.

Identical recipe to `scripts/llama-3.2-3b/train.py` — the 1B and 3B Instruct
models share the same architecture family and chat template (see
`llama_format.py`), just a smaller `--base-model` and different output paths.

Requires a CUDA GPU with `unsloth` installed (Colab, or a local/cloud box with
a GPU) — this script cannot run in a CPU-only dev environment; imports are
deferred into main() so `--help` and argument parsing still work anywhere.

Recipe adapted from the Production-Ready-Instruction-Finetuning-of-Meta-
Llama-3.2-3B-Instruct-Project reference (LoRA r=16/alpha=16/dropout=0,
`train_on_responses_only` response-only loss masking, adamw_8bit + linear
scheduler) — with two fixes over that reference:
  - `lora_dropout` is actually forwarded to `get_peft_model`. The reference's
    `src/finetuning/applying_lora.py` reads it from YAML but never passes it
    through, so it silently fell back to the library default despite the
    config saying 0.
  - Export goes straight to GGUF + an Ollama Modelfile via Unsloth's
    `save_pretrained_gguf` (merge+quantize+convert in one call), instead of a
    manual `PeftModel.from_pretrained(...).merge_and_upload()` — that method
    name is a typo for `merge_and_unload` in the reference's
    `merge_base_and_finetuned_model.py` and would raise `AttributeError` as
    written.

The response-only header markers (`<|start_header_id|>`/`<|end_header_id|>`)
were verified against Llama-3.2-3B-Instruct's real `chat_template.jinja`, not
assumed from the Llama-3.1 reference project.

Usage:
    python scripts/llama-3.2-1b/train.py \\
        --data data/llama-3.2-1b/maze_sft.jsonl \\
        --tools data/llama-3.2-1b/tool_schemas.json \\
        --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--data", default="data/llama-3.2-1b/maze_sft.jsonl", help="JSONL from generate.py")
    ap.add_argument("--tools", default="data/llama-3.2-1b/tool_schemas.json")
    ap.add_argument("--out-dir", default="scripts/llama-3.2-1b/outputs/maze-lora",
                     help="where the LoRA adapter + tokenizer are saved")
    ap.add_argument("--gguf-out", default="scripts/llama-3.2-1b/outputs/maze-gguf",
                     help="where the merged+quantized GGUF + Modelfile are written")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--load-in-4bit", action="store_true",
                     help="quantize the frozen base model to 4-bit (QLoRA-style). Saves VRAM at a small "
                          "speed cost from on-the-fly dequantization every matmul — worth it only when "
                          "the model wouldn't otherwise fit. Off by default: a 1B model leaves plenty of "
                          "headroom on most training GPUs, where full-precision (bf16) LoRA trains faster.")
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=4, help="per_device_train_batch_size")
    ap.add_argument("--grad-accum", type=int, default=4, help="gradient_accumulation_steps")
    ap.add_argument("--epochs", type=float, default=3.0,
                     help="num_train_epochs — scale to dataset size, not a fixed step count "
                          "(a few thousand steps at minimum; the 60/300-step examples in the "
                          "reference repos were tuned for their own much larger or tiny datasets)")
    ap.add_argument("--learning-rate", type=float, default=1.5e-4)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--quant", default="q4_k_m", help="GGUF quantization method for the Ollama export")
    args = ap.parse_args()

    # unsloth must be imported before trl/transformers/peft — it patches them
    # for its speed/memory optimizations, which don't apply if those libraries
    # are imported (and initialized) first.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=args.seed,
        use_rslora=False,
    )

    tools = json.load(open(args.tools))

    # Parse the JSONL in plain Python rather than via `datasets.load_dataset`.
    # That loader infers ONE fixed Arrow struct schema for
    # `tool_calls[0].function.arguments` across the whole file — and since
    # `sense_surroundings` calls have `arguments: {}` while `move` calls have
    # `arguments: {direction, distance}`, it silently backfills every
    # sense_surroundings example with `direction: null, distance: null` before
    # `apply_chat_template` ever sees it, corrupting the rendered tool call
    # (confirmed: `datasets.load_dataset("json", ...)` alone turns `{}` into
    # `{"direction": None, "distance": None}`). Rendering "text" straight from
    # the raw parsed JSON, and only ever wrapping the finished strings in a
    # Dataset, sidesteps this — a single string column can't be miscoerced.
    # It also means "messages" never reaches trl, so its own conversational
    # auto-detection (which re-derives input_ids from "messages" without our
    # `tools`, potentially masking the whole dataset to -100 in
    # `train_on_responses_only`) never comes into play either.
    texts = []
    with open(args.data) as f:
        for line in f:
            example = json.loads(line)
            texts.append(tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False))
    dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            group_by_length=True,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=False,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.02,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.out_dir,
        ),
    )

    # Mask loss to the assistant's tool-call tokens only — the model is only
    # graded on producing the right call given the system+memory context, not
    # on "predicting" that context itself.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    print("Gradient checkpointing:", model.is_gradient_checkpointing)
    print("Device map:", getattr(model, "hf_device_map", None))

    train_result = trainer.train()

    print("\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value}")

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved LoRA adapter to {args.out_dir}")

    # Merge + quantize + convert in one call — Ollama needs a GGUF, not the
    # adapter-only PEFT save above.
    model.save_pretrained_gguf(args.gguf_out, tokenizer, quantization_method=args.quant)
    print(f"Wrote GGUF to {args.gguf_out}")

    # Unsloth's GGUF export usually writes its own Modelfile (with a TEMPLATE
    # derived from the tokenizer's chat template metadata) — only add a
    # minimal fallback if it didn't.
    modelfile_path = os.path.join(args.gguf_out, "Modelfile")
    if not os.path.exists(modelfile_path):
        gguf_name = next(f for f in os.listdir(args.gguf_out) if f.endswith(".gguf"))
        with open(modelfile_path, "w") as f:
            f.write(f"FROM ./{gguf_name}\n")
        print(f"Wrote fallback {modelfile_path}")

    print(
        "\nServe with Ollama:\n"
        f"  cd {args.gguf_out}\n"
        "  ollama create llama-3.2-1b-maze -f Modelfile\n"
        "  ollama run llama-3.2-1b-maze\n"
        "Then point .env's LLM_MODEL=llama-3.2-1b-maze at "
        "LLM_BASE_URL=http://localhost:11434/v1 (or run eval.py against both "
        "the base and fine-tuned Ollama tags to compare)."
    )


if __name__ == "__main__":
    main()
