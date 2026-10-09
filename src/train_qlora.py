"""QLoRA SFT fine-tuning with Unsloth + TRL.

Run on a GPU (Colab / Kaggle / RunPod / Modal / Lambda). The heavy imports
(torch, unsloth, trl, transformers) are performed inside ``main()`` so the rest of
the package stays import-safe for CI and unit tests.

Example:
    python -m src.train_qlora --config config/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from .prompts import build_messages


def load_config(path: str) -> Dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_text_dataset(records: List[Dict], tokenizer):
    """Render each record to a single training string via the model's chat template."""
    from datasets import Dataset
    texts = []
    for r in records:
        messages = build_messages(r["question"], r["snippets"], r["answer"])
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    return Dataset.from_dict({"text": texts})


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="QLoRA SFT with Unsloth.")
    ap.add_argument("--config", default="config/train_config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    # Heavy, GPU-only imports -- intentionally lazy so `import src.train_qlora`
    # (and therefore the test suite) never requires torch/unsloth.
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg.get("max_seq_length", 2048),
        load_in_4bit=cfg.get("load_in_4bit", True),
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 7),
    )

    train_records = load_jsonl(cfg.get("train_path", "data/train.jsonl"))
    dataset = build_text_dataset(train_records, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_length", 2048),
        args=TrainingArguments(
            per_device_train_batch_size=cfg.get("batch_size", 2),
            gradient_accumulation_steps=cfg.get("grad_accum", 4),
            warmup_steps=cfg.get("warmup_steps", 5),
            num_train_epochs=cfg.get("epochs", 3),
            learning_rate=cfg.get("learning_rate", 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=cfg.get("logging_steps", 5),
            optim="adamw_8bit",
            weight_decay=cfg.get("weight_decay", 0.01),
            lr_scheduler_type="linear",
            seed=cfg.get("seed", 7),
            output_dir=cfg.get("output_dir", "outputs"),
            report_to=cfg.get("report_to", "none"),
        ),
    )
    trainer.train()

    adapter_dir = cfg.get("adapter_dir", "outputs/adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

    if cfg.get("save_merged_16bit", False):
        model.save_pretrained_merged(
            cfg.get("merged_dir", "outputs/merged"), tokenizer, save_method="merged_16bit"
        )


if __name__ == "__main__":
    main()
