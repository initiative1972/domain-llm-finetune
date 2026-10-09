"""DPO (Direct Preference Optimization) on top of the SFT adapter, with Unsloth + TRL.

Run on a GPU, after SFT (``src/train_qlora.py``). Starts from the SFT LoRA adapter and
nudges the model to prefer grounded + correctly-cited answers (and correct refusals)
over the tempting failure modes encoded in the rejected responses.

Heavy imports (torch, unsloth, trl) are lazy, inside ``main()``, so the package stays
import-safe for CI and unit tests.

Example:
    python -m src.preferences --out-dir data --n 240       # build pref pairs
    python -m src.train_dpo --config config/train_config.yaml
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from .prompts import build_messages
from .train_qlora import load_config, load_jsonl


def build_pref_dataset(pairs: List[Dict], tokenizer):
    """Build a TRL DPO dataset with prompt / chosen / rejected string columns."""
    from datasets import Dataset

    prompts, chosen, rejected = [], [], []
    for p in pairs:
        messages = build_messages(p["question"], p["snippets"])  # system + user, no answer
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
        chosen.append(p["chosen"])
        rejected.append(p["rejected"])
    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="DPO preference tuning with Unsloth.")
    ap.add_argument("--config", default="config/train_config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    # Heavy, GPU-only imports -- PatchDPOTrainer must run before importing trl.
    import torch
    from unsloth import FastLanguageModel, PatchDPOTrainer

    PatchDPOTrainer()
    from trl import DPOConfig, DPOTrainer

    init_adapter = cfg.get("dpo_init_adapter", cfg.get("adapter_dir", "outputs/adapter"))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=init_adapter,
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

    pairs = load_jsonl(cfg.get("pref_train_path", "data/pref_train.jsonl"))
    dataset = build_pref_dataset(pairs, tokenizer)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth/PEFT uses the frozen base as the implicit reference
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            beta=cfg.get("dpo_beta", 0.1),
            per_device_train_batch_size=cfg.get("dpo_batch_size", 2),
            gradient_accumulation_steps=cfg.get("dpo_grad_accum", 4),
            num_train_epochs=cfg.get("dpo_epochs", 1),
            learning_rate=cfg.get("dpo_learning_rate", 5e-6),
            warmup_steps=cfg.get("warmup_steps", 5),
            logging_steps=cfg.get("logging_steps", 5),
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            seed=cfg.get("seed", 7),
            max_length=cfg.get("max_seq_length", 2048),
            max_prompt_length=cfg.get("max_prompt_length", 1024),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            output_dir=cfg.get("dpo_output_dir", "outputs/dpo"),
            report_to=cfg.get("report_to", "none"),
        ),
    )
    trainer.train()

    dpo_adapter_dir = cfg.get("dpo_adapter_dir", "outputs/dpo_adapter")
    model.save_pretrained(dpo_adapter_dir)
    tokenizer.save_pretrained(dpo_adapter_dir)
    print(f"Saved DPO adapter to {dpo_adapter_dir}")


if __name__ == "__main__":
    main()
