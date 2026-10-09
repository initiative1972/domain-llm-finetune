"""Run base-vs-fine-tuned evaluation on the eval set and write a report.

GPU required (loads models via Unsloth). The deterministic scoring itself lives in
``src/metrics.py`` and ``src/eval_harness.py`` and is tested in CI without a GPU.

Example:
    python scripts/run_eval.py --config config/train_config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_harness import evaluate, load_jsonl, to_markdown, write_report  # noqa: E402
from src.train_qlora import load_config  # noqa: E402


def make_generate_fn(model, tokenizer, max_new_tokens: int = 128):
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)

    def generate_fn(messages):
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )
        text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return text.strip()

    return generate_fn


def load_model(model_name: str, max_seq_length: int = 2048):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Base vs fine-tuned evaluation.")
    ap.add_argument("--config", default="config/train_config.yaml")
    ap.add_argument("--eval-path", default="data/eval.jsonl")
    ap.add_argument("--out-json", default="reports/eval_report.json")
    ap.add_argument("--out-md", default="reports/eval_report.md")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    records = load_jsonl(args.eval_path)
    max_len = cfg.get("max_seq_length", 2048)

    base_model, base_tok = load_model(cfg["model_name"], max_len)
    base_metrics = evaluate(make_generate_fn(base_model, base_tok), records)["metrics"]

    tuned_model, tuned_tok = load_model(cfg.get("adapter_dir", "outputs/adapter"), max_len)
    tuned_metrics = evaluate(make_generate_fn(tuned_model, tuned_tok), records)["metrics"]

    report = {"base": base_metrics, "tuned": tuned_metrics}
    write_report(report, args.out_json, args.out_md)
    print(to_markdown(report))


if __name__ == "__main__":
    main()
