"""domain-llm-finetune: QLoRA fine-tuning of an open-weight LLM into a grounded,
citation-first policy & compliance assistant, with a deterministic eval harness.

Business/eval logic lives in pure functions (no torch/unsloth) so it is unit-testable
in CI without a GPU. Heavy training/inference imports are lazy, inside entrypoints.
"""

__version__ = "0.1.0"
