# domain-llm-finetune
Fine-tune an open-weight LLM into a domain assistant that does three things a regulated business actually cares about: answers only from the provided context, cites the policy it used, and refuses/escalates when no policy covers the question - with a deterministic evaluation harness that proves the behaviour before/after tuning.
> **Why this project exists.** It's Flagship #1 of my accelerated AI-engineering plan:
> turning "I understand evaluation and trust" into "I have fine-tuned an open-weight
> model and measured that it got more grounded, cited correctly, and stopped
> hallucinating sources." Everything here runs on **synthetic data** — no real
> policies, no personal data — so it's fully reproducible and safe to publish.

![CI](https://img.shields.io/badge/CI-green-brightgreen) ![python](https://img.shields.io/badge/python-3.11-blue) ![license](https://img.shields.io/badge/license-MIT-black)

---

## What it demonstrates (for hiring managers)

| Capability | Where to look |
|---|---|
| **Model customisation / fine-tuning** | QLoRA SFT with Unsloth + TRL — `src/train_qlora.py`, `config/train_config.yaml` |
| **Evaluation as engineering** | deterministic, pure-function metrics + before/after report — `src/metrics.py`, `src/eval_harness.py` |
| **Trust by design** | grounded-or-refuse behaviour, citation grounding, hallucinated-citation detection — the metric set itself |
| **Software-engineering discipline** | typed pure functions, unit tests, lint, green CI **without a GPU** — `tests/`, `.github/workflows/ci.yml` |
| **Honesty** | synthetic data, a "what runs vs what needs a GPU" table, a model card with explicit limitations |

This is the shape of real applied-AI work: a clear task, data you control, a model you
customise, and **numbers that say whether it actually got better**.

---

## The task

An employee asks a policy question. The assistant is given a few retrieved policy
snippets as context and must:

1. **Answer only from context** and **cite** the policy id it relied on, e.g. `[POL-101]`; or
2. **Refuse and escalate** when the context contains no policy that answers the question.

The fine-tune teaches the base model to do this reliably. The eval harness measures it.

**Scored behaviours (`src/metrics.py`):**
- `correct_citation_rate` — answerable questions cite the right policy
- `grounded_citation_rate` — every cited id is actually in the provided context
- `hallucinated_citation_rate` — cites an id **not** in context (a trust failure; lower is better)
- `refusal_accuracy` — out-of-scope questions are correctly refused
- `false_refusal_rate` — answerable questions wrongly refused (lower is better)
- `overall_pass_rate` — did the model do the right thing end-to-end

---

## Architecture

```
                 synthetic data                 fine-tune (GPU)              evaluate (GPU)
  data_generator ───────────────► train.jsonl ──► train_qlora.py ──► adapter ──► run_eval.py ──► report (md+json)
       │  (pure, seeded)              eval.jsonl        │ Unsloth QLoRA           │  base vs tuned
       │                                               │                         │
       └──────────────── prompts.py (shared chat formatting: train == eval) ─────┘
                                       │
                                 metrics.py (pure, deterministic) ◄── unit-tested in CI (no GPU)
```

**Design rule — the same one I use across my repos:** all business/eval logic lives in
**pure functions with no `torch`/`unsloth` import**, so the test suite runs green in CI
on CPU. The heavy training/inference imports are **lazy**, inside entrypoints only. That
means the repo's correctness is provable without anyone needing a GPU.

---

## What runs where (honest)

| Component | Runs in CI (CPU) | Needs a GPU |
|---|:---:|:---:|
| Synthetic data generation (`src/data_generator.py`) | ✅ | |
| Prompt/chat formatting (`src/prompts.py`) | ✅ | |
| Metrics + eval harness logic (`src/metrics.py`, `src/eval_harness.py`) | ✅ | |
| Unit tests + lint | ✅ | |
| QLoRA training (`src/train_qlora.py`) | | ✅ |
| Base-vs-tuned model eval (`scripts/run_eval.py`) | | ✅ |

CI proves the *logic*; a GPU produces the *numbers*. Both are documented.

---

## Quickstart

### 1. Logic + tests (no GPU — runs anywhere, including CI)
```bash
pip install -r requirements-dev.txt
python -m src.data_generator --out-dir data --n 240   # writes data/train.jsonl, data/eval.jsonl
pytest -q                                             # unit tests
flake8 src tests scripts                              # lint
```

### 2. Fine-tune + evaluate (GPU — Colab / Kaggle / RunPod / Modal / Lambda)
```bash
pip install -r requirements.txt   # see Unsloth install notes for your CUDA
python -m src.train_qlora --config config/train_config.yaml     # QLoRA SFT -> outputs/adapter
python scripts/run_eval.py --config config/train_config.yaml    # writes reports/eval_report.{md,json}
```

QLoRA on a 7–8B model fits a single 16–24 GB GPU. Swap `model_name` in the config for
`Qwen2.5-7B` or `Mistral-7B` Unsloth 4-bit variants without touching the code.

---

## Results

Populate from your run (`reports/eval_report.md`). Example shape:

| Metric | Base | Fine-tuned |
|---|---|---|
| Overall pass rate | _tbd_ | _tbd_ |
| Correct citation (answerable) | _tbd_ | _tbd_ |
| Grounded citation (answerable) | _tbd_ | _tbd_ |
| False-refusal rate (answerable, lower=better) | _tbd_ | _tbd_ |
| Refusal accuracy (out-of-scope) | _tbd_ | _tbd_ |
| Hallucinated-citation rate (lower=better) | _tbd_ | _tbd_ |

> The interesting story is usually in **refusal accuracy** and **hallucinated-citation
> rate**: base instruct models tend to "help" by answering out-of-scope questions and
> occasionally inventing a plausible-looking policy id. The fine-tune's job is to make
> *grounded-or-refuse* the default.

See `model_card/MODEL_CARD_TEMPLATE.md` for the full card to publish alongside results.

---

## Repository layout

```
domain-llm-finetune/
├── src/
│   ├── prompts.py          # shared chat formatting (train == eval); refusal detection
│   ├── data_generator.py   # synthetic, seeded policy Q&A (answerable + out-of-scope)
│   ├── metrics.py          # pure, deterministic scoring (citation/grounding/refusal)
│   ├── eval_harness.py     # model-agnostic runner + before/after report
│   └── train_qlora.py      # QLoRA SFT (Unsloth + TRL); lazy GPU imports
├── scripts/run_eval.py     # loads base + tuned models, runs the harness (GPU)
├── tests/                  # unit tests for data, prompts, metrics, harness (no GPU)
├── config/train_config.yaml
├── model_card/MODEL_CARD_TEMPLATE.md
├── .github/workflows/ci.yml
├── requirements.txt / requirements-dev.txt / setup.cfg / Makefile
└── LICENSE
```

---

## Roadmap

- [ ] **DPO** preference-tuning on top of SFT (prefer grounded+cited over confident-but-ungrounded answers).
- [ ] **LLM-as-judge** scoring alongside the deterministic metrics.
- [ ] **Flagship #2 — serving:** quantise the adapter (AWQ/GGUF) and serve with vLLM; benchmark latency / throughput / cost vs a closed API.
- [ ] Retrieval front-end so the policy context is retrieved, not supplied.

---

## Honest limitations

- **Synthetic data, single-GPU scale.** Results illustrate the *method*, not production
  performance.
- **Heuristic metrics.** Citation/refusal checks are regex + phrase match, not human eval
  (hence the LLM-as-judge item on the roadmap).
- **No safety red-teaming** and no legal/clinical validation. If adapted to real policies,
  add PII handling, access control, and audit logging.

---

*Part of an accelerated AI-engineering portfolio. Author: Henry Yan ·
github.com/initiative1972. Synthetic/public data only — no employer data is used or
reproduced.*
