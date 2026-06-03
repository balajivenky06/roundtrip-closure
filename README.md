# roundtrip-closure

**Heterogeneous Multi-SLM Closure of the Docstring–Test–Code Triangle: A Mutation-Testing Study**

Chapter 3 of the PhD thesis on Small Language Models for Software Engineering.
Companion to:

- Chapter 1 — Docstring generation (under review at Springer-Nature *Automated Software Engineering*)
- Chapter 2 — Unit-test generation evaluated by mutation kill rate (under review at Springer-Nature *Software Quality Journal*)

## What this experiment measures

For each function `f` in HumanEval + MBPP, we traverse the
docstring–test–code triangle with **3 different Small Language Models
assigned to 3 stages**:

```
Stage 1  L_spec :  code C → docstring D'
Stage 2  L_test :  docstring D' → test suite T'
Stage 3  L_code :  (D' + T') → reconstructed code C'

Closure metric:  did the round-trip preserve semantics?
                  measured by mutation kill rate (T' against C),
                  reference-test pass rate (original tests on C'),
                  and judge-LLM equivalence (D' vs original D).
```

The central question: **does closure rate improve when each stage is
owned by a different SLM (heterogeneous configuration) compared to
the same SLM filling all three stages (mono configuration)?**

## Model lineup (all Small Language Models, all <30B parameters)

| Slot | Ollama tag | Family | Size | Generation |
|---|---|---|---|---|
| Small floor | `llama3.2:3b` | Meta | 3 B dense | Oct 2024 |
| Mid-dense reasoning | `phi4:14b` | Microsoft | 14 B dense | Dec 2024 |
| Latest dense general | `qwen3.6:27b` | Alibaba | 27 B dense | Apr 2026 |
| Latest MoE | `gemma4:26b` | Google | 26 B MoE | Mar 2026 |
| Latest Mistral | `mistral-small3.2:24b` | Mistral | 24 B dense | 2025 |
| Coder specialist | `qwen3-coder:30b` | Alibaba (coder) | 30 B MoE | 2025 |
| **Judge** | `deepseek-r1:14b` | DeepSeek | 14 B dense | 2025 |

Five distinct model families in the pipeline; DeepSeek as the
external judge for closure-validity checks.

## Quick start (once filled in)

```bash
# 0. install
pip install -e .

# 1. one-time setup — pull models, download datasets, build index
ollama pull llama3.2:3b
ollama pull phi4:14b
ollama pull qwen3.6:27b
ollama pull gemma4:26b
ollama pull mistral-small3.2:24b
ollama pull qwen3-coder:30b
ollama pull deepseek-r1:14b

python prepare_roundtrip.py

# 2. run one cell of the DOE (edit CELL_ID in train_roundtrip.py first)
python train_roundtrip.py > logs/cell_M3.log 2>&1

# 3. run the 30-function pilot (6 cells, ~14 GPU-hours on A100)
python scripts/run_pilot.py
```

## Project layout

```
roundtrip-closure/
├── README.md                ← this file
├── concept_note.md          ← signed-off Chapter 3 design doc
├── pyproject.toml           ← deps + project metadata
├── .env.example             ← API-key template (not currently needed)
├── .gitignore
│
├── config.py                ← model lineup + per-cell config
├── doe.py                   ← 20-cell pre-registered DOE table
├── train_roundtrip.py       ← main experiment driver
├── prepare_roundtrip.py     ← one-time dataset prep
│
├── ollama_client.py         ← Ollama wrapper with retry + rate limiting
├── closure_cache.py         ← SHA256-keyed disk cache
├── closure_paths.py         ← Path-1/2/3 traversal drivers
├── closure_metrics.py       ← kill rate / pass rate / BERTScore
├── judge_llm.py             ← DeepSeek-R1 equivalence judge
├── decontaminate.py         ← HumanEval-Mutated transform (AST + LLM)
├── mutation_testing.py      ← copied from autoresearch (Chapter 2)
│
├── scripts/
│   └── run_pilot.py         ← driver for the 30-function pilot
├── tests/
│   └── test_smoke.py        ← end-to-end smoke test (1 function)
│
├── data/                    ← datasets (gitignored)
├── checkpoints/             ← cache + intermediate artifacts
├── results/                 ← per-cell TSV outputs
└── logs/                    ← per-cell run logs
```

## Status (2026-06-03)

**Phase:** Scaffolding complete. Module bodies to be filled in.

| Component | Status |
|---|---|
| Project structure | ✓ scaffold created |
| `mutation_testing.py` | ✓ copied from autoresearch |
| `config.py`, `doe.py` | ⏳ stub files with structure |
| `ollama_client.py` | ⏳ stub |
| `closure_*.py` | ⏳ stubs |
| `judge_llm.py` | ⏳ stub |
| `decontaminate.py` | ⏳ stub |
| `prepare_roundtrip.py` | ⏳ stub |
| `train_roundtrip.py` | ⏳ stub |
| Pilot run | ⏳ blocked on module bodies |
| Full sweep | ⏳ blocked on pilot |

## License

MIT — replication package for the PhD thesis.
