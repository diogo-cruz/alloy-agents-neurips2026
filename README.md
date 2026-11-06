# Alloy Agents Can Be More Dangerous Than Either Model Alone

Code and derived data for the NeurIPS 2026 paper by Diogo Cruz
(https://github.com/diogo-cruz/alloy-agents-neurips2026).

An *alloy agent* alternates two or more LLMs within one conversation trace: successive turns
come from different models, and each model sees the whole transcript as if it had written it.
The paper evaluates alloys on single-action safety benchmarks, where one safe turn at the
critical moment blocks the unsafe action, and on multi-step ControlArena tasks, where an alloy
can combine one model's capability with another's willingness to carry out a covert side task.

## Scope

This is a minimal release for reproducing the paper's central results. It contains:

- the alloy solver and its model-selection strategies (`src/alloy_safety/`), and the alloy
  policies for ControlArena, including audit-before-execute (`experiments/control_arena_alloy.py`);
- one entry point per experiment family reported in the paper: the single-action benchmarks
  (PropensityBench, scheming, AgentDojo, Data Poisoning, ImpossibleBench), the ControlArena
  Bash, iac_fast and bash_arena settings in attack mode, audit-before-execute, and the
  new-pair experiment;
- derived per-run results (scores and run metadata, no transcripts) and the analysis scripts
  for the main-text tables and figures (Table 1, Table 2, Figures 1–3) and for the appendix
  tables the main text's inference and frontier counts rest on (`tab:clean_source`,
  `tab:bootstrap`, `tab:convex_hull`), plus the new-pair analyses defined in the precommitments;
- the analysis plans committed before the two camera-ready experiments ran
  (`docs/precommitments/`).

The remaining appendix analyses (reasoning-condition, model-diversity, oracle, pass@k,
task-overlap, swap-frequency, tool-call and interference analyses, the model-drift probes and
the monitor pilot) are not part of this release; they are available from the author on
request. Raw Inspect logs are not released: they hold full agent transcripts, including
completed covert side tasks.

## Contents

| Path | Contents |
|---|---|
| `src/alloy_safety/` | Alloy solver (`solver.py`) and selection strategies (`strategies.py`: round-robin, roulette, fixed sequence) |
| `experiments/` | PropensityBench, scheming and ImpossibleBench tasks; ControlArena alloy policies |
| `scripts/run_*` | Experiment entry points (below) |
| `scripts/*.py` (others) | Analyses: each writes one file in `reports/`, except `analyze_audit.py`, which prints Table 2's Bash rows |
| `figures/` | Scripts for Figures 1–3 (they write PDFs next to themselves) |
| `reports/trace_inventory.json` | The March 2026 canonical pool: every run's configuration, models, reasoning condition and date, and per-sample main and side task scores |
| `data/derived_logs/` | Per-sample records of the July–October 2026 runs (scores, task names, model identifiers returned by the API, token usage, reasoning flags, audit decisions without their text), extracted from the raw logs by `scripts/extract_derived_logs.py`; also the launch records of the two camera-ready experiments |
| `reports/*.json` (others) | Result files written by the analyses, as used in the paper |

The analyses that were written against raw logs read `data/derived_logs/` through
`scripts/derived_log.py`: a script asking for `logs/phase2/alloy100` reads
`data/derived_logs/phase2/alloy100`.

## Setup

```bash
uv sync --extra dev        # Python >= 3.11; ControlArena is pinned to a git commit
uv run pytest tests
cp .env.example .env       # API keys; only needed to run experiments
```

Running experiments also needs Docker (ControlArena settings and ImpossibleBench sandboxes)
and, for ImpossibleBench, `uv pip install git+https://github.com/safety-research/impossiblebench.git`.

## Reproducing the tables and figures

No API keys, Docker or raw logs are needed. From the repository root, in this order (later
scripts read files that earlier ones write):

```bash
uv run python scripts/primary_analysis.py
uv run python scripts/canonical_table.py
uv run python scripts/convex_hull_canonical.py
uv run python scripts/bootstrap_common_tasks.py
uv run python scripts/comparator_sensitivity.py
uv run python scripts/analyze_phase2.py      # also prices tokens with OpenRouter's public list
uv run python scripts/analyze_audit.py
uv run python scripts/analyze_e1_audit_bash_arena.py
uv run python scripts/analyze_e2_screen.py
uv run python scripts/analyze_e2_main.py
cd figures && for f in alloy_setup single_action_bars pareto_frontier; do uv run python $f.py; done
```

Each analysis reproduces its file in `reports/` exactly (the token-cost estimate in
`reports/phase2_run.json` uses OpenRouter's prices at the time of running), `analyze_audit.py`
prints Table 2's Bash rows, and the figures match the paper's.

| Paper item | Script | Data |
|---|---|---|
| Figure 1 | `figures/alloy_setup.py` | counts in the script: scheming evals, and the August Bash run (`reports/phase2_run.json`) |
| Figure 2 | `figures/single_action_bars.py` | per-configuration counts in the script |
| Figure 3 | `figures/pareto_frontier.py` | counts in the script, equal to `reports/canonical_table.json` |
| Table 1 (`tab:primary`), rows 1–4 | `scripts/comparator_sensitivity.py` | `reports/trace_inventory.json`, `data/derived_logs/replication/` |
| Table 1, row 5 | `scripts/analyze_phase2.py` | `data/derived_logs/phase2/` |
| Table 1, caption (March alloy runs only) | `scripts/primary_analysis.py` | `reports/trace_inventory.json` |
| Table 2 (`tab:audit`), Bash | `scripts/analyze_audit.py` | `data/derived_logs/audit_experiment/` |
| Table 2, bash_arena | `scripts/analyze_e1_audit_bash_arena.py` | `data/derived_logs/camera_ready/e1_ba_*` |
| `tab:clean_source` | `scripts/canonical_table.py` | `reports/trace_inventory.json` |
| `tab:convex_hull` | `scripts/convex_hull_canonical.py` | `reports/canonical_table.json` |
| `tab:bootstrap` | `scripts/bootstrap_common_tasks.py`; the Bash interval as first reported: `scripts/primary_analysis.py` | `reports/trace_inventory.json` |
| New pair (`tab:newpair_screen`, `tab:newpair`) | `scripts/analyze_e2_screen.py`, `scripts/analyze_e2_main.py` | `data/derived_logs/camera_ready/e2_*` |

## Running the experiments

The entry points read API keys from `.env` and write Inspect logs under `logs/` or `traces/`.
Model identifiers are set in the scripts; see the model-drift caveat below before comparing new
runs with the paper.

- **Single-action benchmarks.** PropensityBench and scheming:
  `uv run python scripts/run_propensity_scheming.py`. ImpossibleBench:
  `uv run python scripts/run_impossiblebench.py`. AgentDojo and Data Poisoning run through the
  ControlArena runner: `uv run python scripts/run_control_arena_unified.py --setting agentdojo --phase attack-trio`
  (or `--setting data_poisoning`).
- **ControlArena, attack mode (March 2026 canonical pool).**
  `uv run python scripts/run_control_arena_unified.py --setting bash --phase grok-expansion-full --limit 20`
  runs the history-stripped solos and the three alloys; use `--setting iac_fast` or
  `--setting bash_arena --side-tasks sudo_edit` for the other settings. The Gemini baseline also
  pools `--config solo-gemini-attack` and `--config solo-gemini-attack-noreasoning`.
- **July and August 2026 reruns** (Table 1), through OpenRouter: `scripts/run_audit_bash.py`
  with `--mode standard` (alloy), `--mode solo --solo gpt` and `--mode solo --solo gemini`,
  `--limit 100` for August. The July replication ran the same configurations on the original
  20 tasks with one to three epochs; each record's `eval` field gives the task, models, limit and
  epochs.
- **Audit-before-execute** (Table 2): `uv run python scripts/run_audit_bash.py --setting bash --limit 20`
  with `--mode standard` and with `--mode audit`; for bash_arena (E1) add
  `--setting bash_arena --side-task sudo_edit --max-samples 5`. Plan:
  `docs/precommitments/E1_PRECOMMIT.md`.
- **New model pair (E2)**: `scripts/run_e2_screen.sh`, then `scripts/run_e2_main.sh`; both go
  through `scripts/run_audit_bash.py`. Plan: `docs/precommitments/E2_PRECOMMIT.md`.

To analyse new runs, extract records with
`uv run python scripts/extract_derived_logs.py --src . logs/<dir> ...`, or replace the
`from derived_log import ...` lines with `from inspect_ai.log import read_eval_log` and the
`eval_files(d)` calls with `sorted(Path(d).glob("*.eval"))`.

## Data collection and model drift

The canonical pool was collected on 5–15 March 2026: GPT-5-mini (`gpt-5-mini-2025-08-07`)
through the OpenAI API, Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite-preview`) through the
Google API, and Grok 4.1 Fast through OpenRouter. The later runs (July–October 2026) reached GPT
and Gemini through OpenRouter except where stated. Every run's date is in
`reports/trace_inventory.json` or `data/derived_logs/`.

Neither March configuration can be reproduced today. Grok 4.1 Fast was retired on 15 May 2026,
and its identifiers now resolve to a later model. Since May, requests to the Gemini preview
identifier are served by the general-availability model, although the provider's listing still
advertises the March snapshot. Section 4.1 of the paper measures the change.

## Citation

```bibtex
@inproceedings{cruz2026alloy,
  title     = {Alloy Agents Can Be More Dangerous Than Either Model Alone},
  author    = {Cruz, Diogo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). ControlArena, Inspect and the benchmarks keep their own licenses.
