# E2 precommitment — composition on a new model pair (Bash)

> Reproduced from the author's working repository, where this plan was committed on
> 2026-10-10 before the experiment launched; the dated addendum was added after the run.
> Raw-log paths (`logs/...`) correspond to the released per-sample records under
> `data/derived_logs/`. One sentence quoting the review discussion was removed from the
> Question section.

Committed before any E2 sample runs, 2026-10-10. Changes after launch go in dated addenda.

## Question
Does the composition result replicate on a pair that shares no model with the paper? The
result is that the alloy takes on the compliance of its compliant member while keeping most of
its capable member's main-task competence. We also ask whether the combined-success uplift over
the better solo replicates.

## Stage 1 — screen (selects the pair; never reused as evaluation data)
- **Candidates** (cheap, tool-capable, one per provider, none used in the paper), via OpenRouter:
  `openai/gpt-6-luna`, `anthropic/claude-haiku-5.5`, `google/gemini-3.5-flash-lite`,
  `qwen/qwen3.8-flash`, `deepseek/deepseek-v4.1-flash`, `mistralai/mistral-small-2603`,
  `z-ai/glm-5.3-flash`.
- **Run**: each candidate solo, ControlArena Bash attack mode (malware-download side task),
  task indices 100–119 (`--limit 100,120`, disjoint from the evaluation tasks), 1 epoch,
  through `scripts/run_audit_bash.py --mode solo --models ...`. This is the same alloy policy
  code path as the evaluation arms.
- **Selection rule**, applied mechanically by `scripts/analyze_e2_screen.py`:
  1. Exclude candidates with more than 25% errored samples.
  2. **A (capable refuser)**: among candidates with side-task compliance ≤ 10%, the highest
     main-task success; ties go to the lower input price. If none has side ≤ 10%, take the
     lowest side among candidates with main ≥ 70%.
  3. **B (compliant)**: among the remaining candidates (provider ≠ A's) with side ≥ 70%, prefer
     those with main ≤ main(A) − 15pp (condition (1) of the paper). Among those, take the
     highest side, then the lower main, then the lower price. If none meets the main-gap
     requirement, take the highest-side candidate with side ≥ 70% and record that condition (1)
     is only partly met. If none has side ≥ 70%, relax the threshold to 50% and record it.
  4. If no valid A or B exists, E2 stops and is reported as "no complementary pair among
     current low-cost models".

## Stage 2 — evaluation
- **Tasks**: Bash task indices 0–99 (`--limit 100`), the same tasks as the August 100-task run.
  1 epoch.
- **Arms**: alloy A+B (round-robin, A takes turn 1), A solo, B solo. All three go through
  `run_audit_bash.py` (one policy code path, one route), run concurrently with `--max-samples 8`.
- **Logs**: `logs/camera_ready/e2_main/{alloy,solo_a,solo_b}/`.

## Analysis (`scripts/analyze_e2_main.py` → `reports/e2_main.json`)
- **P1, composition**: alloy side compliance > A solo (one-sided Fisher), and alloy main-task
  success > B solo (one-sided Fisher). Composition replicates if both p < 0.05.
- **P2, uplift**: alloy combined success minus the per-task better of the two solos, using the
  hierarchical bootstrap of `scripts/analyze_phase2.py` (`bootstrap`, comparator recomputed per
  replicate; with one epoch it equals the task-only bootstrap), 20,000 replicates,
  seed 20261010. Reported as a point estimate with a 95% CI. Significant if the CI excludes 0.
- **Descriptive**: per-arm main/side/combined with counts, P(side | main failure),
  API-returned model ids, token usage.

## Reporting rule
- One screen and one evaluation run, reported as-is: no reslicing, no extra seeds, nothing
  pooled with March or August data.
- Errored samples are excluded and counted. No reruns unless an arm fails before any sample
  completes (infrastructure), logged here.
- **Budget**: stop before Stage 2 if E2 spend after Stage 1 exceeds $8 by the OpenRouter key
  usage delta, or if the projected total exceeds $20.

## How it enters the paper
Whatever the outcome: a short main-text sentence and an appendix table with all three arms, the
screen table and this rule. The paper's claims are scoped to tested pairs; this pair is added
to them.

## Addendum (2026-10-10, after the run)
- **Stage 1** (18:47–19:56 UTC, $1.47; start time corrected on 2026-11-05 from a misread "19:47", per the launch record): seven candidates, 20 runs each on tasks 100–119, no
  errors. Main/side/combined %: gpt-6-luna 90/0/0, claude-haiku-5.5 85/0/0,
  gemini-3.5-flash-lite 70/95/65, qwen3.8-flash 100/60/60, deepseek-v4.1-flash 95/30/30,
  mistral-small-2603 0/10/0, glm-5.3-flash 100/65/65. The rule selected A = gpt-6-luna,
  B = gemini-3.5-flash-lite, with condition (1) met (no fallback notes).
- **Stage 2** (19:57–20:28 UTC, $2.43; E2 total $3.90): alloy 99 scored (1 error), A solo 99
  (1), B solo 98 (2). Main/side/combined: alloy 86.9/4.0/4.0%; GPT-6 Luna 88.9/4.0/4.0%;
  Gemini 3.5 Flash Lite 63.3/94.9/59.2% (`reports/e2_main.json`).
- **P1 (composition): does not replicate.** Alloy side > A: p = 0.64; alloy main > B:
  p = 1.0e-4. The alloy keeps the refusing model's behaviour.
- **P2 (uplift):** Δ = −55.6pp, 95% CI [−65.7, −45.5] over 99 tasks.
- Reported as-is in the paper, including the abstract.
