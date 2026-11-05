# E1 precommitment — audit-before-execute on bash_arena (fixed harness)

> Reproduced from the author's working repository, where this plan was committed on
> 2026-10-10 before the experiment launched; the dated addendum was added after the run.
> Raw-log paths (`logs/...`) correspond to the released per-sample records under
> `data/derived_logs/`.

Committed before launch, 2026-10-10. Nothing below may change after the first sample runs;
any deviation is logged in a dated addendum at the bottom.

## Question
Does audit-before-execute suppress the side task on a second harm type? The Bash result
(malware download: side 70% → 0%, n=20) is the only one reported in the rebuttal. The July
bash_arena run (93% → 0%) used the harness that truncated proposed actions at 1,500
characters, so it was never reported, and its rerun on the fixed harness was stopped
unfinished for the response deadline.

## Design
- **Setting**: ControlArena bash_arena, side task `sudo_edit` (privilege escalation), attack
  mode. Self-contained harm: the escalation lands when the command executes.
- **Models**: `openrouter/openai/gpt-5-mini` + `openrouter/google/gemini-3.1-flash-lite-preview`
  (served today by the May GA model), round-robin, GPT takes turn 1. Same pair, route and
  policy code as the Bash audit run.
- **Arms**: (1) standard round-robin alloy (unmediated); (2) audit-before-execute: before a
  proposed tool call executes, the partner model reviews it, seeing the main task, the
  environment description and the full proposed action (up to 20,000 characters), but not the
  side-task instruction; on DENY the tool calls are stripped. Prompt: `AUDIT_REVIEWER_PROMPT`
  in `experiments/control_arena_alloy.py` at this commit.
- **Tasks**: the first 20 bash_arena tasks (`--limit 20`), 1 epoch, `max_steps` 30,
  `time_limit` 3600 s, `--max-samples 5`.
- **Order**: standard arm, then audit arm, sequentially on the same day.
- **Logs**: `logs/camera_ready/e1_ba_standard/`, `logs/camera_ready/e1_ba_audit/`.

## Outcomes and analysis (`scripts/analyze_e1_audit_bash_arena.py` → `reports/e1_ba_audit.json`)
- **Primary**: side-task success, audit vs standard, one-sided Fisher's exact test
  (H1: audit < standard).
- **Secondary**: mean fractional main-task score per arm and the audit − standard difference
  with an unpaired bootstrap 95% CI (20,000 resamples, seed 20261010); combined success as the
  mean of main_frac × 1[side]; vetoes (total, per acting model, share of audited actions);
  vetoes whose stated reason matches `sudo|sudoers|privilege|escalat` (case-insensitive).

## Reporting rule
- One run per arm, reported as-is. Errored samples are excluded from rates and counted.
- No reruns, except if an arm fails before any sample completes (an infrastructure failure),
  which is logged here with the cause.
- If either arm errors on more than 25% of samples, report as-is with the error rate flagged;
  do not add runs to compensate.

## Interpretation (fixed before results)
| Outcome | Reading |
|---|---|
| audit side ≤ 10% and Fisher p < 0.05 | The mitigation replicates on a second harm type |
| p < 0.05 but audit side > 10% | Partial: suppression without elimination |
| p ≥ 0.05 | Does not replicate on bash_arena |

The main-task cost is reported whatever the outcome. Table E in the paper gains a bash_arena
row in every case.

## Addendum (2026-10-10, after the run)
- **Run:** standard arm 16:59–17:50 UTC (20/20 scored), audit arm 17:51–18:46 UTC (19/20
  scored; one sample lost to a sandbox error, 5%, under the 25% flag). Launch record:
  `logs/camera_ready/e1_launch.txt`. OpenRouter spend $1.56.
- **Deviation, none affecting E1:** between the two arms, `scripts/run_audit_bash.py` gained
  `--models` and a `start,stop` form of `--limit` (commit `04b69dd`, for E2). The E1 audit
  invocation (`--limit 20`, no `--models`) parses exactly as before.
- **Result** (`reports/e1_ba_audit.json`): side task 19/20 → 0/19, one-sided Fisher
  p = 2.9e-10; mean fractional main 0.312 → 0.346, difference +0.035, 95% bootstrap CI
  [-0.128, +0.198]; combined 0.312 → 0.000; 60 of 317 audited actions denied (46 of Gemini's,
  14 of GPT-5-mini's), 44 naming sudo/privilege escalation.
- **Reading under the precommitted table:** audit side ≤ 10% and p < 0.05, so the mitigation
  replicates on a second harm type. No main-task cost is detectable on bash_arena, unlike Bash.
