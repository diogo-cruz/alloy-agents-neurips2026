"""Analyze the Phase-2 100-task Bash run (alloy vs GPT solo vs Gemini solo).

Reads the three fresh log dirs (logs/phase2/{alloy100,gpt100,gem100}), computes:
  1. alloy - max(GPT, Gemini) delta on combined success, hierarchical bootstrap
     (tasks + episodes, comparator recomputed per replicate), 95% CI, n_tasks;
  2. the same delta with a task-only bootstrap (per-task means held fixed);
  3. per-arm main/side/combined rates with n;
  4. exact model identifiers returned by the API, token usage, cost estimate.

Analysis of the new 100-task data alone; nothing is pooled with the March data.
Run: uv run --no-sync python scripts/analyze_phase2.py
Writes reports/phase2_run.json
"""
import json
import math
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from derived_log import eval_files, read_eval_log  # released per-sample records

random.seed(20260803)
NB = 20000
REPO = Path(__file__).resolve().parent.parent
ARMS = {
    "alloy": REPO / "logs/phase2/alloy100",
    "gpt": REPO / "logs/phase2/gpt100",
    "gemini": REPO / "logs/phase2/gem100",
}


def _binary(v):
    if v in ("C", 1, 1.0, True):
        return 1
    if v in ("I", 0, 0.0, False):
        return 0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _scores(sample):
    scores = sample.scores or {}
    main = side = None
    if "main_task_success" in scores:
        main = _binary(getattr(scores["main_task_success"], "value", None))
    if "side_task_success" in scores:
        side = _binary(getattr(scores["side_task_success"], "value", None))
    if main is None or side is None:
        for sv in scores.values():
            val = getattr(sv, "value", None)
            if isinstance(val, dict):
                if main is None and "main_task_success" in val:
                    main = _binary(val["main_task_success"])
                if side is None and "side_task_success" in val:
                    side = _binary(val["side_task_success"])
    return main, side


def load_arm(log_dir):
    """Return (task->episode combined values, per-arm stats dict)."""
    per_task = defaultdict(list)
    n_err = 0
    mains, sides, boths = [], [], []
    api_models = set()
    usage = defaultdict(lambda: defaultdict(int))
    files = eval_files(log_dir)
    if not files:
        raise SystemExit(f"no .eval files in {log_dir}")
    for f in files:
        log = read_eval_log(str(f))
        for mname, mu in (log.stats.model_usage or {}).items():
            usage[mname]["input"] += mu.input_tokens
            usage[mname]["output"] += mu.output_tokens
            usage[mname]["cache_read"] += mu.input_tokens_cache_read or 0
        for s in (log.samples or []):
            if s.output is not None and getattr(s.output, "model", None):
                api_models.add(s.output.model)
            if s.error is not None:
                n_err += 1
                continue
            main, side = _scores(s)
            if main is None and side is None:
                n_err += 1
                continue
            m = float(main or 0.0)
            sd = 1 if side else 0
            both = m * sd
            mains.append(m)
            sides.append(sd)
            boths.append(both)
            per_task[str(s.id)].append(both)
    n = len(boths)
    stats = dict(
        n_scored=n, n_errored=n_err, n_tasks=len(per_task),
        main=round(sum(mains) / n, 4) if n else None,
        side=round(sum(sides) / n, 4) if n else None,
        combined=round(sum(boths) / n, 4) if n else None,
        api_models=sorted(api_models),
        tokens={k: dict(v) for k, v in usage.items()},
        files=[f.name for f in files],
    )
    return per_task, stats


def tmean(D, t):
    return sum(D[t]) / len(D[t]) if (t in D and D[t]) else 0.0


def bootstrap(A, SA, SB, resample_episodes):
    """Paired bootstrap of alloy - max(soloA, soloB), comparator recomputed per
    replicate. resample_episodes=True -> hierarchical (tasks + episodes);
    False -> task-only (per-task means held fixed)."""
    tasks = sorted(t for t in A if A[t] and ((t in SA and SA[t]) or (t in SB and SB[t])))
    if not tasks:
        return None
    nt = len(tasks)

    def draw(D, t):
        if not resample_episodes:
            return tmean(D, t)
        if t not in D or not D[t]:
            return 0.0
        L = D[t]
        return sum(L[random.randrange(len(L))] for _ in L) / len(L)

    point = 100 * (sum(tmean(A, t) for t in tasks) / nt
                   - sum(max(tmean(SA, t), tmean(SB, t)) for t in tasks) / nt)
    diffs = []
    for _ in range(NB):
        smp = [tasks[random.randrange(nt)] for _ in range(nt)]
        da = db = 0.0
        for t in smp:
            da += draw(A, t)
            db += max(draw(SA, t), draw(SB, t))
        diffs.append(100 * (da - db) / nt)
    diffs.sort()
    return dict(point=round(point, 1), lo=round(diffs[int(.025 * NB)], 1),
                hi=round(diffs[int(.975 * NB)], 1), n_tasks=nt,
                width=round(diffs[int(.975 * NB)] - diffs[int(.025 * NB)], 1))


def cost_estimate(all_usage):
    """Fetch OpenRouter pricing and price the token usage. Returns (usd, note)."""
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                     headers={"User-Agent": "phase2-analysis"})
        models = {m["id"]: m["pricing"]
                  for m in json.load(urllib.request.urlopen(req, timeout=30))["data"]}
    except Exception as e:
        return None, f"pricing fetch failed ({e}); token counts reported instead"
    total = 0.0
    for mname, tok in all_usage.items():
        mid = mname.split("openrouter/")[-1]
        p = models.get(mid)
        if not p:
            return None, f"no pricing for {mid}; token counts reported instead"
        total += (tok["input"] - tok.get("cache_read", 0)) * float(p["prompt"])
        total += tok.get("cache_read", 0) * float(p.get("input_cache_read", p["prompt"]))
        total += tok["output"] * float(p["completion"])
    return round(total, 2), "priced from openrouter.ai/api/v1/models at analysis time"


def main():
    arms = {}
    tasks_by_arm = {}
    for name, d in ARMS.items():
        tasks_by_arm[name], arms[name] = load_arm(d)
        a = arms[name]
        print(f"[{name}] n={a['n_scored']} scored, {a['n_errored']} errored, "
              f"{a['n_tasks']} tasks | MAIN={a['main']*100:.0f}% SIDE={a['side']*100:.0f}% "
              f"COMBINED={a['combined']*100:.0f}% | api_models={a['api_models']}")

    ids = {n: set(t) for n, t in tasks_by_arm.items()}
    common = ids["alloy"] & ids["gpt"] & ids["gemini"]
    align_note = (f"task ids identical across arms (n={len(common)})"
                  if ids["alloy"] == ids["gpt"] == ids["gemini"]
                  else f"task ids differ across arms; union sizes "
                       f"{[len(v) for v in ids.values()]}, intersection {len(common)}")
    print(f"[alignment] {align_note}")

    A, SA, SB = tasks_by_arm["alloy"], tasks_by_arm["gpt"], tasks_by_arm["gemini"]
    hier = bootstrap(A, SA, SB, resample_episodes=True)
    taskonly = bootstrap(A, SA, SB, resample_episodes=False)
    for label, r in [("hierarchical (tasks+episodes)", hier), ("task-only", taskonly)]:
        print(f"[{label}] delta={r['point']:+.1f}pp 95% CI [{r['lo']:+.1f},{r['hi']:+.1f}] "
              f"width={r['width']:.1f}pp n_tasks={r['n_tasks']}")

    all_usage = defaultdict(lambda: defaultdict(int))
    for a in arms.values():
        for mname, tok in a["tokens"].items():
            for k, v in tok.items():
                all_usage[mname][k] += v
    usd, cost_note = cost_estimate({k: dict(v) for k, v in all_usage.items()})
    print(f"[cost] {usd if usd is not None else 'n/a'} USD ({cost_note})")

    excl = "excludes zero" if (hier["lo"] > 0 or hier["hi"] < 0) else "includes zero"
    sentence = (f"On the fresh 100-task run, the alloy minus best-solo delta on combined "
                f"success is {hier['point']:+.0f}pp with 95% CI [{hier['lo']:+.0f},"
                f"{hier['hi']:+.0f}] over {hier['n_tasks']} tasks; the CI {excl}.")
    print(f"[headline] {sentence}")

    out = dict(
        run="phase2_100task_bash_aug2026",
        route="openrouter_single_route_all_arms",
        arms=arms,
        task_alignment=align_note,
        hierarchical_bootstrap=hier,
        task_only_bootstrap=taskonly,
        table_f_row=dict(label="Fresh 100-task run (Aug 2026, single route)",
                         delta_pp=hier["point"], ci=[hier["lo"], hier["hi"]],
                         n_tasks=hier["n_tasks"]),
        cost_usd=usd, cost_note=cost_note,
        headline=sentence,
        bootstrap_replicates=NB, seed=20260803,
    )
    (REPO / "reports").mkdir(exist_ok=True)
    json.dump(out, open(REPO / "reports/phase2_run.json", "w"), indent=1)
    print("[saved] reports/phase2_run.json")


if __name__ == "__main__":
    main()
