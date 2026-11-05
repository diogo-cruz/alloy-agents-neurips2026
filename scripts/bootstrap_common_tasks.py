"""Table C restricted to tasks run by all three configurations.

The first hierarchical bootstrap (scripts/primary_analysis.py) keeps
every task the alloy ran on together with at least one solo, scoring a solo that never ran a
task as 0 there. That matters only outside Bash, where some solos covered fewer tasks. This
script reruns the same bootstrap (same pool, seed, replicates and per-replicate comparator)
on tasks run by the alloy and both solos, which is how the paper defines Delta.

Run: uv run --no-sync python scripts/bootstrap_common_tasks.py
Writes reports/bootstrap_common_tasks.json
"""
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INV = json.load(open(REPO / "reports/trace_inventory.json"))["traces"]
NB = 20000
ALLOYS = {"GPT+Gemini-Lite": ("GPT", "Gemini-Lite"), "GPT+Grok": ("GPT", "Grok"),
          "Gemini-Lite+Grok": ("Gemini-Lite", "Grok")}


def in_pool(r):
    if r["config_name"] == "Gemini-Lite":
        return True
    return r["reasoning"] == ("C1" if r["is_alloy"] else "C2")


def by_task(setting, cfg):
    d = defaultdict(list)
    for r in INV:
        if r["setting"] != setting or r["config_name"] != cfg or not in_pool(r):
            continue
        for s in r["samples"]:
            ss, ms = s.get("side_score"), s.get("main_score")
            if ss is None:
                continue
            m = 1.0 if ms == "C" else (0.0 if ms == "I" else (float(ms) if isinstance(ms, (int, float)) else 0.0))
            d[s.get("main_task", "?")].append(m * (1 if ss == "C" else 0))
    return d


def mean(xs):
    return sum(xs) / len(xs)


def boot(A, SA, SB, tasks):
    nt = len(tasks)
    rs = lambda L: mean([L[random.randrange(len(L))] for _ in L])
    point = 100 * (sum(mean(A[t]) for t in tasks) - sum(max(mean(SA[t]), mean(SB[t])) for t in tasks)) / nt
    diffs = sorted(100 * sum(rs(A[t]) - max(rs(SA[t]), rs(SB[t]))
                             for t in (tasks[random.randrange(nt)] for _ in range(nt))) / nt
                   for _ in range(NB))
    return dict(point=round(point, 1), lo=round(diffs[int(.025 * NB)], 1),
                hi=round(diffs[int(.975 * NB)], 1), n_tasks=nt)


def main():
    random.seed(20260728)
    out = []
    for setting in ("bash", "iac_fast", "bash_arena"):
        for alloy, (a, b) in ALLOYS.items():
            A, SA, SB = by_task(setting, alloy), by_task(setting, a), by_task(setting, b)
            common = sorted(t for t in A if A[t] and SA.get(t) and SB.get(t))
            r = boot(A, SA, SB, common) if common else None
            out.append(dict(setting=setting, alloy=alloy, **(r or dict(n_tasks=0))))
            print(f"{setting:10s} {alloy:17s} {r}")
    json.dump(out, open(REPO / "reports/bootstrap_common_tasks.json", "w"), indent=1)
    print("[saved] reports/bootstrap_common_tasks.json")


if __name__ == "__main__":
    main()
