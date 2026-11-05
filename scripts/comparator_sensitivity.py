#!/usr/bin/env python3
"""How much does the primary Bash GPT+Gemini result depend on the Gemini comparator?

The canonical pool treats Gemini as degenerate across reasoning conditions and pools
C1+C2+C3, while GPT and Grok solos use C2 only. The observed Gemini cells differ
(C1 41%, C2 57%, C3 44% combined success) even though the task sets are identical,
so the pooling choice moves the comparator and therefore the headline delta.

This recomputes the primary contrast under several comparator definitions, including
fresh replication episodes collected through OpenRouter in July 2026.

Run: uv run --no-sync python scripts/comparator_sensitivity.py
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

random.seed(20260728)
REPO = Path(__file__).resolve().parents[1]
NB = 20000

INV = json.load(open(REPO / 'reports/trace_inventory.json'))['traces']


def canonical_by_task(cfg, gem_cond=None):
    """Per-task combined-success episodes from the committed canonical traces."""
    d = defaultdict(list)
    for r in INV:
        if r['setting'] != 'bash' or r['config_name'] != cfg:
            continue
        if cfg == 'Gemini-Lite':
            if gem_cond and r['reasoning'] != gem_cond:
                continue
        elif r['reasoning'] != ('C1' if r['is_alloy'] else 'C2'):
            continue
        for s in r['samples']:
            ss = s.get('side_score')
            if ss is None:
                continue
            ms = s.get('main_score')
            m = 1.0 if ms == 'C' else (0.0 if ms == 'I' else (float(ms) if isinstance(ms, (int, float)) else 0.0))
            d[s.get('main_task', '?')].append(m * (1 if ss == 'C' else 0))
    return d


def replication_by_task(subdir):
    """Per-task combined success from a fresh replication run."""
    from derived_log import eval_files, read_eval_log  # released per-sample records
    from analyze_audit import _scores
    p = REPO / 'logs/replication' / subdir
    files = eval_files(p)
    if not files:
        return defaultdict(list)
    d = defaultdict(list)
    for f in files:
        log = read_eval_log(str(f))
        for s in log.samples or []:
            if s.error:
                continue
            m, side = _scores(s)
            if m is None and side is None:
                continue
            # Key on the task NAME, not the sample id. The committed canonical traces
            # use main_task names, so keying on s.id here would leave the canonical and
            # replication dictionaries with disjoint keys, and any merge would silently
            # score half the tasks against an absent (zero) comparator.
            task = (s.metadata or {}).get('main_task_name')
            if not task:
                continue
            mv = 1.0 if m in ('C', 1, 1.0, True) else (float(m) if isinstance(m, (int, float)) else 0.0)
            d[str(task)].append(mv * (1 if side == 1 else 0))
    return d


def merge(*ds):
    out = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            out[k].extend(v)
    return out


def rate(d):
    ep = [x for v in d.values() for x in v]
    return (sum(ep) / len(ep) * 100, len(ep)) if ep else (float('nan'), 0)


def boot(A, SA, SB):
    tasks = [t for t in A if A[t] and ((t in SA and SA[t]) or (t in SB and SB[t]))]
    nt = len(tasks)
    tm = lambda D, t: (sum(D[t]) / len(D[t])) if (t in D and D[t]) else 0.0

    def rs(D, t):
        if t not in D or not D[t]:
            return 0.0
        L = D[t]
        return sum(L[random.randrange(len(L))] for _ in L) / len(L)

    pt = 100 * (sum(tm(A, t) for t in tasks) / nt - sum(max(tm(SA, t), tm(SB, t)) for t in tasks) / nt)
    ds = []
    for _ in range(NB):
        smp = [tasks[random.randrange(nt)] for _ in range(nt)]
        ds.append(100 * (sum(rs(A, t) for t in smp) - sum(max(rs(SA, t), rs(SB, t)) for t in smp)) / nt)
    ds.sort()
    return pt, ds[int(.025 * NB)], ds[int(.975 * NB)], nt


# --- the alloy and GPT sides are held fixed; only the Gemini comparator varies ---
A_canon = canonical_by_task('GPT+Gemini-Lite')
G_canon = canonical_by_task('GPT')
A_rep = replication_by_task('alloy')
G_rep = replication_by_task('gpt_solo')

gem_variants = {
    'canonical, pooled C1+C2+C3 (as used in the paper)': canonical_by_task('Gemini-Lite'),
    'canonical, C2 only (protocol-matched to GPT/Grok)': canonical_by_task('Gemini-Lite', 'C2'),
    'fresh replication only (OpenRouter, Jul 2026)': merge(replication_by_task('gem_solo'),
                                                           replication_by_task('gem_solo_more')),
    'ALL Gemini episodes (canonical + replication)': merge(canonical_by_task('Gemini-Lite'),
                                                          replication_by_task('gem_solo'),
                                                          replication_by_task('gem_solo_more')),
}

print("=" * 78)
print("PRIMARY CONTRAST, Bash GPT+Gemini, varying ONLY the Gemini comparator")
print("=" * 78)
alloy_all = merge(A_canon, A_rep)
gpt_all = merge(G_canon, G_rep)
ar, an = rate(alloy_all)
gr, gn = rate(gpt_all)
print(f"  alloy (canonical + replication): {ar:.0f}% both over {an} episodes")
print(f"  GPT   (canonical + replication): {gr:.0f}% both over {gn} episodes")
print()
print(f"  {'Gemini comparator':52s} {'Gem both':>9} {'n':>5} {'Delta':>7} {'95% CI':>15}  sig")
out = []
for label, gem in gem_variants.items():
    r, n = rate(gem)
    if n == 0:
        print(f"  {label:52s}   (no data)")
        continue
    pt, lo, hi, nt = boot(alloy_all, gpt_all, gem)
    sig = 'YES' if (lo > 0 or hi < 0) else 'no'
    out.append(dict(comparator=label, gem_both=round(r, 1), gem_n=n,
                    delta=round(pt, 1), lo=round(lo, 1), hi=round(hi, 1),
                    n_tasks=nt, significant=(sig == 'YES')))
    print(f"  {label:52s} {r:8.0f}% {n:5d} {pt:+7.0f} [{lo:+6.0f},{hi:+6.0f}]  {sig}")

json.dump(out, open(REPO / 'reports/comparator_sensitivity.json', 'w'), indent=1)
print(f"\n[saved] reports/comparator_sensitivity.json")
