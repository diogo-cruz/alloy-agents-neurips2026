#!/usr/bin/env python3
"""Primary confirmatory analysis of the primary contrast (Bash, GPT+Gemini), canonical pool.

Computes:
  1. Inclusion-rule robustness for the primary Bash GPT+Gemini contrast.
  2. Hierarchical (two-level) paired bootstrap with n_tasks reported and the
     comparator computed as alloy - max(solo A, solo B) *per replicate*.
  3. A mixed-effects logistic model (task random intercept) as a parametric check.

Run: uv run --no-sync python scripts/primary_analysis.py
Writes reports/primary_analysis.json
"""
import json
import math
import random
from collections import defaultdict

random.seed(20260728)

REPO = __import__('pathlib').Path(__file__).resolve().parents[1]
INV = json.load(open(REPO / 'reports/trace_inventory.json'))['traces']
NB = 20000

PRETTY = {'GPT': 'GPT', 'Gemini-Lite': 'Gemini', 'Grok': 'Grok',
          'GPT+Gemini-Lite': 'GPT+Gemini', 'GPT+Grok': 'GPT+Grok',
          'Gemini-Lite+Grok': 'Gemini+Grok'}
ALLOYS = {'GPT+Gemini-Lite': ('GPT', 'Gemini-Lite'),
          'GPT+Grok': ('GPT', 'Grok'),
          'Gemini-Lite+Grok': ('Gemini-Lite', 'Grok')}


def reasoning_ok(r):
    """Paper-primary condition: alloys use C1 (history stripped inherently),
    solos use C2 (reasoning-matched). Gemini-Lite is degenerate, so pool it."""
    if r['config_name'] == 'Gemini-Lite':
        return True
    return r['reasoning'] == ('C1' if r['is_alloy'] else 'C2')


def by_task(setting, cfg, date_prefix=None, single_file=False):
    """task_id -> list of per-episode combined-success values."""
    d = defaultdict(list)
    chosen = None
    for r in INV:
        if r['setting'] != setting or r['config_name'] != cfg or not reasoning_ok(r):
            continue
        if date_prefix and not r['created'].startswith(date_prefix):
            continue
        if single_file:
            # keep only the largest single eval file, for a strict single-run rule
            if chosen is None or r['n_scored'] > chosen['n_scored']:
                chosen = r
            continue
        for s in r['samples']:
            ss = s.get('side_score')
            if ss is None:
                continue
            ms = s.get('main_score')
            m = 1.0 if ms == 'C' else (0.0 if ms == 'I' else (float(ms) if isinstance(ms, (int, float)) else 0.0))
            d[s.get('main_task', '?')].append(m * (1 if ss == 'C' else 0))
    if single_file and chosen is not None:
        for s in chosen['samples']:
            ss = s.get('side_score')
            if ss is None:
                continue
            ms = s.get('main_score')
            m = 1.0 if ms == 'C' else (0.0 if ms == 'I' else (float(ms) if isinstance(ms, (int, float)) else 0.0))
            d[s.get('main_task', '?')].append(m * (1 if ss == 'C' else 0))
    return d


def hier_bootstrap(A, SA, SB, per_replicate_max=True):
    """Two-level paired bootstrap of alloy - best solo.

    Resamples tasks with replacement AND episodes within each task. When
    per_replicate_max is True the comparator is recomputed as max(soloA, soloB)
    inside each replicate, which is the conservative choice (the max of two noisy
    means is upward biased, so it makes the alloy's gain harder to establish)."""
    tasks = [t for t in A if (t in SA and SA[t]) or (t in SB and SB[t])]
    tasks = [t for t in tasks if A[t]]
    if not tasks:
        return None
    nt = len(tasks)

    def tmean(D, t):
        return sum(D[t]) / len(D[t]) if (t in D and D[t]) else 0.0

    def resample(D, t):
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
            da += resample(A, t)
            if per_replicate_max:
                db += max(resample(SA, t), resample(SB, t))
            else:
                db += max(tmean(SA, t), tmean(SB, t))
        diffs.append(100 * (da - db) / nt)
    diffs.sort()
    return dict(point=round(point, 1), lo=round(diffs[int(.025 * NB)], 1),
                hi=round(diffs[int(.975 * NB)], 1), n_tasks=nt)


OUT = {}

# ---------------------------------------------------------------------------
# 1. Inclusion-rule robustness, primary pair (Bash GPT+Gemini)
# ---------------------------------------------------------------------------
print("=" * 74)
print("1. INCLUSION-RULE ROBUSTNESS — primary pair, Bash GPT+Gemini")
print("   (comparator = max(GPT, Gemini) recomputed per bootstrap replicate)")
print("=" * 74)
rules = [
    ("canonical pool (all reasoning-matched runs)", dict()),
    ("date-matched: 2026-03-14 only", dict(date_prefix='2026-03-14')),
    ("single largest eval file per config", dict(single_file=True)),
]
OUT['inclusion_robustness'] = []
print(f"{'rule':44s} {'Δpp':>6} {'95% CI':>16} {'n_tasks':>8}")
for label, kw in rules:
    A = by_task('bash', 'GPT+Gemini-Lite', **kw)
    SA = by_task('bash', 'GPT', **kw)
    SB = by_task('bash', 'Gemini-Lite', **kw)
    r = hier_bootstrap(A, SA, SB)
    if not r:
        print(f"{label:44s}  (no data)")
        continue
    OUT['inclusion_robustness'].append(dict(rule=label, **r))
    print(f"{label:44s} {r['point']:+6.0f} [{r['lo']:+5.0f},{r['hi']:+5.0f}] {r['n_tasks']:8d}")

# ---------------------------------------------------------------------------
# 2. Full hierarchical bootstrap, all cells, with n_tasks
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("2. HIERARCHICAL BOOTSTRAP — all nine cells, canonical pool")
print("=" * 74)
print(f"{'setting':11s} {'alloy':13s} {'Δpp':>6} {'95% CI':>16} {'n_tasks':>8}  sig")
OUT['hierarchical'] = []
for setting in ['bash', 'iac_fast', 'bash_arena']:
    for alloy, (sa, sb) in ALLOYS.items():
        A = by_task(setting, alloy)
        SA = by_task(setting, sa)
        SB = by_task(setting, sb)
        r = hier_bootstrap(A, SA, SB)
        if not r:
            continue
        sig = (r['lo'] > 0 or r['hi'] < 0)
        OUT['hierarchical'].append(dict(setting=setting, alloy=PRETTY[alloy], sig=sig, **r))
        print(f"{setting:11s} {PRETTY[alloy]:13s} {r['point']:+6.0f} "
              f"[{r['lo']:+5.0f},{r['hi']:+5.0f}] {r['n_tasks']:8d}  {'YES' if sig else 'no'}")

# ---------------------------------------------------------------------------
# 3. Mixed-effects logistic model (task random intercept), primary pair
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("3. MIXED-EFFECTS LOGISTIC — primary pair, Bash, task random intercept")
print("=" * 74)


def fit_mixed_logit(rows, n_quad=25, iters=300):
    """Logistic regression with a task-level random intercept, fitted by
    Gauss-Hermite marginal likelihood on a grid over (beta0, beta1, sigma).
    rows: list of (task, x, y) with x=1 for alloy, 0 for solo comparator."""
    import itertools
    tasks = sorted({t for t, _, _ in rows})
    by_t = {t: [(x, y) for tt, x, y in rows if tt == t] for t in tasks}
    # Gauss-Hermite nodes
    gh_x, gh_w = np_hermgauss(n_quad)

    def negll(b0, b1, sigma):
        total = 0.0
        for t in tasks:
            acc = 0.0
            for xi, wi in zip(gh_x, gh_w):
                u = math.sqrt(2.0) * sigma * xi
                lp = 0.0
                for x, y in by_t[t]:
                    eta = b0 + b1 * x + u
                    p = 1.0 / (1.0 + math.exp(-max(-30, min(30, eta))))
                    p = min(max(p, 1e-12), 1 - 1e-12)
                    lp += y * math.log(p) + (1 - y) * math.log(1 - p)
                acc += (wi / math.sqrt(math.pi)) * math.exp(lp)
            total += math.log(max(acc, 1e-300))
        return -total

    best = None
    for b0 in [x * 0.5 for x in range(-10, 7)]:
        for b1 in [x * 0.5 for x in range(-4, 17)]:
            for sg in [0.25, 0.5, 1.0, 1.5, 2.0]:
                v = negll(b0, b1, sg)
                if best is None or v < best[0]:
                    best = (v, b0, b1, sg)
    # refine b1 on a fine grid at the best (b0, sigma)
    _, b0, b1, sg = best
    fine = None
    for bb in [b1 - 0.5 + i * 0.05 for i in range(21)]:
        v = negll(b0, bb, sg)
        if fine is None or v < fine[0]:
            fine = (v, bb)
    b1 = fine[1]
    # likelihood-ratio test against b1 = 0
    ll_full = -negll(b0, b1, sg)
    b0_null = min([x * 0.25 for x in range(-20, 13)], key=lambda z: negll(z, 0.0, sg))
    ll_null = -negll(b0_null, 0.0, sg)
    lr = 2 * (ll_full - ll_null)
    return b1, sg, lr


def np_hermgauss(n):
    import numpy as np
    x, w = np.polynomial.hermite.hermgauss(n)
    return list(x), list(w)


def chi2_sf_1df(x):
    return math.erfc(math.sqrt(max(x, 0) / 2.0))


rows = []
A = by_task('bash', 'GPT+Gemini-Lite')
SA = by_task('bash', 'GPT')
SB = by_task('bash', 'Gemini-Lite')
for t in A:
    for v in A[t]:
        rows.append((t, 1, 1 if v > 0.5 else 0))
for D in (SA, SB):
    for t in D:
        for v in D[t]:
            rows.append((t, 0, 1 if v > 0.5 else 0))
b1, sigma, lr = fit_mixed_logit(rows)
p = chi2_sf_1df(lr)
odds = math.exp(b1)
print(f"  n episodes = {len(rows)}, n tasks = {len({t for t,_,_ in rows})}")
print(f"  alloy fixed effect (log-odds) = {b1:+.2f}  -> odds ratio {odds:.1f}x")
print(f"  task random-intercept sd      = {sigma:.2f}")
print(f"  likelihood-ratio test vs beta=0: LR = {lr:.1f}, p {'< 0.001' if p < 0.001 else f'= {p:.4f}'}")
OUT['mixed_effects'] = dict(beta_alloy=round(b1, 3), odds_ratio=round(odds, 2),
                           task_sd=round(sigma, 3), lr_stat=round(lr, 2),
                           p_value=p, n_episodes=len(rows),
                           n_tasks=len({t for t, _, _ in rows}),
                           note="Combined success (binary) ~ alloy + (1|task); comparator = both solos pooled.")

json.dump(OUT, open(REPO / 'reports/primary_analysis.json', 'w'), indent=1)
print("\n[saved] reports/primary_analysis.json")
