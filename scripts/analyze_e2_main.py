"""E2 stage 2: composition (P1) and uplift (P2) on the selected new pair, Bash tasks 0-99.

Analysis fixed in docs/precommitments/E2_PRECOMMIT.md. Logs: logs/camera_ready/e2_main/.
Run: uv run --no-sync python scripts/analyze_e2_main.py
Writes reports/e2_main.json
"""
import json
import random
import sys
from pathlib import Path

from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import analyze_phase2 as ph2  # noqa: E402
from derived_log import eval_files, read_eval_log  # noqa: E402  released per-sample records

MAIN = REPO / "logs/camera_ready/e2_main"
SEED, NB = 20261010, 20_000


def counts(log_dir):
    """(n, main_k, side_k, side_k among main failures, main failures)."""
    n = mk = sk = sk_fail = n_fail = 0
    for f in eval_files(log_dir):
        for s in read_eval_log(str(f)).samples or []:
            if s.error is not None:
                continue
            main, side = ph2._scores(s)
            if main is None and side is None:
                continue
            m, sd = float(main or 0.0) > 0.5, bool(side)
            n += 1
            mk += m
            sk += sd
            if not m:
                n_fail += 1
                sk_fail += sd
    return dict(n=n, main_k=mk, side_k=sk, side_given_main_fail=[sk_fail, n_fail])


def main():
    screen = json.load(open(REPO / "reports/e2_screen.json"))["selected"]
    arms = {"alloy": MAIN / "alloy", "solo_a": MAIN / "solo_a", "solo_b": MAIN / "solo_b"}
    per_task, stats, cnt = {}, {}, {}
    for k, d in arms.items():
        per_task[k], stats[k] = ph2.load_arm(d)
        cnt[k] = counts(d)

    al, a, b = cnt["alloy"], cnt["solo_a"], cnt["solo_b"]
    _, p_side = fisher_exact([[al["side_k"], al["n"] - al["side_k"]],
                              [a["side_k"], a["n"] - a["side_k"]]], alternative="greater")
    _, p_main = fisher_exact([[al["main_k"], al["n"] - al["main_k"]],
                              [b["main_k"], b["n"] - b["main_k"]]], alternative="greater")

    random.seed(SEED)
    ph2.NB = NB
    uplift = ph2.bootstrap(per_task["alloy"], per_task["solo_a"], per_task["solo_b"],
                           resample_episodes=True)

    out = dict(pair=screen, stats=stats, counts=cnt,
               P1=dict(alloy_side_gt_A_p=p_side, alloy_main_gt_B_p=p_main,
                       replicates=bool(p_side < 0.05 and p_main < 0.05)),
               P2=uplift, bootstrap_replicates=NB, seed=SEED)
    for k in arms:
        s = stats[k]
        print(f"{k:7s} n={s['n_scored']} err={s['n_errored']} main={s['main']} "
              f"side={s['side']} comb={s['combined']}")
    print(f"P1: side alloy>A p={p_side:.2e}; main alloy>B p={p_main:.2e}; "
          f"replicates={out['P1']['replicates']}")
    print(f"P2: alloy - per-task best solo = {uplift}")
    json.dump(out, open(REPO / "reports/e2_main.json", "w"), indent=1)
    print("[saved] reports/e2_main.json")


if __name__ == "__main__":
    main()
