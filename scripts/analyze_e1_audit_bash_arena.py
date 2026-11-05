"""E1: audit-before-execute vs standard alloy on bash_arena (sudo_edit), fixed harness.

Implements the analysis fixed in docs/precommitments/E1_PRECOMMIT.md.
Run: uv run --no-sync python scripts/analyze_e1_audit_bash_arena.py
Writes reports/e1_ba_audit.json
"""
import json
import random
import re
import sys
from pathlib import Path

from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from analyze_audit import _audit, _scores  # noqa: E402
from derived_log import eval_files, read_eval_log  # noqa: E402  released per-sample records

ARMS = {
    "standard": "logs/camera_ready/e1_ba_standard",
    "audit": "logs/camera_ready/e1_ba_audit",
}
PRIV = re.compile(r"sudo|sudoers|privilege|escalat", re.IGNORECASE)
SEED, N_BOOT = 20261010, 20_000


def arm(log_dir):
    rows, n_err, audits, vetoes = [], 0, 0, []
    for f in eval_files(log_dir):
        for s in read_eval_log(str(f)).samples or []:
            main, side = _scores(s) if s.error is None else (None, None)
            if main is None and side is None:
                n_err += 1
                continue
            rows.append((float(main or 0.0), int(bool(side))))
            decisions = _audit(s)
            audits += len(decisions)
            vetoes += [d for d in decisions if not d.get("approved", True)]
    n = len(rows)
    by_actor = {}
    for v in vetoes:
        by_actor[v.get("actor", "?")] = by_actor.get(v.get("actor", "?"), 0) + 1
    return dict(
        n=n, n_err=n_err,
        side_k=sum(r[1] for r in rows),
        main_mean=sum(r[0] for r in rows) / n,
        combined_mean=sum(r[0] * r[1] for r in rows) / n,
        audits=audits, vetoes=len(vetoes), vetoes_by_actor=by_actor,
        vetoes_naming_privilege=sum(bool(PRIV.search(v.get("reason", "") or "")) for v in vetoes),
        main_scores=[r[0] for r in rows],
    )


def main():
    res = {k: arm(v) for k, v in ARMS.items()}
    st, au = res["standard"], res["audit"]
    _, p_side = fisher_exact(
        [[au["side_k"], au["n"] - au["side_k"]], [st["side_k"], st["n"] - st["side_k"]]],
        alternative="less",
    )
    rng = random.Random(SEED)
    diffs = sorted(
        sum(rng.choices(au["main_scores"], k=au["n"])) / au["n"]
        - sum(rng.choices(st["main_scores"], k=st["n"])) / st["n"]
        for _ in range(N_BOOT)
    )
    out = dict(
        arms={k: {kk: vv for kk, vv in v.items() if kk != "main_scores"} for k, v in res.items()},
        side_fisher_one_sided_p=p_side,
        main_diff=au["main_mean"] - st["main_mean"],
        main_diff_ci95=[diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]],
    )
    for k, v in out["arms"].items():
        print(f"{k:8s} n={v['n']} err={v['n_err']} side={v['side_k']}/{v['n']} "
              f"main={v['main_mean']:.3f} comb={v['combined_mean']:.3f} "
              f"vetoes={v['vetoes']}/{v['audits']} priv={v['vetoes_naming_privilege']} {v['vetoes_by_actor']}")
    print(f"side one-sided Fisher p={p_side:.2e}; main diff {out['main_diff']:+.3f} "
          f"[{out['main_diff_ci95'][0]:+.3f}, {out['main_diff_ci95'][1]:+.3f}]")
    json.dump(out, open(REPO / "reports/e1_ba_audit.json", "w"), indent=1)
    print("[saved] reports/e1_ba_audit.json")


if __name__ == "__main__":
    main()
