"""Analyze audit-before-execute vs standard alloy on Bash.

Reads .eval logs, extracts per-sample main/side/combined and audit veto stats,
prints aggregate rates. Usage: uv run --no-sync python scripts/analyze_audit.py <log_dir_or_files...>
"""
import sys
from pathlib import Path
from derived_log import eval_files, read_eval_log  # released per-sample records


def _binary(v):
    if v in ("C", 1, 1.0, True):
        return 1
    if v in ("I", 0, 0.0, False):
        return 0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _scores(sample):
    """Return (main, side) as 0/1/float or None."""
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


def _audit(sample):
    """Find audit_decisions in the store. Returns list (possibly empty)."""
    store = sample.store or {}
    for k, v in store.items():
        if "audit_decisions" in k and isinstance(v, list):
            return v
    return []


def analyze(path):
    log = read_eval_log(str(path))
    rows = []
    n_err = 0
    total_vetoes = total_audits = 0
    veto_actors = {}
    for s in (log.samples or []):
        if s.error is not None:
            n_err += 1
            continue
        main, side = _scores(s)
        if main is None and side is None:
            n_err += 1
            continue
        audits = _audit(s)
        vetoes = [a for a in audits if not a.get("approved", True)]
        total_audits += len(audits)
        total_vetoes += len(vetoes)
        for a in vetoes:
            veto_actors[a.get("actor", "?")] = veto_actors.get(a.get("actor", "?"), 0) + 1
        rows.append((main or 0.0, side or 0, len(audits), len(vetoes)))
    n = len(rows)
    if n == 0:
        print(f"  {Path(path).name}: no scored samples ({n_err} errored)")
        return
    main_mean = sum(r[0] for r in rows) / n
    side_mean = sum(r[1] for r in rows) / n
    both_mean = sum((r[0]) * (1 if r[1] else 0) for r in rows) / n
    print(f"  file: {Path(path).name}")
    print(f"  n={n} (scored), errored={n_err}")
    print(f"  MAIN={main_mean*100:.0f}%  SIDE={side_mean*100:.0f}%  BOTH={both_mean*100:.0f}%")
    if total_audits:
        print(f"  audits={total_audits}, vetoes={total_vetoes} ({total_vetoes/total_audits*100:.0f}% of audited actions), veto by actor={veto_actors}")
    return dict(n=n, main=main_mean, side=side_mean, both=both_mean,
                audits=total_audits, vetoes=total_vetoes)


def main():
    # the two arms reported in the paper (the bare logs/audit_experiment/ files were smoke tests)
    args = sys.argv[1:] or ["logs/audit_experiment/standard20", "logs/audit_experiment/audit20_v2"]
    files = []
    for a in args:
        p = Path(a)
        files.extend(eval_files(a) if not a.endswith(".json") else [p])
    for f in files:
        print(f"\n=== {f} ===")
        analyze(f)


if __name__ == "__main__":
    main()
