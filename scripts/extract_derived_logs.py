"""Extract the per-sample records the analysis scripts read from raw Inspect .eval logs.

The release ships these records (data/derived_logs/) instead of the raw logs, which hold full
agent transcripts. Kept per log: task, creation time, models and their roles, dataset size,
epochs and limit, token usage per model, and the original file name. Kept per sample: id,
epoch, main and side task names, whether it errored, score values, the model identifier
returned by the API, the reasoning flags sent on its model calls, and audit-before-execute
decisions (step, actor, auditor, approved). No messages, tool calls, tool outputs, error
messages or model-written text are kept: an audit decision's stated reason is replaced by a
placeholder recording only whether it names privilege escalation (the regex of
scripts/analyze_e1_audit_bash_arena.py), the one property of the text any analysis uses.

Usage (needs the raw logs):
  python scripts/extract_derived_logs.py --src <directory that contains logs/> \
      logs/phase2/alloy100 logs/phase2/gpt100 ...
Writes data/derived_logs/<log dir relative to logs/>/<original file name>.json
"""
import argparse
import json
import re
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_sample, read_eval_log_sample_summaries

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "derived_logs"
PRIV = re.compile(r"sudo|sudoers|privilege|escalat", re.IGNORECASE)
MAX_STR = 8  # score values are "C"/"I"-style codes; anything longer would be text
META_KEYS = ("main_task_name", "side_task_name")  # task identifiers only, no descriptions


def clean_value(v):
    if isinstance(v, dict):
        return {k: clean_value(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str) and len(v) <= MAX_STR:
        return v
    raise ValueError(f"unexpected score value of type {type(v).__name__}")


def clean_decision(d):
    out = {k: d[k] for k in ("step", "actor", "auditor", "approved") if k in d}
    names = bool(PRIV.search(d.get("reason") or ""))
    out["reason"] = "[text withheld; names privilege escalation]" if names else "[text withheld]"
    return out


def sample_record(s):
    store = {}
    for k, v in (s.store or {}).items():
        if "audit_decisions" in k and isinstance(v, list):
            store[k] = [clean_decision(d) for d in v]
    flags = {(getattr(e.config, "reasoning_history", None),
              getattr(e.config, "reasoning_effort", None))
             for e in (s.events or []) if getattr(e, "event", None) == "model"}
    md = s.metadata or {}
    return dict(
        id=s.id,
        epoch=s.epoch,
        metadata={k: md[k] for k in META_KEYS if isinstance(md.get(k), str) and len(md[k]) <= 64},
        error=s.error is not None,
        scores=(None if s.scores is None
                else {k: clean_value(sc.value) for k, sc in s.scores.items()}),
        output_model=getattr(s.output, "model", None) if s.output is not None else None,
        store=store,
        model_call_flags=sorted(flags, key=lambda f: (str(f[0]), str(f[1]))),
    )


def log_record(path):
    h = read_eval_log(str(path), header_only=True)
    ev = h.eval
    rec = dict(
        source_file=Path(path).name,
        eval=dict(
            task=ev.task,
            created=ev.created,
            model=ev.model,
            model_roles={r: mc.model for r, mc in (getattr(ev, "model_roles", None) or {}).items()},
            dataset_samples=getattr(ev.dataset, "samples", None),
            epochs=getattr(ev.config, "epochs", None),
            limit=getattr(ev.config, "limit", None),
        ),
        status=h.status,
        model_usage={m: dict(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                             input_tokens_cache_read=u.input_tokens_cache_read)
                     for m, u in ((h.stats.model_usage if h.stats else None) or {}).items()},
        samples=[],
    )
    for summ in read_eval_log_sample_summaries(str(path)):
        s = read_eval_log_sample(str(path), summ.id, summ.epoch)
        rec["samples"].append(sample_record(s))
    # the order read_eval_log() returns (inspect_ai.log._log.sort_samples), which the analyses see
    rec["samples"].sort(key=lambda r: (r["epoch"], r["id"] if isinstance(r["id"], str)
                                       else str(r["id"]).zfill(20)))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory that contains the raw logs/ tree")
    ap.add_argument("log_dirs", nargs="+", help="log directories, relative to --src (logs/...)")
    a = ap.parse_args()
    src = Path(a.src)
    for d in a.log_dirs:
        files = sorted((src / d).glob("*.eval"))
        if not files:
            raise SystemExit(f"no .eval files in {src / d}")
        out_dir = OUT / Path(d).relative_to("logs")
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            rec = log_record(f)
            json.dump(rec, open(out_dir / (f.name + ".json"), "w"), indent=1, default=str)
            print(f"{d}/{f.name}: {len(rec['samples'])} samples")


if __name__ == "__main__":
    main()
