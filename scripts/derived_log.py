"""Stand-in for inspect_ai.log.read_eval_log over the released per-sample records.

The analysis scripts were written against raw Inspect .eval logs under logs/. The release ships,
for each log, a JSON record of the fields those scripts read (data/derived_logs/, written by
scripts/extract_derived_logs.py). eval_files(log_dir) lists the records for a raw log directory
(given as logs/...), and read_eval_log(path) returns an object with the attributes the scripts
use: samples (id, epoch, task names in metadata, error, scores[...].value, output.model, store,
model events' reasoning flags) and stats.model_usage. Sample order is the one read_eval_log()
returns.
"""
import json
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parent.parent
DERIVED = REPO / "data" / "derived_logs"


class EvalFile:
    """A derived record; .name is the original .eval file name, str() its path on disk."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name[: -len(".json")]

    def __fspath__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)


def eval_files(log_dir):
    """Records for the raw log directory log_dir (absolute, or relative to the repo root)."""
    p = Path(log_dir)
    rel = (p if not p.is_absolute() else p.relative_to(REPO)).relative_to("logs")
    files = [EvalFile(f) for f in (DERIVED / rel).glob("*.eval.json")]
    return sorted(files, key=lambda f: f.name)


def read_eval_log(path):
    rec = json.load(open(path))
    samples = [
        NS(
            id=s["id"],
            epoch=s["epoch"],
            metadata=s.get("metadata") or {},
            error=NS(message="(not retained)") if s["error"] else None,
            scores=(None if s["scores"] is None
                    else {k: NS(value=v) for k, v in s["scores"].items()}),
            output=NS(model=s["output_model"]),
            store=s["store"],
            events=[NS(event="model", config=NS(reasoning_history=h, reasoning_effort=e))
                    for h, e in s["model_call_flags"]],
        )
        for s in rec["samples"]
    ]
    usage = {m: NS(**u) for m, u in rec["model_usage"].items()}
    return NS(samples=samples, stats=NS(model_usage=usage), eval=NS(**rec["eval"]),
              status=rec["status"], location=rec["source_file"])
