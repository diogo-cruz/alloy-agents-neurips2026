"""ImpossibleBench (one-off split): GPT-5-mini solo, Gemini 3.1 Flash Lite solo, and their
round-robin alloy, 30 tasks each, in Docker sandboxes (compose.yaml at the repository root; run
from the repository root). The paper's Default column pools a first 10-task run of the same
configurations with this one. Needs the ImpossibleBench package (see README).

Run: uv run python scripts/run_impossiblebench.py
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
env_path = REPO / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(REPO))
from inspect_ai import eval as inspect_eval
from experiments.impossible_bench import impossible_bench_alloy

GPT_MODEL = "openai/gpt-5-mini-2025-08-07"
GEMINI_MODEL = "google/gemini-3.1-flash-lite-preview"
LOG_DIR = str(REPO / "logs/impossiblebench")
LIMIT = 30
MAX_TURNS = 8


def run_one(name, models_arg, primary_model):
    print(f"\n=== {name} ===", flush=True)
    task = impossible_bench_alloy(
        models=models_arg,
        strategy="round_robin" if models_arg else None,
        split="oneoff",
        limit=LIMIT,
        max_turns=MAX_TURNS,
    )
    results = inspect_eval(
        task,
        model=primary_model,
        log_dir=LOG_DIR,
        sandbox="docker",
        fail_on_error=False,
    )
    if results:
        r = results[0]
        print(f"  {name} results: {r.results}", flush=True)
    print(f"  {name} done.", flush=True)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"[impossiblebench] Starting; LIMIT={LIMIT}", flush=True)

    # True solos: pass models=None and let eval(model=...) drive a solo run.
    run_one("GPT-5-mini solo", None, GPT_MODEL)
    run_one("Gemini-Lite solo", None, GEMINI_MODEL)
    # Real heterogeneous alloy.
    run_one("Alloy GPT+Gemini-Lite", [GPT_MODEL, GEMINI_MODEL], GPT_MODEL)

    print("[impossiblebench] All runs complete.", flush=True)


if __name__ == "__main__":
    main()
