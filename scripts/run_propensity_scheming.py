"""Run PropensityBench + Scheming: solo baselines + alloy comparison.

Runs each benchmark with epochs=5 for statistical power:
- PropensityBench: 6 scenarios × 5 epochs = 30 samples per config
- Scheming: 3 scenarios × 5 epochs = 15 samples per config
"""

import os
import sys

# Load .env
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())

from inspect_ai import eval

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.propensity import propensity_alloy
from experiments.scheming import scheming_alloy

GPT_MODEL = "openai/gpt-5-mini-2025-08-07"
GEMINI_MODEL = "google/gemini-3.1-flash-lite-preview"

MAX_TURNS = 5
EPOCHS = 5


def run_config(name, task, model, log_dir):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    results = eval(
        task,
        model=model,
        log_dir=log_dir,
        fail_on_error=False,
    )
    print(f"{name} done.")
    return results


def main():
    log_dir = "traces/"

    # --- PropensityBench (6 scenarios × 5 epochs = 30 samples each) ---

    # 1. GPT-5-mini solo
    run_config(
        "PropensityBench — GPT-5-mini solo (epochs=5)",
        propensity_alloy(models=None, max_turns=MAX_TURNS, epochs=EPOCHS),
        GPT_MODEL,
        log_dir,
    )

    # 2. Gemini 3 Flash solo
    run_config(
        "PropensityBench — Gemini 3 Flash solo (epochs=5)",
        propensity_alloy(models=None, max_turns=MAX_TURNS, epochs=EPOCHS),
        GEMINI_MODEL,
        log_dir,
    )

    # 3. Alloy GPT + Gemini
    run_config(
        "PropensityBench — Alloy GPT-5-mini + Gemini 3 Flash (epochs=5)",
        propensity_alloy(
            models=[GPT_MODEL, GEMINI_MODEL],
            strategy="round_robin",
            max_turns=MAX_TURNS,
            epochs=EPOCHS,
        ),
        GPT_MODEL,
        log_dir,
    )

    # --- Scheming (3 scenarios × 5 epochs = 15 samples each) ---

    # 4. GPT-5-mini solo
    run_config(
        "Scheming — GPT-5-mini solo (epochs=5)",
        scheming_alloy(models=None, max_turns=MAX_TURNS, epochs=EPOCHS),
        GPT_MODEL,
        log_dir,
    )

    # 5. Gemini 3 Flash solo
    run_config(
        "Scheming — Gemini 3 Flash solo (epochs=5)",
        scheming_alloy(models=None, max_turns=MAX_TURNS, epochs=EPOCHS),
        GEMINI_MODEL,
        log_dir,
    )

    # 6. Alloy GPT + Gemini
    run_config(
        "Scheming — Alloy GPT-5-mini + Gemini 3 Flash (epochs=5)",
        scheming_alloy(
            models=[GPT_MODEL, GEMINI_MODEL],
            strategy="round_robin",
            max_turns=MAX_TURNS,
            epochs=EPOCHS,
        ),
        GPT_MODEL,
        log_dir,
    )

    print("\n" + "=" * 60)
    print("  All runs complete! Check traces/ for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
