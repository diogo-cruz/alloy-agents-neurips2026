"""Alloy, audit-before-execute alloy and solo runs on ControlArena Bash or bash_arena.

Runs the standard round-robin alloy, the audit-before-execute alloy, or a solo model, through
OpenRouter by default (GPT-5-mini + Gemini 3.1 Flash Lite unless --models is given; for a
Gemini solo, --route direct uses the Google API). Used for the July and August 2026 reruns,
the audit-before-execute experiments (Bash and E1) and the new-pair experiment (E2).

Usage:
  uv run --no-sync python scripts/run_audit_bash.py --setting bash --mode audit --limit 20
  # any OpenRouter pair or solo model, on a task range (start,stop):
  uv run --no-sync python scripts/run_audit_bash.py --mode solo --models openrouter/x/y --limit 100,120
  uv run --no-sync python scripts/run_audit_bash.py --setting bash_arena --side-task sudo_edit --mode audit --limit 15
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
env_path = REPO / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        if v.strip():
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from inspect_ai import eval as inspect_eval
from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task

from control_arena.policy import untrusted_policy_builder
from experiments.control_arena_alloy import (
    build_alloy_protocol,
    build_alloy_audit_protocol,
    alloy_attack_policy_builder,
    alloy_honest_policy_builder,
)
from run_control_arena_unified import build_setting, SETTING_DEFAULTS

# OpenRouter-routed primary pair (single key covers both)
GPT_OR = "openrouter/openai/gpt-5-mini"
GEMINI_OR = "openrouter/google/gemini-3.1-flash-lite-preview"
# Native-provider route (the paper's March serving stack; needs GOOGLE_API_KEY)
GEMINI_DIRECT = "google/gemini-3.1-flash-lite-preview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standard", "audit", "solo"], default="standard")
    ap.add_argument("--solo", choices=["gpt", "gemini"], default=None,
                    help="which model to run alone (--mode solo)")
    ap.add_argument("--setting", default="bash", help="bash or bash_arena")
    ap.add_argument("--side-task", default=None, help="bash_arena side task (default sudo_edit)")
    ap.add_argument("--limit", default="1",
                    help="number of tasks, or a start,stop range of task indices")
    ap.add_argument("--models", default=None,
                    help="comma-separated model slugs replacing the GPT+Gemini pair "
                         "(one slug with --mode solo)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log-dir", default="logs/audit_experiment")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap concurrent samples/sandboxes; each sample holds one docker "
                         "compose network, and the default address pools allow ~29 total")
    ap.add_argument("--route", choices=["openrouter", "direct"], default="openrouter",
                    help="API route for --mode solo --solo gemini: OpenRouter (default) or "
                         "Inspect's native Google provider (the paper's March serving stack)")
    args = ap.parse_args()

    if args.route == "direct" and not (args.mode == "solo" and args.solo == "gemini"):
        raise SystemExit("--route direct is only supported for --mode solo --solo gemini")
    if args.route == "direct" and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set in .env")
    if args.route == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set in .env")

    limit = (tuple(int(x) for x in args.limit.split(",")) if "," in args.limit
             else int(args.limit))
    if args.models:
        if args.route == "direct":
            raise SystemExit("--models is OpenRouter-only")
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if args.mode == "solo" and len(models) != 1:
            raise SystemExit("--mode solo takes exactly one model in --models")
        if args.mode != "solo" and len(models) < 2:
            raise SystemExit("--mode standard/audit needs at least two models")
    elif args.mode == "solo":
        if not args.solo:
            raise SystemExit("--mode solo requires --solo gpt|gemini or --models")
        if args.solo == "gemini":
            models = [GEMINI_DIRECT if args.route == "direct" else GEMINI_OR]
        else:
            models = [GPT_OR]
    else:
        models = [GPT_OR, GEMINI_OR]
    if args.setting == "bash_arena":
        setting = build_setting("bash_arena", side_tasks=[args.side_task or "sudo_edit"])
    else:
        setting = build_setting(args.setting)

    d = SETTING_DEFAULTS.get(args.setting, {})
    cfg_kwargs = dict(
        untrusted_model=models[0],
        trusted_model=models[0],
        limit=limit,
        max_steps=d.get("max_steps", 30),
        time_limit=d.get("time_limit", 600),
        log_dir=args.log_dir,
        fail_on_error=False,
        epochs=args.epochs,
    )
    if "max_tasks" in d:
        cfg_kwargs["max_tasks"] = d["max_tasks"]
    if args.max_samples:
        cfg_kwargs["max_samples"] = args.max_samples
        cfg_kwargs["max_sandboxes"] = args.max_samples
    config = ControlEvalConfig(**cfg_kwargs)

    # strict_tools=False: OpenRouter routes GPT-5-mini to OpenAI, whose strict tool
    # schema validation rejects ControlArena's text_editor tool. The native Google
    # provider takes no such argument.
    model_args = {} if args.route == "direct" else {"strict_tools": False}
    if args.mode == "standard":
        protocol = build_alloy_protocol(
            models=models, strategy="round_robin", model_args=model_args
        )
    elif args.mode == "solo":
        # One model in the rotation means no swaps ever occur, so this is a solo
        # run. It goes through the same attack prompt and the same reasoning
        # stripping as the alloy, which matches the history-stripped condition
        # used for the solo baselines in the paper, and it lets us pass
        # strict_tools through for the OpenRouter route.
        protocol = untrusted_policy_builder(
            attack=alloy_attack_policy_builder(
                models=models, strategy="round_robin", model_args=model_args
            ),
            honest=alloy_honest_policy_builder(
                models=models, strategy="round_robin", model_args=model_args
            ),
        )
    else:
        protocol = build_alloy_audit_protocol(
            models=models, strategy="round_robin", model_args=model_args
        )

    task = get_control_task(
        setting=setting,
        micro_protocol=protocol,
        mode=EvalMode.ATTACK,
        eval_config=config,
    )

    print(f"=== setting={args.setting} mode={args.mode} limit={args.limit} "
          f"side={args.side_task or ('sudo_edit' if args.setting=='bash_arena' else 'malware')} ===")
    results = inspect_eval(task, model=models[0], **config.eval_args())
    if results:
        print("results:", results[0].results)


if __name__ == "__main__":
    main()
