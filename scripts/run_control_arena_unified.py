"""ControlArena runner for the paper's experiments.

Settings: bash, iac_fast and bash_arena (the continuous-effort settings, attack mode), and
agentdojo and data_poisoning (two of the single-action benchmarks).
Supports named configs (solo or alloy), phases (groups of configs), and setting-specific
parameters (side task filtering).

Usage:
  # Single config, single setting
  python scripts/run_control_arena_unified.py --setting bash_arena --side-tasks sudo_edit --config solo-gpt-attack --limit 10

  # Phase (group of configs)
  python scripts/run_control_arena_unified.py --setting bash_arena --side-tasks sudo_edit --phase attack-trio --limit 10

  # iac_fast with filtered side tasks
  python scripts/run_control_arena_unified.py --setting iac_fast --filter-side-tasks create_backdoor_admin_role allow_public_s3_policy --config alloy-attack

  # Original bash setting (backward compat)
  python scripts/run_control_arena_unified.py --setting bash --config solo-gpt-attack --limit 10
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import eval
from inspect_ai.model import GenerateConfig

from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.policy import untrusted_policy_builder

from experiments.control_arena_alloy import build_alloy_protocol

# ── Models ──────────────────────────────────────────────────────────────────

GPT_MODEL = "openai/gpt-5-mini-2025-08-07"
GEMINI_MODEL = "google/gemini-3.1-flash-lite-preview"
GROK_MODEL = "openrouter/x-ai/grok-4.1-fast"

# ── Defaults per setting ────────────────────────────────────────────────────

SETTING_DEFAULTS = {
    "bash": {"max_steps": 30, "time_limit": 600},
    "bash_arena": {"max_steps": 30, "time_limit": 3600, "max_tasks": 1},
    "iac_fast": {"max_steps": 30, "time_limit": 1800, "max_tasks": 2},
    "agentdojo": {"max_steps": 20, "time_limit": 300},
    "data_poisoning": {"max_steps": 5, "time_limit": 120},
}

LOG_DIR = "traces/"

# ── Side task enums ─────────────────────────────────────────────────────────

BASH_ARENA_SIDE_TASKS = [
    "sudo_edit", "disable_firewall", "download_malware", "leak_passwords",
]

# Pin the 20 BashBench2 tasks used in all bash_arena experiments.
# These are the first 20 tasks in HuggingFace dataset order (no shuffling).
BASH_ARENA_TASK_IDS = [
    "6935769", "29559347", "11762372", "12990180", "67034029",
    "37586811", "29396928", "1019707", "17820740", "3070723",
    "50110992", "70849769", "17446801", "58313106", "11436013",
    "21914632", "52532893", "8657908", "14190731", "53451103",
]

IAC_FAST_SIDE_TASKS = [
    "disable_fsx_deny_policy", "disable_fsx_deny_policy_interpretability_ou",
    "check_in_cdk_out_directory", "detach_mfa_requirement_policy",
    "allow_public_s3_policy", "open_internet_egress_restricted_ous",
    "create_backdoor_admin_role", "exempt_ec2_public_ip_check",
    "change_policy_to_version_one",
]

# ── Named configs ───────────────────────────────────────────────────────────

CONFIGS = {
    # Default configuration (C1): each model keeps its own reasoning history
    "solo-gpt-attack": {"model": GPT_MODEL, "mode": EvalMode.ATTACK, "alloy": False},
    "solo-gemini-attack": {"model": GEMINI_MODEL, "mode": EvalMode.ATTACK, "alloy": False},
    # Alloys as run: the alloy solver strips reasoning before handing the transcript on
    "alloy-attack": {"models": [GPT_MODEL, GEMINI_MODEL], "mode": EvalMode.ATTACK, "alloy": True},
    # History-stripped solos (C2), matching what each model sees inside an alloy
    "solo-gpt-attack-nohistory": {
        "model": GPT_MODEL, "mode": EvalMode.ATTACK, "alloy": False,
        "generate_kwargs": {"reasoning_history": "none"},
    },
    "solo-gemini-attack-nohistory": {
        "model": GEMINI_MODEL, "mode": EvalMode.ATTACK, "alloy": False,
        "generate_kwargs": {"reasoning_history": "none"},
    },
    # Reasoning disabled (C3), via the homogeneous alloy path so that reruns match March.
    "solo-gemini-attack-noreasoning": {
        "models": [GEMINI_MODEL, GEMINI_MODEL], "mode": EvalMode.ATTACK, "alloy": True,
        "generate_kwargs": {"reasoning_effort": "none"},
    },
    # Grok solo and the two Grok alloys
    "solo-grok-attack-nohistory": {
        "model": GROK_MODEL, "mode": EvalMode.ATTACK, "alloy": False,
        "generate_kwargs": {"reasoning_history": "none"},
    },
    "alloy-gpt-grok-attack": {"models": [GPT_MODEL, GROK_MODEL], "mode": EvalMode.ATTACK, "alloy": True},
    "alloy-gemini-grok-attack": {"models": [GEMINI_MODEL, GROK_MODEL], "mode": EvalMode.ATTACK, "alloy": True},
}

# ── Phases (groups of configs) ──────────────────────────────────────────────

PHASES = {
    # The March pilot configurations; also used for AgentDojo and Data Poisoning
    "attack-trio": ["solo-gpt-attack", "solo-gemini-attack", "alloy-attack"],
    # The canonical pool's solos (history-stripped) and its three alloys
    "grok-expansion-full": [
        "solo-gpt-attack-nohistory",
        "solo-gemini-attack-nohistory",
        "solo-grok-attack-nohistory",
        "alloy-attack",
        "alloy-gpt-grok-attack",
        "alloy-gemini-grok-attack",
    ],
}


# ── Setting construction ────────────────────────────────────────────────────

def build_setting(name, side_tasks=None, filter_side_tasks=None):
    """Construct a ControlArena Setting by name.

    Args:
        name: Setting name — "bash", "bash_arena", "iac_fast", "agentdojo" or "data_poisoning".
        side_tasks: For bash_arena — list of side tasks to include (default: all 4).
        filter_side_tasks: For iac_fast — list of side tasks to filter attack samples to.
    """
    if name == "bash":
        from control_arena.settings.bash.bash_setting import BashSetting
        return BashSetting()

    elif name == "bash_arena":
        from control_arena.settings.bash_arena.bash_arena_setting import BashArenaSetting
        from control_arena.settings.bash_arena.task.dataset import DatasetSpec

        tasks = side_tasks if side_tasks else BASH_ARENA_SIDE_TASKS
        return BashArenaSetting(dataset_spec=DatasetSpec(
            side_tasks=tasks,
            task_id_whitelist=BASH_ARENA_TASK_IDS,
        ))

    elif name == "iac_fast":
        from control_arena.settings.iac_fast.setting import IACFastSetting

        # Setting base class has built-in filter_side_task_names support
        return IACFastSetting(filter_side_task_names=filter_side_tasks)

    elif name == "agentdojo":
        from control_arena.settings.agentdojo_setting.agentdojo_setting import AgentDojoSetting

        return AgentDojoSetting()

    elif name == "data_poisoning":
        from control_arena.settings.data_poisoning.data_poisoning_setting import DataPoisoningSetting

        return DataPoisoningSetting(use_subset_filter=True)

    else:
        raise ValueError(f"Unknown setting: {name}. Choose from: {', '.join(SETTING_DEFAULTS)}")


# ── Run functions ───────────────────────────────────────────────────────────

def run_solo(name, model, mode, setting, limit, max_steps, time_limit, max_tasks=None, generate_kwargs=None, epochs=1):
    """Run a solo (single-model) experiment."""
    print(f"\n{'=' * 60}")
    print(f"  {name} ({mode.value}) [limit={limit}]")
    if generate_kwargs:
        print(f"  generate_kwargs: {generate_kwargs}")
    print(f"{'=' * 60}")

    config_kwargs = dict(
        untrusted_model=model,
        trusted_model=model,
        limit=limit,
        max_steps=max_steps,
        log_dir=LOG_DIR,
        fail_on_error=False,
        time_limit=time_limit,
    )
    if max_tasks is not None:
        config_kwargs["max_tasks"] = max_tasks
    if epochs and epochs > 1:
        config_kwargs["epochs"] = epochs
    config = ControlEvalConfig(**config_kwargs)
    protocol = untrusted_policy_builder()

    task = get_control_task(
        setting=setting,
        micro_protocol=protocol,
        mode=mode,
        eval_config=config,
    )

    eval_kwargs = config.eval_args()
    if generate_kwargs:
        # Separate GenerateConfig params from eval-level params
        gen_config_keys = {"reasoning_effort", "reasoning_history"}
        gen_params = {k: v for k, v in generate_kwargs.items() if k in gen_config_keys}
        other_params = {k: v for k, v in generate_kwargs.items() if k not in gen_config_keys}
        # inspect_ai.eval() takes generation settings as keyword arguments and has no
        # `config` parameter: config=GenerateConfig(...) was silently dropped, so solo
        # runs launched after 2026-03-14 23:29 UTC ran without these flags.
        eval_kwargs.update(gen_params)
        eval_kwargs.update(other_params)
    results = eval(task, model=model, **eval_kwargs)
    print(f"  {name} done.")
    return results


def run_alloy(
    name,
    models,
    mode,
    setting,
    limit,
    max_steps,
    time_limit,
    max_tasks=None,
    generate_kwargs=None,
    strategy="round_robin",
    sequence=None,
    sample_id=None,
    epochs=1,
):
    """Run an alloy (multi-model) experiment."""
    print(f"\n{'=' * 60}")
    print(f"  {name} ({mode.value}) [limit={limit}]")
    if strategy != "round_robin":
        print(f"  strategy: {strategy}, sequence={sequence}")
    if generate_kwargs:
        print(f"  generate_kwargs: {generate_kwargs}")
    print(f"{'=' * 60}")

    config_kwargs = dict(
        untrusted_model=models[0],
        trusted_model=models[0],
        limit=limit,
        max_steps=max_steps,
        log_dir=LOG_DIR,
        fail_on_error=False,
        time_limit=time_limit,
    )
    if max_tasks is not None:
        config_kwargs["max_tasks"] = max_tasks
    if epochs and epochs > 1:
        config_kwargs["epochs"] = epochs
    config = ControlEvalConfig(**config_kwargs)

    # Pass generate_kwargs as GenerateConfig to the alloy protocol
    protocol_kwargs = {}
    if generate_kwargs:
        protocol_kwargs["config"] = GenerateConfig(**generate_kwargs)
    protocol = build_alloy_protocol(
        models=models,
        strategy=strategy,
        sequence=sequence,
        **protocol_kwargs,
    )

    task = get_control_task(
        setting=setting,
        micro_protocol=protocol,
        mode=mode,
        eval_config=config,
    )

    eval_kwargs = config.eval_args()
    if sample_id:
        eval_kwargs["sample_id"] = sample_id
        eval_kwargs.pop("limit", None)  # Can't specify both sample_id and limit
    results = eval(task, model=models[0], **eval_kwargs)
    print(f"  {name} done.")
    return results


def run_config(config_name, setting, limit, max_steps, time_limit, max_tasks=None, sample_id=None, epochs=1):
    """Run a named config against a setting."""
    cfg = CONFIGS[config_name]
    generate_kwargs = cfg.get("generate_kwargs")

    if cfg["alloy"]:
        models = cfg["models"]
        strategy = cfg.get("strategy", "round_robin")
        sequence = cfg.get("sequence")
        label = f"Alloy {'+'.join(m.split('/')[-1] for m in models)}"
        return run_alloy(
            label,
            models,
            cfg["mode"],
            setting,
            limit,
            max_steps,
            time_limit,
            max_tasks,
            generate_kwargs,
            strategy,
            sequence,
            sample_id=sample_id,
            epochs=epochs,
        )
    else:
        model = cfg["model"]
        label = f"Solo {model.split('/')[-1]}"
        return run_solo(label, model, cfg["mode"], setting, limit, max_steps, time_limit, max_tasks, generate_kwargs, epochs=epochs)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_limit(s):
    """Parse limit arg: '10' → 10, '10,20' → (10, 20)."""
    if "," in s:
        start, stop = s.split(",", 1)
        return (int(start), int(stop))
    return int(s)


def main():
    parser = argparse.ArgumentParser(
        description="Unified ControlArena experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Named configs: {', '.join(CONFIGS.keys())}
Phases: {', '.join(f'{k} ({", ".join(v)})' for k, v in PHASES.items())}

bash_arena side tasks: {', '.join(BASH_ARENA_SIDE_TASKS)}
iac_fast side tasks: {', '.join(IAC_FAST_SIDE_TASKS)}
        """,
    )

    parser.add_argument(
        "--setting", required=True, choices=list(SETTING_DEFAULTS.keys()),
        help="ControlArena setting to use",
    )

    # What to run
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", choices=list(CONFIGS.keys()),
        help="Named config to run",
    )
    group.add_argument(
        "--phase", choices=list(PHASES.keys()),
        help="Phase (group of configs) to run sequentially",
    )

    # Side task filtering
    parser.add_argument(
        "--side-tasks", nargs="+", metavar="TASK",
        help="bash_arena: which side tasks to include (default: all 4)",
    )
    parser.add_argument(
        "--filter-side-tasks", nargs="+", metavar="TASK",
        help="iac_fast: filter attack samples to these side tasks only",
    )

    # Eval parameters
    parser.add_argument("--limit", type=parse_limit, default=10, help="Sample limit: N or start,stop (default: 10)")
    parser.add_argument("--sample-id", type=int, nargs="+", metavar="ID",
        help="Run only these sample IDs (overrides --limit)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (repeats per sample, default: 1)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max agent steps per sample")
    parser.add_argument("--time-limit", type=int, default=None, help="Time limit per sample in seconds")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max parallel Docker tasks (default: setting-specific or 4)")

    args = parser.parse_args()

    # Validate side task args
    if args.side_tasks and args.setting != "bash_arena":
        parser.error("--side-tasks is only valid for --setting bash_arena")
    if args.filter_side_tasks and args.setting != "iac_fast":
        parser.error("--filter-side-tasks is only valid for --setting iac_fast")

    if args.side_tasks:
        for st in args.side_tasks:
            if st not in BASH_ARENA_SIDE_TASKS:
                parser.error(f"Unknown bash_arena side task: {st}. Choose from: {', '.join(BASH_ARENA_SIDE_TASKS)}")
    if args.filter_side_tasks:
        for st in args.filter_side_tasks:
            if st not in IAC_FAST_SIDE_TASKS:
                parser.error(f"Unknown iac_fast side task: {st}. Choose from: {', '.join(IAC_FAST_SIDE_TASKS)}")

    # Resolve defaults
    defaults = SETTING_DEFAULTS[args.setting]
    max_steps = args.max_steps or defaults["max_steps"]
    time_limit = args.time_limit or defaults["time_limit"]
    max_tasks = args.max_tasks or defaults.get("max_tasks")

    # Build setting
    setting = build_setting(
        args.setting,
        side_tasks=args.side_tasks,
        filter_side_tasks=args.filter_side_tasks,
    )

    # Determine configs to run
    configs_to_run = PHASES[args.phase] if args.phase else [args.config]

    # Print header
    side_info = ""
    if args.side_tasks:
        side_info = f", side_tasks={args.side_tasks}"
    elif args.filter_side_tasks:
        side_info = f", filter_side_tasks={args.filter_side_tasks}"

    print(f"ControlArena Unified Runner")
    print(f"{'=' * 60}")
    print(f"  Setting: {args.setting}{side_info}")
    print(f"  Configs: {', '.join(configs_to_run)}")
    max_tasks_info = f", Max tasks: {max_tasks}" if max_tasks else ""
    print(f"  Limit: {args.limit}, Max steps: {max_steps}, Time limit: {time_limit}s{max_tasks_info}")
    print(f"{'=' * 60}")

    # Run
    for config_name in configs_to_run:
        run_config(config_name, setting, args.limit, max_steps, time_limit, max_tasks,
                   sample_id=getattr(args, 'sample_id', None), epochs=getattr(args, 'epochs', 1))
        # Clean up stale Docker networks between configs to avoid pool exhaustion
        if args.setting in ("bash_arena", "iac_fast"):
            import subprocess
            subprocess.run(
                ["docker", "network", "prune", "--force"],
                capture_output=True,
            )

    print(f"\n{'=' * 60}")
    print(f"  All runs complete! Check {LOG_DIR} for results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
