#!/usr/bin/env bash
# E2 stage 2 (docs/precommitments/E2_PRECOMMIT.md): the pair selected by
# scripts/analyze_e2_screen.py (reports/e2_screen.json) on Bash task indices 0-99, one
# epoch: alloy A+B (A first), A solo, B solo, concurrently, one policy code path and route.
set -u
cd "$(dirname "$0")/.."
OUT=logs/camera_ready/e2_main
E2_START_USAGE=13.533   # OpenRouter key usage when the screen started (screen launch.txt)
usage() {
  set -a; . ./.env; set +a
  curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/key |
    python3 -c "import json,sys; print('%.3f' % json.load(sys.stdin)['data']['usage'])"
}
read -r A B < <(python3 -c "import json; s=json.load(open('reports/e2_screen.json'))['selected']; print(s['A'] or '', s['B'] or '')")
if [ -z "${A:-}" ] || [ -z "${B:-}" ]; then echo "no pair selected: E2 stops"; exit 1; fi
NOW=$(usage)
SPENT=$(python3 -c "print(round($NOW - $E2_START_USAGE, 3))")
# Budget rule: stop if stage-1 spend > \$8 or projected total > \$20 (stage 2 has ~2.1x the
# screen's samples: 300 vs 140).
if python3 -c "import sys; s=$SPENT; sys.exit(0 if (s > 8 or s * (1 + 300 / 140) > 20) else 1)"; then
  echo "budget rule stops E2: screen spend \$$SPENT"; exit 1
fi
mkdir -p "$OUT"
echo "main start $(date -u +%FT%TZ) commit $(git rev-parse --short HEAD) A=$A B=$B openrouter_usage=$NOW screen_spend=$SPENT" >> "$OUT/launch.txt"
arm() {  # $1 = arm name, rest = runner arguments
  local name=$1; shift
  uv run --no-sync python scripts/run_audit_bash.py "$@" --limit 100 --max-samples 8 \
    --log-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "$name exit=$? $(date -u +%FT%TZ)" >> "$OUT/launch.txt"
}
arm alloy --mode standard --models "openrouter/$A,openrouter/$B" &
arm solo_a --mode solo --models "openrouter/$A" &
arm solo_b --mode solo --models "openrouter/$B" &
wait
echo "main end $(date -u +%FT%TZ) openrouter_usage=$(usage)" >> "$OUT/launch.txt"
