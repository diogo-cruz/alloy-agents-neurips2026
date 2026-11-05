#!/usr/bin/env bash
# E2 stage 1 (docs/precommitments/E2_PRECOMMIT.md): each candidate solo on Bash task
# indices 100-119, one epoch, through the alloy policy code path (run_audit_bash.py --mode
# solo). Two concurrent waves keep the number of Docker networks well under the ~29 limit.
set -u
cd "$(dirname "$0")/.."
OUT=logs/camera_ready/e2_screen
mkdir -p "$OUT"
usage() {
  set -a; . ./.env; set +a
  curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/key |
    python3 -c "import json,sys; print('%.3f' % json.load(sys.stdin)['data']['usage'])"
}
echo "screen start $(date -u +%FT%TZ) commit $(git rev-parse --short HEAD) openrouter_usage=$(usage)" >> "$OUT/launch.txt"
run() {  # $1 = slug without the openrouter/ prefix
  local dir="$OUT/${1//\//__}"
  uv run --no-sync python scripts/run_audit_bash.py --mode solo --models "openrouter/$1" \
    --limit 100,120 --max-samples 5 --log-dir "$dir" > "$dir.log" 2>&1
  echo "$1 exit=$? $(date -u +%FT%TZ)" >> "$OUT/launch.txt"
}
run openai/gpt-6-luna & run anthropic/claude-haiku-5.5 & run google/gemini-3.5-flash-lite & run qwen/qwen3.8-flash &
wait
run deepseek/deepseek-v4.1-flash & run mistralai/mistral-small-2603 & run z-ai/glm-5.3-flash &
wait
echo "screen end $(date -u +%FT%TZ) openrouter_usage=$(usage)" >> "$OUT/launch.txt"
