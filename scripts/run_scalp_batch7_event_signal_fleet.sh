#!/bin/zsh
set -euo pipefail

cd /Users/wl/projects/LiangH
mkdir -p output/fleet/scalp_suite_batch7_24bot/logs
export PYTHONUNBUFFERED=1

duration="${SCALP_BATCH7_SIGNAL_DURATION_SECONDS:-86400}"

exec .venv/bin/python -m langlang_trader.hft_scalping \
  --config configs/fleet/scalp_suite_batch7_18bot_event_signal_paper.json \
  --duration-seconds "$duration" \
  >> output/fleet/scalp_suite_batch7_24bot/logs/event_signal_fleet.log 2>&1
