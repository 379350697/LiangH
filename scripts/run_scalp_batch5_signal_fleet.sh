#!/bin/zsh
set -euo pipefail

cd /Users/wl/projects/LiangH
mkdir -p output/fleet/scalp_suite_batch5_30bot/logs
export PYTHONUNBUFFERED=1

exec .venv/bin/python -m langlang_trader.fleet_cli \
  --config configs/fleet/scalp_suite_batch5_24bot_paper.json \
  --loop \
  --interval-seconds 60 \
  >> output/fleet/scalp_suite_batch5_30bot/logs/signal_fleet.log 2>&1
