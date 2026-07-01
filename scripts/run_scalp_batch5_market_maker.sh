#!/bin/zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: run_scalp_batch5_market_maker.sh <config>" >&2
  exit 2
fi

cd /Users/wl/projects/LiangH
mkdir -p output/fleet/scalp_suite_batch5_30bot/logs
export PYTHONUNBUFFERED=1

config="$1"
name="$(basename "$config" .json)"
duration="${SCALP_BATCH5_MM_DURATION_SECONDS:-86400}"

exec .venv/bin/python -m liangh_trader.market_maker.cli \
  --config "$config" \
  --duration-seconds "$duration" \
  >> "output/fleet/scalp_suite_batch5_30bot/logs/${name}.log" 2>&1
