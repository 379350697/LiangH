#!/bin/zsh
set -euo pipefail

uid="$(id -u)"
domain="gui/${uid}"
agent_dir="${HOME}/Library/LaunchAgents"

labels=(
  com.liangh.scalp.batch7.signal
  com.liangh.scalp.batch7.mm.btcusdt
  com.liangh.scalp.batch7.mm.ethusdt
  com.liangh.scalp.batch7.mm.dogeusdt
  com.liangh.scalp.batch7.mm.hypeusdt
  com.liangh.scalp.batch7.mm.xrpusdt
  com.liangh.scalp.batch7.mm.bnbusdt
)

for label in "${labels[@]}"; do
  launchctl bootout "${domain}/${label}" >/dev/null 2>&1 || true
  launchctl remove "${label}" >/dev/null 2>&1 || true
  if [[ "${REMOVE_SCALP_BATCH7_LAUNCHAGENTS:-0}" == "1" ]]; then
    rm -f "${agent_dir}/${label}.plist"
  fi
done
