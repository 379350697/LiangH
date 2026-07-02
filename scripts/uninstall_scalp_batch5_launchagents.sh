#!/bin/zsh
set -euo pipefail

uid="$(id -u)"
domain="gui/${uid}"
agent_dir="${HOME}/Library/LaunchAgents"

labels=(
  com.liangh.scalp.batch5.signal
  com.liangh.scalp.batch5.mm.btcusdt
  com.liangh.scalp.batch5.mm.ethusdt
  com.liangh.scalp.batch5.mm.dogeusdt
  com.liangh.scalp.batch5.mm.hypeusdt
  com.liangh.scalp.batch5.mm.xrpusdt
  com.liangh.scalp.batch5.mm.bnbusdt
)

for label in "${labels[@]}"; do
  launchctl bootout "${domain}/${label}" >/dev/null 2>&1 || true
  launchctl remove "${label}" >/dev/null 2>&1 || true
  rm -f "${agent_dir}/${label}.plist"
done
