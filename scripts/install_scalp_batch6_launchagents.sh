#!/bin/zsh
set -euo pipefail

repo="/Users/wl/projects/LiangH"
uid="$(id -u)"
domain="gui/${uid}"
agent_dir="${HOME}/Library/LaunchAgents"
log_dir="${repo}/output/fleet/scalp_suite_batch6_30bot/logs"

labels=(
  com.liangh.scalp.batch6.signal
  com.liangh.scalp.batch6.mm.btcusdt
  com.liangh.scalp.batch6.mm.ethusdt
  com.liangh.scalp.batch6.mm.dogeusdt
  com.liangh.scalp.batch6.mm.hypeusdt
  com.liangh.scalp.batch6.mm.xrpusdt
  com.liangh.scalp.batch6.mm.bnbusdt
)

mkdir -p "${agent_dir}" "${log_dir}"

for label in "${labels[@]}"; do
  launchctl bootout "${domain}/${label}" >/dev/null 2>&1 || true
  launchctl remove "${label}" >/dev/null 2>&1 || true
done

for label in "${labels[@]}"; do
  plist="${label}.plist"
  source_plist="${repo}/configs/launchagents/scalp_batch6/${plist}"
  target_plist="${agent_dir}/${plist}"
  cp "${source_plist}" "${target_plist}"
  plutil -lint "${target_plist}" >/dev/null
  launchctl bootstrap "${domain}" "${target_plist}"
done

for label in "${labels[@]}"; do
  launchctl print "${domain}/${label}" | awk '/state =|pid =|runs =|last exit code/ {print "'"${label}"': " $0}'
done
