#!/bin/zsh
set -euo pipefail

repo="/Users/wl/projects/LiangH"
uid="$(id -u)"
domain="gui/${uid}"
agent_dir="${HOME}/Library/LaunchAgents"
log_dir="${repo}/output/fleet/scalp_suite_batch7_24bot/logs"

labels=(
  com.liangh.scalp.batch7.signal
  com.liangh.scalp.batch7.mm.btcusdt
  com.liangh.scalp.batch7.mm.ethusdt
  com.liangh.scalp.batch7.mm.dogeusdt
  com.liangh.scalp.batch7.mm.hypeusdt
  com.liangh.scalp.batch7.mm.xrpusdt
  com.liangh.scalp.batch7.mm.bnbusdt
)

mkdir -p "${agent_dir}" "${log_dir}"

for label in "${labels[@]}"; do
  launchctl bootout "${domain}/${label}" >/dev/null 2>&1 || true
  launchctl remove "${label}" >/dev/null 2>&1 || true
done

for label in "${labels[@]}"; do
  plist="${label}.plist"
  source_plist="${repo}/configs/launchagents/scalp_batch7/${plist}"
  target_plist="${agent_dir}/${plist}"
  cp "${source_plist}" "${target_plist}"
  plutil -lint "${target_plist}" >/dev/null
  launchctl bootstrap "${domain}" "${target_plist}"
done

for label in "${labels[@]}"; do
  launchctl print "${domain}/${label}" | awk '/state =|pid =|runs =|last exit code/ {print "'"${label}"': " $0}'
done
