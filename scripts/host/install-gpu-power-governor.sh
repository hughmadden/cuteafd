#!/usr/bin/env bash
# Print (default) the exact root commands that install the GPU thermal power governor on
# this host; nothing here runs as root. The operator reviews and runs them, e.g. one per
# `agent-sudo -n --agent-context "<why>" <command>` so each is visible to the approver.
# Only for hosts whose GPUs overheat (raptor's two 600 W RTX PRO 6000); never the Sparks.
#
#   scripts/host/install-gpu-power-governor.sh            # print the commands
#   scripts/host/install-gpu-power-governor.sh --check    # read-only check as this user
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host="$(hostname -s)"
config_src="${GOVERNOR_CONFIG:-$here/gpu-power-governor.example.json}"

python3 - "$config_src" "$host" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
if sys.argv[2] not in config.get("hosts", []):
    sys.exit(f"{sys.argv[2]} is not in {sys.argv[1]} hosts {config.get('hosts')}: this host does not get the governor")
EOF

if [[ "${1:-}" == --check ]]; then
  exec python3 "$here/gpu-power-governor" --config "$config_src" --check
fi

cat <<EOF
# 1. Read-only check (no root): readings, board power range and config.
python3 $here/gpu-power-governor --config $config_src --check
# 2. Install the program, config (default ceiling 325 W: no change until raised) and unit.
install -o root -g root -m 0755 $here/gpu-power-governor /usr/local/sbin/gpu-power-governor
install -o root -g root -m 0644 $config_src /etc/gpu-power-governor.json
install -o root -g root -m 0644 $here/gpu-power-governor.service /etc/systemd/system/gpu-power-governor.service
systemctl daemon-reload
# 3. Start now and at boot (after nvidia-power-limit.service, so the governor's limit wins).
systemctl enable --now gpu-power-governor.service
# 4. Verify: active, both limits at the ceiling, state file written.
systemctl is-active gpu-power-governor.service
nvidia-smi --query-gpu=index,power.limit,temperature.gpu --format=csv
cat /run/gpu-power-governor/state.json
# Raising the ceiling later (TJ's decision): edit ceiling_w in /etc/gpu-power-governor.json, then
#   systemctl restart gpu-power-governor.service
# Uninstall: systemctl disable --now gpu-power-governor.service (ExecStopPost restores safe_w), then
#   rm /usr/local/sbin/gpu-power-governor /etc/gpu-power-governor.json /etc/systemd/system/gpu-power-governor.service
#   systemctl daemon-reload
EOF
