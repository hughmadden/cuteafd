#!/usr/bin/env bash
# Read-only receipts; never reset counters. Redirect stdout before and after a run.
set -euo pipefail
printf 'UTC %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
timeout 30s ssh -n -o BatchMode=yes -o ConnectTimeout=10 agent@switch.200gb \
    '/interface ethernet switch qos export verbose; /interface ethernet switch qos port print stats; /interface ethernet print stats; /interface ethernet switch qos priority-flow-control print detail'
for host in raptor ostrich dodo emu kiwi rhea moa; do
    printf '\nHOST %s UTC %s\n' "$host" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    timeout 30s ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host" 'set -eu
        rdma link show
        for dev in /sys/class/infiniband/*; do
            for port in "$dev"/ports/*; do
                printf "RDMA %s\n" "$port"
                cat "$port/state" "$port/rate"
                for counter in "$port"/hw_counters/* "$port"/counters/*; do
                    test -f "$counter" || continue
                    printf "%s " "$counter"; cat "$counter"
                done
            done
        done
        for nic in /sys/class/net/*; do
            name=${nic##*/}
            case "$name" in enp1s0np0|enp1s0f0np0|enP2p1s0f0np0)
                ethtool -S "$name"
                if command -v mlnx_qos >/dev/null; then mlnx_qos -i "$name"; fi
            esac
        done'
done
