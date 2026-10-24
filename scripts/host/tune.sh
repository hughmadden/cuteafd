#!/usr/bin/env bash
# Host policy only: never alter NIC/fabric, GPU, or switch settings.
set -euo pipefail
shopt -s nullglob

COORDINATOR_IDLE_LATENCY_US=50
SPARK_IDLE_LATENCY_US=200
SYSFS=${TUNE_SYSFS_ROOT:-/sys}
SYSFS=${SYSFS%/}
# A fixture root must never accidentally read the host's /proc or run host probes.
PROCFS=/proc
LIVE=1
if [[ $SYSFS != /sys ]]; then
    PROCFS=$SYSFS/proc
    LIVE=0
fi
ROLE=
ACTION=report
DRY_RUN=0
ACTION_SET=0

usage() {
    echo 'Usage: tune.sh [report|apply|persist] [--role coordinator|spark] [--dry-run]'
    echo 'TUNE_SYSFS_ROOT substitutes /sys; fixtures put proc files under ROOT/proc.'
}

while (($#)); do
    case $1 in
        report|apply|persist)
            if ((ACTION_SET)); then usage >&2; exit 2; fi
            ACTION=$1; ACTION_SET=1; shift ;;
        --role)
            if (($# < 2)); then usage >&2; exit 2; fi
            ROLE=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ -z $ROLE ]]; then
    case $(uname -m) in
        x86_64) ROLE=coordinator ;;
        aarch64|arm64) ROLE=spark ;;
        *) echo 'Unknown architecture; specify --role coordinator|spark.' >&2; exit 2 ;;
    esac
fi
case $ROLE in
    coordinator) IDLE_LATENCY_US=$COORDINATOR_IDLE_LATENCY_US ;;
    spark) IDLE_LATENCY_US=$SPARK_IDLE_LATENCY_US ;;
    *) usage >&2; exit 2 ;;
esac
if ((DRY_RUN)) && [[ $ACTION != apply ]]; then usage >&2; exit 2; fi

read_value() {
    local value
    if [[ -r $1 ]] && value=$(cat -- "$1" 2>/dev/null); then
        printf '%s' "$value"
    else
        printf 'unknown'
    fi
}

row() {
    local label=$1 current=$2 recommended=$3 marker=unknown
    if [[ $current != unknown && $recommended != unknown && $recommended != 'site policy (read-only)' ]]; then
        marker=differs
        [[ $current != "$recommended" ]] || marker=pass
    fi
    # Keep each observation on one line, including multiline command results.
    current=${current//$'\n'/; }
    label=${label#devices/system/}
    label=${label#cpu/}
    label=${label/\/cpuidle\//\/}
    printf '[%s] %s: %s | target=%s\n' "$marker" "$label" "$current" "$recommended"
}

write_value() {
    local file=$1 expected=$2 current
    current=$(read_value "$file")
    [[ $current != "$expected" ]] || return 0
    if ((DRY_RUN)); then
        printf 'DRY-RUN write %s <- %s (was %s)\n' "$file" "$expected" "$current"
    else
        printf 'write %s <- %s (was %s)\n' "$file" "$expected" "$current"
        printf '%s\n' "$expected" > "$file"
    fi
}

setting() {
    local file=$1 expected=$2
    if [[ $ACTION == apply ]]; then
        write_value "$file" "$expected"
    else
        row "${file#"$SYSFS"/}" "$(read_value "$file")" "$expected"
    fi
}

probe() {
    local label=$1 output
    shift
    if ((LIVE)) && command -v "$1" >/dev/null 2>&1 &&
        output=$(timeout 10s "$@" 2>&1); then
        row "$label" "$output" 'site policy (read-only)'
    else
        row "$label" unknown 'site policy (read-only)'
    fi
}

if [[ $ACTION == persist ]]; then
    # Stable install location; do not embed a transient checkout or fixture root.
    cat <<EOF
# Install tune.sh as /usr/local/sbin/cuteafd-host-tune (root-owned, mode 0755).
# Save this unit as /etc/systemd/system/cuteafd-host-tune.service, then run:
# systemctl daemon-reload && systemctl enable --now cuteafd-host-tune.service
[Unit]
Description=cuteafd host tuning ($ROLE)
After=systemd-udev-settle.service ondemand.service
Wants=systemd-udev-settle.service
ConditionPathIsExecutable=/usr/local/sbin/cuteafd-host-tune

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/cuteafd-host-tune apply --role $ROLE
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    exit 0
fi
# Also query the effective UID: an exported EUID can shadow bash's value.
if [[ $ACTION == apply ]] && ((!DRY_RUN)) && { ((EUID != 0)) || [[ $(id -u) != 0 ]]; }; then
    echo 'apply requires root (EUID 0); a human must run it. Use --dry-run to preview.' >&2
    exit 1
fi
printf 'role=%s idle_disable_latency_us>=%s mode=%s\n' "$ROLE" "$IDLE_LATENCY_US" "$ACTION"

policies=("$SYSFS"/devices/system/cpu/cpufreq/policy*)
if ((${#policies[@]} == 0)); then
    policies=("$SYSFS"/devices/system/cpu/cpu[0-9]*/cpufreq)
fi
if [[ $ACTION == report ]] && ((${#policies[@]} == 0)); then
    row 'CPU governor' unknown performance
    row 'CPU energy_performance_preference' unknown "$(
        if [[ $ROLE == coordinator ]]; then echo performance; else echo 'site policy (read-only)'; fi)"
fi
for policy in "${policies[@]}"; do
    setting "$policy/scaling_governor" performance
    epp=$policy/energy_performance_preference
    if [[ $ROLE == coordinator ]]; then
        if [[ -e $epp || $ACTION == report ]]; then setting "$epp" performance; fi
    elif [[ $ACTION == report ]]; then
        row "${epp#"$SYSFS"/}" "$(read_value "$epp")" 'site policy (read-only)'
    fi
done

states=("$SYSFS"/devices/system/cpu/cpu[0-9]*/cpuidle/state[0-9]*)
if [[ $ACTION == report ]] && ((${#states[@]} == 0)); then
    row cpuidle unknown "disable states with latency >= $IDLE_LATENCY_US us"
fi
for state in "${states[@]}"; do
    latency=$(read_value "$state/latency")
    target=unknown
    if [[ $latency =~ ^[0-9]+$ ]]; then
        target=0
        if ((10#$latency >= IDLE_LATENCY_US)); then target=1; fi
    fi
    if [[ $ACTION == apply ]]; then
        # Never re-enable a shallow state that an administrator disabled.
        if [[ $target == 1 ]]; then write_value "$state/disable" 1; fi
    else
        row "${state#"$SYSFS"/} name=$(read_value "$state/name") latency_us=$latency usage=$(read_value "$state/usage") disable" \
            "$(read_value "$state/disable")" "$target"
    fi
done

compaction=$PROCFS/sys/vm/compaction_proactiveness
if [[ $ROLE == spark ]]; then
    setting "$compaction" 0
elif [[ $ACTION == report ]]; then
    row vm.compaction_proactiveness "$(read_value "$compaction")" 'site policy (read-only)'
fi
setting "$PROCFS/sys/vm/swappiness" 10
[[ $ACTION == report ]] || exit 0
row vm.zone_reclaim_mode "$(read_value "$PROCFS/sys/vm/zone_reclaim_mode")" 0
for name in enabled defrag; do
    modes=$(read_value "$SYSFS/kernel/mm/transparent_hugepage/$name")
    active=unknown
    if [[ $modes =~ \[([^]]+)\] ]]; then active=${BASH_REMATCH[1]}; fi
    row "THP $name (modes: $modes)" "$active" madvise
done
if ((LIVE)); then
    row 'memlock (report process, KiB)' "$(ulimit -l)" unlimited
    irqbalance=unknown
    if command -v systemctl >/dev/null 2>&1; then
        irqbalance=$(timeout 10s systemctl is-active irqbalance.service 2>/dev/null) || true
        [[ -n $irqbalance ]] || irqbalance=unknown
    fi
    row irqbalance "$irqbalance" inactive
else
    row memlock unknown unlimited
    row irqbalance unknown inactive
fi

if [[ $ROLE == spark ]]; then
    topology=unknown
    if ((LIVE)); then
        topology=$(LC_ALL=C timeout 10s lscpu -p=CPU,MAXMHZ 2>/dev/null) || topology=unknown
    else
        topology=$(read_value "$SYSFS/lscpu.csv")
    fi
    declare -A cpu_mhz=()
    max_mhz=0
    min_mhz=0
    while IFS=, read -r cpu mhz; do
        if [[ $cpu =~ ^[0-9]+$ && $mhz =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            mhz=${mhz%%.*}
            cpu_mhz[$cpu]=$mhz
            if ((mhz > max_mhz)); then max_mhz=$mhz; fi
            if ((min_mhz == 0 || mhz < min_mhz)); then min_mhz=$mhz; fi
        fi
    done <<< "$topology"
    pcores=()
    ecores=()
    if ((max_mhz > min_mhz && min_mhz > 0)); then
        while read -r cpu; do
            if ((${cpu_mhz[$cpu]} == max_mhz)); then pcores+=("$cpu"); else ecores+=("$cpu"); fi
        done < <(printf '%s\n' "${!cpu_mhz[@]}" | sort -n)
        row 'GB10 cores (derived from MAXMHZ)' "P=${pcores[*]} (${max_mhz}MHz) E=${ecores[*]} (${min_mhz}MHz)" 'site policy (read-only)'
    else
        row 'GB10 P/E cores (MAXMHZ unavailable or uniform)' unknown 'site policy (read-only)'
    fi
    # Report placement only; never change an expert worker's affinity.
    found=0
    if ((LIVE)); then
        for cmdline in "$PROCFS"/[0-9]*/cmdline; do
            command_line=$(tr '\0' ' ' < "$cmdline" 2>/dev/null) || continue
            [[ $command_line =~ (^|/|[[:space:]])cuteafd[[:space:]]+expertd(-native)?([[:space:]]|$) ]] || continue
            pid=${cmdline%/cmdline}; pid=${pid##*/}
            found=1
            probe "expertd pid=$pid threads tid/psr/comm" ps -L -p "$pid" -o tid,psr,comm
            hot_tid=unknown; hot_cpu=unknown; hot_pct=-1
            while read -r tid cpu pct; do
                [[ $tid =~ ^[0-9]+$ && $cpu =~ ^[0-9]+$ && $pct =~ ^[0-9]+([.][0-9]+)?$ ]] || continue
                # ps %CPU is lifetime CPU, not an instantaneous utilization sample.
                pct=${pct/./}
                if ((10#$pct > hot_pct)); then hot_pct=$((10#$pct)); hot_tid=$tid; hot_cpu=$cpu; fi
                probe "expertd pid=$pid tid=$tid affinity" taskset -p "$tid"
            done < <(LC_ALL=C ps -L -p "$pid" -o tid=,psr=,pcpu= 2>/dev/null || true)
            placement=unknown
            if [[ $hot_cpu != unknown && ${cpu_mhz[$hot_cpu]:-0} != 0 ]] && ((max_mhz > min_mhz)); then
                placement='P core'
                if ((${cpu_mhz[$hot_cpu]} < max_mhz)); then placement='E core'; fi
            fi
            row "expertd pid=$pid hottest thread (lifetime %CPU) tid=$hot_tid last_cpu=$hot_cpu" "$placement" 'P core'
            probe "expertd pid=$pid limits (check worker memlock too)" cat "$PROCFS/$pid/limits"
        done
    fi
    if ((!found)); then row 'expertd threads' 'not running (or not visible)' 'site policy (read-only)'; fi
fi
probe 'rdma link' rdma link show
ports=("$SYSFS"/class/infiniband/*/ports/*)
if ((${#ports[@]} == 0)); then row 'RoCE port rate' unknown 'site policy (read-only)'; fi
for port in "${ports[@]}"; do
    row "${port#"$SYSFS"/}/rate" "$(read_value "$port/rate")" 'site policy (read-only)'
done

# ndevs is port-specific; device/net also works when GID attributes are hidden.
declare -A netdevs=()
for entry in "$SYSFS"/class/infiniband/*/ports/*/gid_attrs/ndevs/*; do
    netdev=$(read_value "$entry")
    if [[ $netdev != unknown && -n $netdev && $netdev != */* ]]; then netdevs[$netdev]=1; fi
done
for entry in "$SYSFS"/class/infiniband/*/device/net/*; do netdevs[${entry##*/}]=1; done
if ((${#netdevs[@]} == 0)); then row 'RoCE NIC diagnostics' unknown 'site policy (read-only)'; fi
for netdev in "${!netdevs[@]}"; do
    probe "$netdev coalescing" ethtool -c "$netdev"
    probe "$netdev PFC" dcb pfc show dev "$netdev"
    probe "$netdev DCB ETS" dcb ets show dev "$netdev"
    probe "$netdev DCB application priorities" dcb app show dev "$netdev"
    probe "$netdev DCBX" dcb dcbx show dev "$netdev"
    irqs=("$SYSFS/class/net/$netdev/device/msi_irqs/"*)
    if ((${#irqs[@]} == 0)); then row "$netdev IRQ affinity" unknown 'site policy (read-only)'; fi
    # Group equal affinities without hiding any IRQ IDs.
    declare -A affinity_groups=()
    for irq in "${irqs[@]}"; do
        irq=${irq##*/}
        affinity=$(read_value "$PROCFS/irq/$irq/smp_affinity_list")
        affinity_groups[$affinity]+=" $irq"
    done
    for affinity in "${!affinity_groups[@]}"; do
        row "$netdev IRQs${affinity_groups[$affinity]} affinity_list" "$affinity" 'site policy (read-only)'
    done
    unset affinity_groups
done
probe 'GPU performance / clocks-event reasons' nvidia-smi -q -d PERFORMANCE
