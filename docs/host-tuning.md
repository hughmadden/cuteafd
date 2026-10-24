# Host tuning

One inventory for host policy: `scripts/host/tune.sh`. Run outside containers.
Auto-detection maps x86-64 to coordinator and ARM64 to Spark; other sites
should pass `--role coordinator|spark` explicitly. ARM64 is not proof of GB10.

## Applied policy

| Setting | Role | Recommended value | Reason |
| --- | --- | --- | --- |
| CPU governor | Both | `performance` | Avoid frequency ramp-up on short host phases. |
| `vm.swappiness` | Both | `10` | Recommended: reduce swap preference while retaining reclaim flexibility (was 60 on both hosts). |
| Energy-performance preference (EPP) | Coordinator, where exposed | `performance` | Prefer responsiveness over energy saving. |
| cpuidle `disable` | Coordinator | `1` when exit `latency >= 50` microseconds | Avoid deep-idle wake latency between waves; retain shallow states. |
| cpuidle `disable` | Spark | `1` when exit `latency >= 200` microseconds | Avoid deep GB10 wake latency; retain shallow states. |
| `vm.compaction_proactiveness` | Spark | `0` | TJ's 2026-10-23 policy: stop proactive background compaction competing with serving on unified memory. This does not disable allocation-time compaction. |

The cutoffs are variables at the top of the script. Selection uses each
state's measured sysfs `latency`, never its name. On raptor this excludes
C2 (100 microseconds), retaining POLL/C1. On GB10 it excludes LPI-2
(231 microseconds) and LPI-3 (433 microseconds), retaining LPI-0/LPI-1.
The script only disables deep states; it never re-enables a shallow state
an operator deliberately disabled. Governor and EPP apply per cpufreq policy.

**Measured evidence:** [#717](https://github.com/tpurtell/cuteafd/issues/717)
profiles identify per-wave host gaps: V4.1 had 11.8 seconds host-late in a
25.2-second window, with 9.1 seconds in gaps at least 200 microseconds.
Other families show route/exchange waits. These establish latency-sensitive
host work, not that idle exit caused every gap.

**Recommended, not measured speedups:** the idle cutoffs are TJ's
2026-10-25 decision, approved without an A/B. They trade idle power for
wake responsiveness. Governor/EPP performance and Spark compaction zero
were already cluster policy; the script enforces them. No throughput gain
is claimed for this host-tuning change.

## Read-only inventory

- CPU: governor/EPP and every CPU's idle-state name, exit latency, disable
  bit and usage counter. `[pass]` matches a recommendation, `[differs]`
  does not, `[unknown]` means unavailable or no universal recommendation.
- THP `enabled` and `defrag`: recommend `madvise` (already set on our
  hosts); report modes and compare the bracketed active mode. No writes.
- `vm.zone_reclaim_mode=0`, unlimited memlock and inactive irqbalance:
  recommended and already set on our hosts; report only. Memlock is the
  report process's limit; a running expert worker's limits are also shown.
  Service/container limits can differ from an interactive shell.
- RoCE netdev IRQ CPU lists and `ethtool -c` coalescing: report only.
  Affinities are grouped by identical CPU list, retaining every IRQ ID.
  Choose IRQ/NUMA placement and coalescing against the site's topology and
  traffic; this script does not set them.
- `rdma link` and each RDMA port's sysfs `rate`: negotiated rates, not
  assumed hardware maxima. Check the actual rails before interpreting runs.
- PFC and DCB ETS/application/DCBX state via `dcb`: report only. Missing
  tools, unsupported drivers or insufficient privileges yield unknown.
  No generic PFC/DCB/ECN recommendation; coordinate with the fabric owner.
- `nvidia-smi -q -d PERFORMANCE`: GPU clocks-event reasons, read-only.
  GPU power is gpu-powerd's job on our cluster. Never change GPU limits or
  clocks, switch settings, NIC firmware, DCB, ECN or PFC with this script.

## GB10 core placement

GB10 has 10 Cortex-X925 performance cores (3.9 GHz) and 10 Cortex-A725
efficiency cores (about 2.8 GHz; ostrich reports 2808 MHz). Report derives the P/E sets from `lscpu`
`MAXMHZ`, never fixed CPU ranges. Missing or uniform frequencies yield
unknown. Recommend pinning the Spark expert worker to the reported P
cores; do not copy another machine's CPU IDs. The
[NVIDIA discussion](https://forums.developer.nvidia.com/t/fine-tune-kernel-level/372645)
is guidance, not a cuteafd performance measurement.

For a visible `cuteafd expertd`, report lists thread IDs, last CPUs and
`taskset -p` masks. It labels the highest lifetime-%CPU thread's placement
P/E/unknown. This is a snapshot heuristic, not proof of which thread is
currently hot or where it spends all its time. Pinning is a recommendation
only; this PR does not change launcher affinity.

## Considered and not applied

- `isolcpus`/`nohz_full`/`rcu_nocbs`/`processor.max_cstate=1`: boot parameters
  need a reboot; use sysfs for idle policy and affinity for core placement.
- TCP buffers/BBR/`somaxconn`: expert traffic uses RDMA verbs, not TCP.
- `panic_on_oom=1`: turns OOM into a reboot on shared hosts.
- `overcommit_memory=1`, NVMe queue tuning: low value; leave site defaults.
- NCCL: not used by design; serving uses our verbs/P2P collectives.

## GB10 memory

CUDA allocations reclaim page cache. Size from `/proc/meminfo`'s
`MemAvailable`, not `MemFree` or `nvidia-smi`'s unavailable GB10 memory
reading. Never drop caches. Keep headroom and bound simultaneous CUDA
context creation: [#730](https://github.com/tpurtell/cuteafd/issues/730)
records context-creation OOMs despite high MemAvailable; reclaimable bytes
are not a guarantee that parallel contexts can allocate immediately.

## Run and persist

```bash
scripts/host/tune.sh report --role coordinator
scripts/host/tune.sh apply --role coordinator --dry-run
# Human operator only, after reviewing the proposed writes:
sudo scripts/host/tune.sh apply --role coordinator
scripts/host/tune.sh persist --role coordinator
```

Use `--role spark` on GB10. Default action is `report`; no root needed.
`apply` refuses non-root callers; `--dry-run` is an unprivileged preview
that prints only proposed writes and changes nothing. Repeated apply
skips values already correct. Writes fail visibly rather than claiming
success when the kernel rejects them.

`persist` only prints a systemd unit and installation instructions. A human
installs the reviewed script at `/usr/local/sbin/cuteafd-host-tune`, owned by
root and mode 0755, and the unit at
`/etc/systemd/system/cuteafd-host-tune.service`, then enables it. Idle-state
`disable` resets on reboot. Coordinate with any other governor/tuning
service; the unit runs after `ondemand.service` where present. CPU hotplug
or later policy changes may require reapplying; this is a boot oneshot,
not a watchdog. No service or host setting is installed by `persist`.

Tests use `TUNE_SYSFS_ROOT=PATH`: sysfs files live under PATH, proc files
under PATH/proc, and external hardware probes are suppressed. Do not set
this fixture override for real host tuning.
