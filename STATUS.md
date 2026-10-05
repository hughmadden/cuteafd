# WP-4 status

Branch: work/mm-spark-encoder; base origin/work/p0 163c3a36.

Plan:
1. Add pure encoder placement policy and G9 matrix, charge before KV pool sizing, hash/report placement.
2. Add bounded TCP remote client/server, identity handshake, replica dispatch and health; integrate expertd modes.
3. Run workspace/script gates and matching-SM121 remote byte gate; measure transport and interference under allocated locks.

Coordination: WP-5 owns local adapter and MiMo serving; WP-4 owns remote adapter and placement. EncoderClient remains unchanged. V4.1 serving remains untouched.

Green host step: pure placement/G9 policy, tower+scratch charged before KV sizing, checkpoint/placement hash, bounded TCP replica channel with identity readiness and idle heartbeat, encoder-only and same-process expertd admission. Review fixes: absolute handshake/frame deadline; explicit unmet placement fails closed; explicit KV pool preserved; replicas require --layout inventory.

Gates: full cargo workspace exit 0; full scripts 993 passed, 2 skipped, 193 subtests passed (77.11 s; no failing ids). Post-review loader 245 passed/12 ignored (12.12 s), daemon 416 passed/123 ignored (0.75 s), daemon integrations green; replica CLI regression passes. Loopback byte identity/cancel/drain and trickle deadline pass. Actual SM121 byte gate, transfer fraction and concurrent/idle/off C1 gates pending. No placement-default promotion.

Build: matching dev image pair si-2c76d55 rebuilt/distributed to ostrich,dodo,emu,kiwi, exit 0. Standard WIP build next. Shared release config currently rejects MiMo SPARK_COUNT=2 unless EXPERT_FORMAT=exl3 (V4.1-specific validator); use 4-Spark build config and direct family min launcher, do not relax shared validator here.

Integration still pending: launcher selected encoder flags/hash/replicas/readiness; WP-5 consumes adapter and owns API health integration. Idle-Spark policy unit tested; real idle-host inventory remains pending. Hardware: no reservation queued or held.
