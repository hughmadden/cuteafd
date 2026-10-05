# WP-4 status

Branch: work/mm-spark-encoder; base origin/work/p0 163c3a36.

Plan:
1. Add pure encoder placement policy and G9 matrix, charge before KV pool sizing, hash/report placement.
2. Add bounded TCP remote client/server, identity handshake, replica dispatch and health; integrate expertd modes.
3. Run workspace/script gates and matching-SM121 remote byte gate; measure transport and interference under allocated locks.

Coordination: WP-5 owns local adapter and MiMo serving; WP-4 owns remote adapter and placement. EncoderClient remains unchanged. V4.1 serving remains untouched.

Green host step: pure placement/G9 policy, tower+scratch charged before KV sizing, checkpoint/placement hash, bounded TCP replica channel with identity readiness and idle heartbeat, encoder-only and same-process expertd admission. Review fixes: absolute handshake/frame deadline; explicit unmet placement fails closed; explicit KV pool preserved; replicas require --layout inventory.

Gates: full cargo workspace exit 0; full scripts 993 passed, 2 skipped, 193 subtests passed (77.11 s; no failing ids). Post-review loader 245 passed/12 ignored (12.12 s), daemon 416 passed/123 ignored (0.75 s), daemon integrations green; replica CLI regression passes. Loopback byte identity/cancel/drain and trickle deadline pass. Actual SM121 byte/identity gate passes; concurrent/idle/off C1 gates pending. No placement-default promotion.

Build: matching dev image pair si-2c76d55 rebuilt/distributed to ostrich,dodo,emu,kiwi, exit 0. Standard WIP both-role build exit 0. Incremental MiMo LM exports (mimof,mimop) and expert packages (mimof:fp8,mimop:fp8) build exit 0, build.lock released; the first standard build omitted them. Shared program manifest: /wip/build/coordinator/native/dsv4_programs/dsv4_programs.json; shared mm-wp4 slot fingerprints coordinator 17b5c529820aa7410a58c390577ec56292e27f148577e694d5f7dff08f6efadd and Spark acc5ca3e0cb7b4acb80091a729ebdadee76e4e923ed45b2c8704ec292039437f. Four pool hosts have identical Spark artifacts; WP-5 reuses these native assets. Shared release config currently rejects MiMo SPARK_COUNT=2 unless EXPERT_FORMAT=exl3 (V4.1-specific validator); use 4-Spark build config and direct family min launcher, do not relax shared validator here.

Launcher green step: MiMo preflight resolves placement/hash before restart, sends selected same-process --encoder flags and coordinator peers/hash/revision, bounds expert readiness and skips planner/encoder startup entirely off. Explicit rtx[:gpu]/spark[:rank] validation added; V4.1 retains auto/off. Launcher 192 tests pass (18.83 s); full scripts 998 passed, 2 skipped, 193 subtests passed (71.93 s), no failing ids. WP-5 consumes adapter and owns API health integration. Idle-Spark policy unit tested; real idle-host inventory remains pending. Hardware: no reservation queued or held.

Matching-SM121 gate: ostrich native reference versus encoder-only TCP server, MiMo Flash MOPD revision 2479e2d0029eca9a34cc7e7f55a121925f81908e, si-2c76d55; 256/1024/4096-token BF16 outputs byte-identical including three interleaved measured samples after warm-up. Independent EncoderId d2f30e4914af1d1abf056c17a9539bf6b74d757bef48a6fa5a987db497e0afa0 matches server. Median wall/encode ms: 37.88/36.68, 168.18/159.73, 814.70/777.55. Median (wall-minus-encode)/encode: 3.29%, 4.98%, 4.66%; one 1024-token sample is 6.52%, so TCP is not uniformly below 5%. This includes host overhead, not pure NIC transfer. Both RoCE rails measured active at 200 Gb/s. Exact mm-wp4-reference and mm-wp4-encoder containers absent after bounded run; sparks.lock released. Qualification harness host tests: 7 passed in 0.17 s; full scripts 1001 passed, 2 skipped, 193 subtests passed in 75.82 s, no failing ids. Same-Spark C1 interference, idle-zero-slowdown, VISION=off runtime read accounting and actual idle-host launch remain open.
