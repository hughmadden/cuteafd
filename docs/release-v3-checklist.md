# v3 release cut

Run from the frozen checkout. This is a cut checklist, not permission to cut.
`[TJ]` requires explicit approval. Keep all machine-specific configuration,
logs and build outputs outside Git. No dense Qwen, Step-5 or V4.1 EXL3 card.

1. **Freeze:** merge every required #346 gate into `work/p0`, record the
   commit, and refresh PLAN's gate status. No intermediate v3 release.
   ```sh
   git fetch origin && git switch work/p0 && git pull --ff-only
   git rev-parse HEAD
   git submodule update --init --recursive
   scripts/dev/cpu-gate --task v3-cut
   python3 scripts/build/compare-sm-exports.py --sms 188,170
   ```
   No failing CPU test ids; review the object/cubin differences.
2. **RC pair [TJ for publication/tag]:** prepare private `$BUILD_CONFIG`
   with both inference images tagged `v3.0.0-rc1` and both build roots on
   guarded NVMe. `./build.sh --config "$BUILD_CONFIG"` derives both legs' export
   set from `scripts/bench/release-smoke-v3.json` (or
   `CUTEAFD_RELEASE_MATRIX`), with no hand-set export environment.
   Scope defaults to `all` and preserves the v2.0.0 export floor. Both images
   record the selected matrix's SHA256. Family/entry `--scope` builds may publish
   only `dev-*`/`wip-*` tags with `--no-latest`. Every other tag and any `latest`
   update requires `all`, the default matrix digest at the image's Git revision,
   and a passing full inventory check; a custom matrix cannot weaken publication.
   `./push-containers.sh --config "$BUILD_CONFIG" --no-latest v3.0.0-rc1`
   publishes only the RC tags; all RC spellings refuse `--latest`. `git tag -a v3.0.0-rc1 -m 'v3 release candidate 1' && git push origin v3.0.0-rc1`.
3. **Smoke:** copy `scripts/bench/release-smoke-v3.json` to private
   `$MATRIX`; replace image/build placeholders and portable Spark names/lanes.
   Split it by hardware before running: 32 PRO/reference entries locally,
   ten `*-5090` entries through Hugh's agent via the collaboration repo.
   Verify the latter's GPU names really say RTX 5090; the class marker is
   metadata, not hardware emulation. Four other 5090 cells stay pending
   (GLM 5.3 EXL3/NVFP4, V4 Pro, MiMo Pro: no demonstrated fit).
   ```sh
   cuteafd bench smoke --matrix "$MATRIX" --dry-run
   cuteafd bench smoke --matrix "$MATRIX"
   ```
   28 min/max + four extra references + ten real-5090 entries. GLM Flash
   EXL3 max is the 0-Spark reference; all-Spark and 22-local-layer references
   are additional. V4.1 adds TP4 min and TP6 max references. Every dual card
   drafts. Check actual ownership, admitted pool, quality and fabric before
   accepting exports; inability to admit the mixed reference blocks that card.
4. **Publish results:** update `benchmarks/current-matrix.json` PRO report
   pins to the accepted RC exports; replace a pending 5090 cell only with a
   real export. `cuteafd bench publish --matrix benchmarks/current-matrix.json`
   rebuilds the table/index. Edit highlights and one aggregated v3 changelog,
   not the table. Add model changelog rows only for changed paths with fresh
   basic evals (all six MoE families); verify generated links and known limits.
5. **Promote unchanged [TJ]:** tag the exact final RC commit `v3.0.0`,
   retag the same image digests in private `$PROMOTE_CONFIG`, then
   `./push-containers.sh --config "$PROMOTE_CONFIG" v3.0.0` (moves `latest`).
   `git tag -a v3.0.0 -m 'v3.0.0' && git push origin v3.0.0`.
   Update `main`/`release/v3` only with TJ's approved fast-forward/merge.
6. **GitHub release [TJ]:** `gh release create v3.0.0 --verify-tag --title v3.0.0 --notes-file "$NOTES"`.
   Notes: new features and capabilities only (short highlights, one changelog
   since v2, image digests). No performance comparisons with earlier releases.
   Remove task Cargo targets/staging and leave no serving containers behind.
