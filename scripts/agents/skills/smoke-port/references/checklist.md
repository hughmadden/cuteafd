# Registration checklist: every place a new family is wired in

Derived by grepping `step5` and `kimi` outside their own directories at upstream `main`
`1adc5bfb3`, plus the Kimi and Step-5 launch commits and MiniMax's port. **[S]** smoke, **[D]**
deferrable, **[X]** only with that feature. `<fam>` is the long id, `<tag>` the short kernel and
package tag. Files and symbols were checked to exist on that tip; find lines by grep. When in doubt,
`git grep -n <sibling> -- ':!*/<sibling>/*'` shows the current full list: MiniMax touched about 60
shared files, Step-5 is named in 72 non-test files outside its own directories.

## A. Recognition and planning (CPU only; first)

1. [S] `cuteafd plan MODEL` on the stub: with no family it prints the registration hint.
2. [S] Config reader `L/families/<fam>/config.rs`, `pub mod` in `L/families/mod.rs`. Every
   unsupported semantic is a typed refusal.
3. [S] Descriptor `L/plan/families/<fam>.rs` implementing `Family`: `id`, `detect`, `open`,
   `classify`, `component_hint`, `decode_lanes` = `One { reason }` citing an issue, `spark_decode`,
   `prefill_limits`, `expert_catalog`. Add `pub mod` in `L/plan/families/mod.rs`.
4. [S] `REGISTRY` in `L/plan/family.rs` (bump its length).
5. [S] `FamilyModel::accepts` (tensor contract) and `experts` (an `ExpertContract` with package
   `<tag>:nvfp4` or `:fp8`, and `spark_worlds`).
6. [S] `FamilyModel::cache_geometry`. Add a `CacheOptions` field in `L/serving_capacity.rs` only if
   the cache format is selectable; add the family to `compiled_index_extent_required` if it has a
   compiled index extent.
7. [S] Placement adapter `FamilyModel::placement` over the shared solver; executor modes in
   `L/placement/families/mod.rs`.
8. [S] Spark recipe: the family in the NVFP4/FP8 lists and packaged worlds in `L/plan/experts.rs`;
   rows in `L/plan/layout/spark_slot_census.json` (the planner refuses without them); explicit
   `spark_endpoints` / `spark_slot_bytes` arms in `L/plan/layout.rs` (an unknown family gets 0,
   silently).
9. [S] Launch description: the `model_type` arm in `L/plan/launch.rs`, its Python twin
   `scripts/lib/checkpoint-family.py` (`FAMILIES`, `describe`, `routed_layers`) and cases in
   `rust/crates/cuteafd-loader/tests/fixtures/launch-families.json`.
10. [X] Draft ownership `DraftOwnership` (`L/plan/spec.rs`), plan speculator defaults in
    `D/commands/plan.rs`.
11. [S] Family-pinned planner tests: the `decode_lanes.rs` reason tests and the `deployment.rs`
    family list.

## B. Core geometry

12. [S] `ExpertGeometry::<FAM>` const in `C/expert_geometry.rs`, its tag in `family()`, its clamp in
    `swiglu_limit()` (**unlisted shapes clamp at 10**), its arithmetic in `activation()`. Add a new
    `FfnActivation` variant (`C/ffn_activation.rs`) only for new arithmetic.

## C. Spark experts

13. [S] SparkInfer `GEOMETRIES["<tag>_nvfp4"]` in `b12x/integration/cuteafd/fp8_moe.py`; land it on
    SparkInfer first, then bump the pin and tree lock.
14. [S] Packager choices in `python/tools/aot/package_fp8_moe_aot.py`; the CMake regex in
    `native/cmake/shared/fp8_moe.cmake` (and the exclusion in `expert_families.cmake`).
15. [S] Expert catalog arm in `read_expert_catalog_with_checkpoint`
    (`L/families/deepseek_v41/v41_catalog.rs`, before the `text_config` → V4.1 fallthrough);
    name and prefix rules in `L/formats/fp8_experts.rs` (`read_selected` expects HF names).
16. [D] RTX-onboarded expert packages.

## D. Coordinator daemon

17. [S] `D/families/<fam>/`: `mod.rs`, weights and admission, the layer chain, `backend.rs`
    (`impl Backend`), `serve.rs` (`ServeArgs`, `run_serve`), `golden.rs` (`GoldenArgs`); module in
    `D/families/mod.rs`.
18. [S] CLI variants `Serve<Fam>` and `<Fam>Golden` in `D/cli.rs`; dispatch, budget and memory-report
    arms in `D/main.rs`; the `FAMILIES` table and help text in `D/commands/family.rs`.
19. [S] Prefill through `PrefillHooks` + `run_prefill_units`; decode through the shared scheduler,
    or `LaneSet::run_one` + `lanes::commit`.
20. [S] Source gates: the family file in `D/shared/spark_waves_tests.rs`; readiness tests in
    `D/shared/serve.rs` (shared-scheduler families are covered by the scheduler's own tests).
21. [X] An FFI module (`rust/crates/cuteafd-ffi/src/families/`) only for entry points outside the
    generic program table.
22. [D] Vision tower arm in `D/shared/vision.rs`; golden scorer hooks.

## E. API

23. [S] `A/openai/chat/<fam>.rs` (+ parser, prompt, fixtures and goldens); `pub mod` in
    `A/openai/chat.rs`.
24. [S] `ModelEncoding::<Fam>` with its thinking-off, EOS and think-close arms; stop ids;
    `Templated` / `OutputProcessor` variants and dispatch (`A/openai.rs`).
25. [X] A `ToolSyntax` variant (`A/openai/tools.rs`) and constraint/grammar arms for tool calls.
26. [S] Tokenizer: anything but `tokenizer.json` needs a shim like `L/tokenizer/tiktoken.rs`, hooked
    in `L/tokenizer.rs`.

## F. Native build and AOT

27. [S] Coordinator programs: a `CUTEAFD_ENABLE_<FAM>_AOT` option and the OR-lists in
    `native/CMakeLists.txt`; geometry names and exporter `DEPENDS` in
    `native/cmake/shared/dsv4_programs.cmake`.
28. [S] Exporter: geometry context, accepted names and builder dispatch in
    `python/tools/aot/export_b12x_dsv4_aot.py`, or a family exporter like
    `export_b12x_step5_aot.py`, plus SparkInfer integration geometry.
29. [S] Family listings in `scripts/build/compare-sm-exports.py`.

## G. Build and launch scripts

30. [S] WIP scopes: `PROGRAM_SWITCHES` and `family_scopes` in `scripts/lib/wip_aot_scopes.py` (add a
    test that each emitted scope is in the CMake list); CMake pass-through, ON-check and package
    names in `scripts/build/build-wip-artifacts.sh`; the BF16-sibling regex in `wip.sh`.
31. [D] Release: `scripts/build/build-release-artifacts.sh`, `scripts/build/release-scope.py`,
    `build.sh`.
32. [S] `scripts/launch/run-family.sh`: allowed keys, `family_prefill_mode`, attention placement,
    serve command, text-only guard, backend and Spark-count guards, speculator table, prefix args,
    packing, family args, worker endpoint geometry, plan args. New keys in `release_known_key`
    (`scripts/lib/release-common.sh`).
33. [S] `examples/configs/<fam>-<quant>-{min,max}.config` and `examples/configs/README.md`; the
    installed-example sweep in `rust/crates/cuteafd-daemon/tests/budget_cli.rs` picks them up.

## H. Gates and docs

34. [S] Script tests that name families: `test_serve_launch_new_families.py`,
    `test_run_family_dispatch.py`, `test_example_configs.py`, `test_wip_family_aot.py`,
    `test_release_derived_scope.py`, `test_compare_sm_exports.py`, `test_golden_run.py`.
35. [S] `SUBCOMMANDS` in `scripts/bench/golden-run.py`; the gate card in
    `scripts/bench/kits/gate/matrix-gate.json`.
36. [D] `FULL_TESTS` in `scripts/dev/cpu-gate`; release smoke JSON; fidelity publication pins.
37. [S] `docs/models/<fam>.md` linking the example configs (`test_example_configs.py` checks the
    links), with the smoke section and the deferral list.
