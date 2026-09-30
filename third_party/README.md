# Third-party source

All four dependency directories are pinned Git submodules. Initialize the
complete source graph after cloning CuteAFD:

```bash
git submodule update --init --recursive
```

`sparkinfer/` is pinned from
<https://github.com/tpurtell/sparkinfer-glmrt>. CuteAFD uses that source for
every Spark and coordinator CuTe AOT export; an independently installed
`b12x` or `sparkinfer` package is not a supported build input.

Initialize it after cloning CuteAFD:

```bash
git submodule update --init --recursive third_party/sparkinfer
python3 scripts/verify-sparkinfer-source.py \
  --source third_party/sparkinfer \
  --lock third_party/sparkinfer.lock.json
```

The lock records both the fork commit and a deterministic content digest.
The digest keeps release archives verifiable after Git metadata is removed.
When intentionally updating the pin, update the submodule first, obtain the
new digest with `--print-tree-sha256`, update the lock, then run the full
verification command. The schema is:

```json
{
  "schema": 1,
  "repository": "https://github.com/tpurtell/sparkinfer-glmrt.git",
  "revision": "<lowercase 40-hex commit>",
  "source_tree_sha256": "<lowercase 64-hex digest>"
}
```

Generate the digest from a clean checkout; full verification also rejects a
wrong Git origin, a different `HEAD`, and tracked or non-ignored untracked
source changes. Never point a build at an unverified cache checkout.

`xgrammar/` is the pinned constrained-decoding implementation. It owns nested
source dependencies, which is why the top-level initialization command uses
`--recursive`.

```bash
python3 scripts/verify-xgrammar-source.py \
  --source third_party/xgrammar \
  --lock third_party/xgrammar.lock.json
```

`gptqmodel/` is the pinned calibration and conversion engine used by the
quantization workflow (`quantization/`, `docker/Dockerfile.quant-*`). It points
at the user's fork so CuteAFD can carry source-decoding, routed-only inclusion,
evidence, and resume changes while they are qualified for upstreaming.

`transformers/` is the pinned model-code fork the quantization coordinator
image installs for checkpoints that upstream Transformers does not yet load.
