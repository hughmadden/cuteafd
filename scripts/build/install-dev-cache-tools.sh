#!/usr/bin/env bash
set -euo pipefail
# Only invoked by Dockerfile.dev; pins are checked in rather than trusting a mutable checksum URL.
case "$(uname -m)" in
  x86_64)
    triple=x86_64-unknown-linux-musl
    kache_sha=756e9701a6afb8354fd8b1d76164e13272d320354d84f01197e57d8a3b4be397
    sccache_sha=45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89
    ;;
  aarch64)
    triple=aarch64-unknown-linux-musl
    kache_sha=88abd848be7d300d4e30b8510ebdc45990dae3f96b6591e3498209639cf34701
    sccache_sha=2b3284d5da3b46a47dc4229e75bb7b88ac4aa99c8d754fb7d2f84997e5a4354a
    ;;
  *) printf 'unsupported cache-tool architecture\n' >&2; exit 1 ;;
esac
[[ "$KACHE_VERSION" == 1.0.0 && "$SCCACHE_VERSION" == 0.18.0 ]]
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
curl --retry 3 -fsSL "https://github.com/kunobi-ninja/kache/releases/download/v$KACHE_VERSION/kache-$triple.tar.gz" -o "$work/kache.tar.gz"
curl --retry 3 -fsSL "https://github.com/mozilla/sccache/releases/download/v$SCCACHE_VERSION/sccache-v$SCCACHE_VERSION-$triple.tar.gz" -o "$work/sccache.tar.gz"
printf '%s  %s\n' "$kache_sha" "$work/kache.tar.gz" "$sccache_sha" "$work/sccache.tar.gz" | sha256sum --check --strict
mkdir "$work/kache" "$work/sccache"
tar -xzf "$work/kache.tar.gz" -C "$work/kache"
tar -xzf "$work/sccache.tar.gz" -C "$work/sccache"
install -m 0755 "$(find "$work/kache" -type f -name kache)" /opt/cuteafd-kache
ln -s /opt/cuteafd-kache /usr/local/bin/kache
install -m 0755 "$(find "$work/sccache" -type f -name sccache)" /usr/local/bin/sccache
kache --version
sccache --version
