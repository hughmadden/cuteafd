#!/usr/bin/env bash
# Print the release-relevant identity of one image as KEY=value lines.
# Read-only. Used by scripts/release/verify-release-artifacts.sh locally and over
# SSH on the Spark experts, so its output is stable and easy to grep.
#
# Usage: scripts/release/image-identity-probe.sh IMAGE_REF
set -euo pipefail

image="${1:?usage: image-identity-probe.sh IMAGE_REF}"

docker image inspect "$image" >/dev/null 2>&1 || {
  echo "image=$image"
  echo "status=missing"
  exit 0
}

inspect() { docker image inspect -f "$1" "$image" 2>/dev/null || true; }
label() {
  local value
  value="$(inspect "{{index .Config.Labels \"$1\"}}")"
  [[ "$value" == "<no value>" ]] && value=""
  printf '%s' "$value"
}

echo "image=$image"
echo "status=present"
echo "id=$(inspect '{{.Id}}')"
echo "architecture=$(inspect '{{.Architecture}}')"
echo "os=$(inspect '{{.Os}}')"
echo "created=$(inspect '{{.Created}}')"
echo "size_bytes=$(inspect '{{.Size}}')"
echo "repo_digests=$(inspect '{{join .RepoDigests " "}}')"
echo "label.org.opencontainers.image.revision=$(label org.opencontainers.image.revision)"
echo "label.org.opencontainers.image.version=$(label org.opencontainers.image.version)"
echo "label.io.cuteafd.sparkinfer.revision=$(label io.cuteafd.sparkinfer.revision)"
echo "label.io.cuteafd.cuda_arch=$(label io.cuteafd.cuda_arch)"
echo "label.io.cuteafd.role=$(label io.cuteafd.role)"
echo "label.io.cuteafd.v41.spark_tp_roles=$(label io.cuteafd.v41.spark_tp_roles)"
echo "label.io.cuteafd.source-manifest.sha256=$(label io.cuteafd.source-manifest.sha256)"
