#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
source "$repo_root/scripts/lib/build-supervision.sh"
usage() {
  printf '%s\n' 'Usage: build-dev-images.sh [--config FILE] [--spark-hosts HOST,...]' \
    '  [--role coordinator|expert|both] [--publish] [--dry-run]' \
    'Build toolchain-only dev images natively, independent of SparkInfer pins.' \
    '--publish uses only ghcr.io/tpurtell/cuteafd-dev:tc-HASH{-amd64,-arm64} and latest.' \
    'Publishing never retags local development defaults. Use rhea as the Spark seed.' \
    'Both architectures build concurrently; CUTEAFD_DEV_IMAGE_SEQUENTIAL=1 restores coordinator-first order.' \
    'Hold build.lock and the Spark seed host lock around actual builds.'
}
config="$repo_root/cuteafd.config"; hosts_csv=; role=both; dry_run=0; publish=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?}"; shift 2 ;;
    --spark-hosts) hosts_csv="${2:?}"; shift 2 ;;
    --role) role="${2:?}"; shift 2 ;;
    --publish) publish=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) release_die "unknown argument: $1" ;;
  esac
done
case "$role" in coordinator|expert|both) ;; *) release_die 'invalid --role' ;; esac
(( ! publish )) || [[ "$role" == both ]] || release_die 'publishing requires both architectures'
sequential="${CUTEAFD_DEV_IMAGE_SEQUENTIAL:-0}"
[[ "$sequential" == 0 || "$sequential" == 1 ]] || release_die 'CUTEAFD_DEV_IMAGE_SEQUENTIAL must be 0 or 1'
release_load_config "$config"
for tool in docker ssh rsync python3; do release_need "$tool"; done
if ((publish)); then
  [[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] ||
    release_die 'refusing to publish from a dirty tree'
fi
hash="$(python3 "$repo_root/scripts/build/dev-toolchain.py")"
revision="$(git -C "$repo_root" rev-parse HEAD)"
base_amd64="$(python3 "$repo_root/scripts/build/dev-toolchain.py" amd64)"
base_arm64="$(python3 "$repo_root/scripts/build/dev-toolchain.py" arm64)"
repo_of() { local image="$1"; [[ "${image##*/}" != *:* ]] && printf '%s' "$image" || printf '%s' "${image%:*}"; }
coordinator_tag="$(repo_of "$COORDINATOR_DOCKER_DEV"):tc-$hash"
spark_tag="$(repo_of "$SPARK_EXPERT_DOCKER_DEV"):tc-$hash"
package=ghcr.io/tpurtell/cuteafd-dev
if ((publish)); then
  coordinator_tag="$package:tc-$hash-amd64"; spark_tag="$package:tc-$hash-arm64"
fi
if [[ -n "$hosts_csv" ]]; then IFS=, read -ra spark_hosts <<<"$hosts_csv"
else
  spark_hosts=()
  for i in 0 1 2 3 4 5; do name="SPARK_${i}_HOST"; [[ -z "${!name:-}" ]] || spark_hosts+=("${!name}"); done
fi
if [[ "$role" != coordinator ]]; then
  ((${#spark_hosts[@]})) || release_die 'no Spark build host'
  seed_host="${spark_hosts[0]}"
fi
printf 'Toolchain: tc-%s\n  amd64: %s\n  arm64: %s\n' "$hash" "$coordinator_tag" "$spark_tag"
((!publish)) || printf '  index: %s:tc-%s\n  latest: %s:latest\n' "$package" "$hash" "$package"
case "$role:$sequential" in
  both:0) printf 'Plan: coordinator (amd64) and expert (arm64) legs build concurrently; checks/retag/distribution/publish wait for both.\n' ;;
  both:1) printf 'Plan: sequential: coordinator then expert (CUTEAFD_DEV_IMAGE_SEQUENTIAL=1); shared steps wait for both.\n' ;;
  *) printf 'Plan: %s leg only.\n' "$role" ;;
esac
if ((dry_run)); then printf 'Dry run: nothing built or pushed.\n'; exit 0; fi
# Preserve old si-* images for campaigns, and old tc-* images before moving a local default.
retag_body='set -euo pipefail
image="$1"; versioned="$2"; repo="$3"
old="$(docker image inspect -f "{{.Id}}" "$image" 2>/dev/null || true)"
if [[ -n "$old" ]]; then
  si="$(docker image inspect -f "{{index .Config.Labels \"io.cuteafd.sparkinfer.revision\"}}" "$old")"
  tc="$(docker image inspect -f "{{index .Config.Labels \"io.cuteafd.toolchain.hash\"}}" "$old")"
  [[ -z "$si" || "$si" == "<no value>" ]] || docker tag "$old" "$repo:si-${si:0:7}"
  [[ -z "$tc" || "$tc" == "<no value>" ]] || docker tag "$old" "$repo:tc-$tc"
fi
docker tag "$versioned" "$image"'
check_body='set -euo pipefail
[[ "$(docker image inspect -f "{{index .Config.Labels \"io.cuteafd.toolchain.hash\"}}" "$1")" == "$2" ]] || { printf "toolchain label mismatch\n" >&2; exit 1; }
[[ "$(docker image inspect -f "{{.Architecture}}" "$1")" == "$3" ]] || { printf "architecture mismatch\n" >&2; exit 1; }'
release_need mkfifo
log_root="${CUTEAFD_DEV_IMAGE_LOG_ROOT:-$HOME/.cache/cuteafd/builds/devimg}"
release_validate_path_setting CUTEAFD_DEV_IMAGE_LOG_ROOT "$log_root"
python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$log_root"
mkdir -p "$log_root"
leg_log_dir="$(mktemp -d "$log_root/build-legs.XXXXXXXX")"
run_id="${leg_log_dir##*/}"

build_coordinator_dev() (
  [[ "$(uname -m)" == x86_64 ]] || release_die 'coordinator build must be native amd64'
  DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="$base_amd64" --build-arg CUTEAFD_ENGINE_COMMIT="$revision" \
    --build-arg CUTEAFD_TOOLCHAIN_HASH="$hash" -f "$repo_root/docker/Dockerfile.dev" -t "$coordinator_tag" "$repo_root"
  bash -c "$check_body" _ "$coordinator_tag" "$hash" amd64
)
build_expert_dev() (
  [[ "$(ssh -o BatchMode=yes "$seed_host" uname -m)" == aarch64 ]] || release_die 'Spark build must be native arm64'
  remote_dir="${CUTEAFD_DEV_IMAGE_REMOTE_DIR:-$(ssh -o BatchMode=yes "$seed_host" 'printf %s "$HOME"')/.cache/cuteafd/builds/devimg-$hash}"
  release_validate_path_setting CUTEAFD_DEV_IMAGE_REMOTE_DIR "$remote_dir"
  ssh -o BatchMode=yes "$seed_host" python3 - "$remote_dir" <"$repo_root/scripts/build/assert-build-filesystem.py"
  ssh -o BatchMode=yes "$seed_host" "mkdir -p '$remote_dir'"
  (cd "$repo_root" && rsync -aR --delete --exclude .git .dockerignore docker/ \
    scripts/build/install-dev-cache-tools.sh "$seed_host:$remote_dir/")
  ssh -o BatchMode=yes "$seed_host" setsid --wait bash -s -- \
    "$run_id" "$remote_dir" "$base_arm64" "$revision" "$hash" "$spark_tag" <<'REMOTE'
set -euo pipefail
# Independent of the staging directory, so cancellation can arrive during rsync.
process_dir="$HOME/.cache/cuteafd/builds/devimg-processes"
process_file="$process_dir/$1.pid"
cancel_file="$process_dir/$1.cancel"
mkdir -p "$process_dir"
[[ ! -e "$cancel_file" ]] || exit 143
printf '%s\n' "$$" >"$process_file"
cleanup_dev_phase() {
  trap '' HUP INT TERM
  kill -TERM -- "-$$" 2>/dev/null || true
  rm -f "$process_file"
}
trap cleanup_dev_phase EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
# Close the race between the first marker check and PID publication.
[[ ! -e "$cancel_file" ]] || exit 143
cd "$2"
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="$3" --build-arg CUTEAFD_ENGINE_COMMIT="$4" \
  --build-arg CUTEAFD_TOOLCHAIN_HASH="$5" -f docker/Dockerfile.dev -t "$6" .
REMOTE
  ssh -o BatchMode=yes "$seed_host" bash -s -- "$spark_tag" "$hash" arm64 <<<"$check_body"
)
cancel_coordinator_dev() { :; } # The supervised group owns the local Docker client.
cancel_expert_dev() {
  # Killing the SSH client alone does not reliably stop a remote Docker build.
  timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=5 "$seed_host" bash -s -- "$run_id" <<'CANCEL'
set -euo pipefail
process_dir="$HOME/.cache/cuteafd/builds/devimg-processes"
mkdir -p "$process_dir"
: >"$process_dir/$1.cancel"
pid=""
[[ ! -f "$process_dir/$1.pid" ]] || read -r pid <"$process_dir/$1.pid" || true
if [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/cmdline" ]] &&
   grep -zFq -- "$1" "/proc/$pid/cmdline"; then
  kill -TERM -- "-$pid" 2>/dev/null || true
  for ((i=0; i<5; i++)); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
fi
rm -f "$process_dir/$1.pid"
CANCEL
}
case "$role" in
  both) build_supervise 'dev image build' "$leg_log_dir" "$sequential" \
    coordinator build_coordinator_dev cancel_coordinator_dev amd64.log \
    expert build_expert_dev cancel_expert_dev arm64.log ;;
  coordinator) build_supervise 'dev image build' "$leg_log_dir" 0 \
    coordinator build_coordinator_dev cancel_coordinator_dev amd64.log ;;
  expert) build_supervise 'dev image build' "$leg_log_dir" 0 \
    expert build_expert_dev cancel_expert_dev arm64.log ;;
esac
# Shared mutations happen only after every selected leg's build and check pass.
if ((!publish)); then
  if [[ "$role" != expert ]]; then
    bash -c "$retag_body" _ "$COORDINATOR_DOCKER_DEV" "$coordinator_tag" "$(repo_of "$COORDINATOR_DOCKER_DEV")"
  fi
  if [[ "$role" != coordinator ]]; then
    ssh -o BatchMode=yes "$seed_host" bash -s -- "$SPARK_EXPERT_DOCKER_DEV" "$spark_tag" "$(repo_of "$SPARK_EXPERT_DOCKER_DEV")" <<<"$retag_body"
    for host in "${spark_hosts[@]:1}"; do
      ssh -o BatchMode=yes "$seed_host" "docker image save '$spark_tag'" | ssh -o BatchMode=yes "$host" 'docker image load'
      ssh -o BatchMode=yes "$host" bash -s -- "$SPARK_EXPERT_DOCKER_DEV" "$spark_tag" "$(repo_of "$SPARK_EXPERT_DOCKER_DEV")" <<<"$retag_body"
    done
  fi
fi
if ((publish)); then
  # Validate both before pushing either; publish only this explicitly approved package.
  bash -c "$check_body" _ "$coordinator_tag" "$hash" amd64
  ssh -o BatchMode=yes "$seed_host" bash -s -- "$spark_tag" "$hash" arm64 <<<"$check_body"
  docker push "$coordinator_tag"
  ssh -o BatchMode=yes "$seed_host" docker push "$spark_tag"
  docker buildx imagetools create -t "$package:tc-$hash" -t "$package:latest" "$coordinator_tag" "$spark_tag"
  for tag in "$coordinator_tag" "$spark_tag" "$package:tc-$hash"; do docker buildx imagetools inspect "$tag"; done
  # Use a temporary empty Docker config so the visibility probe cannot reuse login credentials.
  anon="$(mktemp -d)"; trap 'rm -rf "$anon"' EXIT
  if DOCKER_CONFIG="$anon" docker pull "$package:tc-$hash" >/dev/null; then
    printf 'Anonymous manifest access: public\n'
  else
    printf 'Anonymous access failed; if private, TJ must flip package visibility in GitHub UI.\n' >&2
  fi
fi
printf 'Done. Recreate WIP containers only when changing toolchains.\n'
