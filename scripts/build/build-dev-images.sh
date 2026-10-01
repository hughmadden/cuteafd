#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/scripts/lib/release-common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/build/build-dev-images.sh [--config FILE] [--spark-hosts HOST,...]
                                         [--role coordinator|expert|both] [--dry-run]

Rebuilds the shared development images at the SparkInfer revision pinned by
this checkout (third_party/sparkinfer.lock.json): COORDINATOR_DOCKER_DEV here
(SM120) and SPARK_EXPERT_DOCKER_DEV natively on the first Spark host (SM121),
then loads the Spark image on every other Spark host. Run it after every
SparkInfer pin bump: wip.sh and run.sh --wip refuse a development image whose
io.cuteafd.sparkinfer.revision label differs from the lock.

Each image is also tagged <name>:si-<rev7>; the image it replaces keeps a
<name>:si-<oldrev7> tag. Running containers keep their image, so nothing is
stopped; WIP containers made from the old image need ./wip.sh --recreate.
--spark-hosts defaults to every SPARK_N_HOST in the configuration, active or
not; hosts already holding the new image are skipped. Serialize with other
Spark work (wrap in flock ~/.cache/cuteafd/sparks.lock when agents share it).
CUTEAFD_DEV_IMAGE_REMOTE_DIR selects the Spark staging directory (default
~/.cache/cuteafd/builds/devimg-<rev12> on the first Spark host).
EOF
}

config="$repo_root/cuteafd.config"
hosts_csv=""
role=both
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?$1 requires a configuration file}"; shift 2 ;;
    --spark-hosts) hosts_csv="${2:?$1 requires a comma-separated host list}"; shift 2 ;;
    --role) role="${2:?$1 requires coordinator, expert, or both}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) release_die "unknown argument: $1" ;;
  esac
done
case "$role" in coordinator|expert|both) ;; *) release_die "--role must be coordinator, expert, or both" ;; esac

release_load_config "$config"
for tool in docker ssh rsync python3; do release_need "$tool"; done

# The source must match the lock before anything is built from it. Python
# caches are fine here: .dockerignore and the Spark staging sync drop them,
# and the image build re-checks the copied tree without them.
revision="$(python3 "$repo_root/scripts/build/verify-sparkinfer-source.py" \
  --source "$repo_root/third_party/sparkinfer" \
  --lock "$repo_root/third_party/sparkinfer.lock.json" \
  --print-revision)"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || release_die "cannot read the pinned SparkInfer revision"

if [[ -n "$hosts_csv" ]]; then
  IFS=, read -ra spark_hosts <<<"$hosts_csv"
else
  spark_hosts=()
  for i in 0 1 2 3 4 5; do
    name="SPARK_${i}_HOST"
    [[ -z "${!name:-}" ]] || spark_hosts+=("${!name}")
  done
fi
if [[ "$role" != coordinator ]]; then
  ((${#spark_hosts[@]})) || release_die "no Spark hosts to build on"
  seed_host="${spark_hosts[0]}"
fi
remote_dir="${CUTEAFD_DEV_IMAGE_REMOTE_DIR:-}"

# <name>:si-<rev7>, keeping a registry port out of the tag split.
versioned_tag() {
  local image="$1" rev="$2" repo="$1"
  [[ "${image##*/}" != *:* ]] || repo="${image%:*}"
  printf '%s:si-%s\n' "$repo" "${rev:0:7}"
}

# Runs on the image's host (bash -c locally, bash -s over SSH): keep the image being
# replaced reachable by its revision before the name moves.
# shellcheck disable=SC2016
retag_body='set -euo pipefail
image="$1"; versioned="$2"; repo="$3"
new_id="$(docker image inspect -f "{{.Id}}" "$versioned")"
old_id="$(docker image inspect -f "{{.Id}}" "$image" 2>/dev/null || true)"
if [[ -n "$old_id" && "$old_id" != "$new_id" ]]; then
  old_rev="$(docker image inspect -f "{{index .Config.Labels \"io.cuteafd.sparkinfer.revision\"}}" "$image")"
  [[ -z "$old_rev" || "$old_rev" == "<no value>" ]] || docker tag "$old_id" "$repo:si-${old_rev:0:7}"
fi
docker tag "$versioned" "$image"'

repo_of() { local image="$1"; [[ "${image##*/}" != *:* ]] && echo "$image" || echo "${image%:*}"; }

label_of() {  # host|"" image
  local cmd="docker image inspect -f '{{.Id}} {{index .Config.Labels \"io.cuteafd.sparkinfer.revision\"}}' '$2'"
  if [[ -z "$1" ]]; then bash -c "$cmd"; else ssh -o BatchMode=yes "$1" "$cmd"; fi
}

coordinator_versioned="$(versioned_tag "$COORDINATOR_DOCKER_DEV" "$revision")"
spark_versioned="$(versioned_tag "$SPARK_EXPERT_DOCKER_DEV" "$revision")"
echo "SparkInfer pin: $revision"
[[ "$role" == expert ]] || echo "  coordinator: $COORDINATOR_DOCKER_DEV (+ $coordinator_versioned) on $(hostname)"
[[ "$role" == coordinator ]] ||
  echo "  Spark: $SPARK_EXPERT_DOCKER_DEV (+ $spark_versioned) built on $seed_host, loaded on $(IFS=,; echo "${spark_hosts[*]}")"
if ((dry_run)); then
  echo "Dry run: nothing built."
  exit 0
fi

if [[ "$role" != expert ]]; then
  echo "== building $COORDINATOR_DOCKER_DEV =="
  docker build \
    --build-arg CUTEAFD_ROLE=coordinator \
    --build-arg CUDA_ARCH=120 \
    --build-arg TARGET_PLATFORM=linux/amd64 \
    --build-arg CUTEAFD_SPARKINFER_COMMIT="$revision" \
    -f "$repo_root/docker/Dockerfile.dev" \
    -t "$coordinator_versioned" \
    "$repo_root"
  bash -c "$retag_body" _ "$COORDINATOR_DOCKER_DEV" "$coordinator_versioned" "$(repo_of "$COORDINATOR_DOCKER_DEV")"
fi

if [[ "$role" != coordinator ]]; then
  if [[ -z "$remote_dir" ]]; then
    remote_dir="$(ssh -o BatchMode=yes "$seed_host" 'printf %s "$HOME"')/.cache/cuteafd/builds/devimg-${revision:0:12}"
  fi
  release_validate_path_setting CUTEAFD_DEV_IMAGE_REMOTE_DIR "$remote_dir"
  ssh -o BatchMode=yes "$seed_host" python3 - "$remote_dir" <"$repo_root/scripts/build/assert-build-filesystem.py"
  echo "== staging the Dockerfile.dev context on $seed_host:$remote_dir =="
  ssh -o BatchMode=yes "$seed_host" "mkdir -p '$remote_dir'"
  # Dockerfile.dev reads only these paths; the SparkInfer tree is reconciled
  # with --delete-excluded so stale bytecode cannot survive into the image.
  (cd "$repo_root" && rsync -aR --delete --delete-excluded \
    --exclude .git --exclude __pycache__/ --exclude '*.pyc' --exclude '*.pyo' \
    .dockerignore docker/ scripts/build/verify-sparkinfer-source.py \
    third_party/sparkinfer.lock.json third_party/sparkinfer/ \
    "$seed_host:$remote_dir/")
  echo "== building $SPARK_EXPERT_DOCKER_DEV natively on $seed_host =="
  ssh -o BatchMode=yes "$seed_host" bash -s -- "$remote_dir" "$revision" "$spark_versioned" <<'REMOTE'
set -euo pipefail
cd "$1"
python3 scripts/build/verify-sparkinfer-source.py \
  --source third_party/sparkinfer --lock third_party/sparkinfer.lock.json \
  --require-no-python-cache
docker build \
  --build-arg CUTEAFD_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg TARGET_PLATFORM=linux/arm64 \
  --build-arg CUTEAFD_SPARKINFER_COMMIT="$2" \
  -f docker/Dockerfile.dev -t "$3" .
REMOTE
  spark_repo="$(repo_of "$SPARK_EXPERT_DOCKER_DEV")"
  remote_retag() {
    ssh -o BatchMode=yes "$1" bash -s -- "$SPARK_EXPERT_DOCKER_DEV" "$spark_versioned" "$spark_repo" <<<"$retag_body"
  }
  remote_retag "$seed_host"

  seed_id="$(label_of "$seed_host" "$SPARK_EXPERT_DOCKER_DEV" | awk '{print $1}')"
  targets=()
  for host in "${spark_hosts[@]:1}"; do
    [[ "$(label_of "$host" "$SPARK_EXPERT_DOCKER_DEV" 2>/dev/null | awk '{print $1}')" == "$seed_id" ]] ||
      targets+=("$host")
  done
  if ((${#targets[@]})); then
    echo "== loading $spark_versioned on $(IFS=,; echo "${targets[*]}") =="
    pids=()
    for host in "${targets[@]}"; do
      (
        set -o pipefail
        if ssh -o BatchMode=yes "$seed_host" 'command -v rdmapipe >/dev/null' &&
          ssh -o BatchMode=yes "$host" 'command -v rdmapipe >/dev/null'; then
          ssh -o BatchMode=yes "$seed_host" "docker image save '$spark_versioned' | rdmapipe --send" |
            ssh -o BatchMode=yes "$host" 'rdmapipe --recv | docker image load'
        else
          ssh -o BatchMode=yes "$seed_host" "docker image save '$spark_versioned'" |
            ssh -o BatchMode=yes "$host" 'docker image load'
        fi
        remote_retag "$host"
      ) &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    ((failed == 0)) || release_die "Spark development image distribution failed"
  fi
fi

echo "== development images =="
status=0
check() {
  local where="$1" host="$2" image="$3" id rev
  read -r id rev < <(label_of "$host" "$image")
  echo "  $where $image $id sparkinfer=$rev"
  [[ "$rev" == "$revision" ]] || status=1
}
[[ "$role" == expert ]] || check "$(hostname)" "" "$COORDINATOR_DOCKER_DEV"
if [[ "$role" != coordinator ]]; then
  for host in "${spark_hosts[@]}"; do check "$host" "$host" "$SPARK_EXPERT_DOCKER_DEV"; done
fi
((status == 0)) || release_die "a development image does not carry SparkInfer $revision"
echo "Done. WIP containers made from older images need ./wip.sh --recreate."
