#!/usr/bin/env bash
set -euo pipefail

if [[ -d /workspace/cuteafd ]]; then
  export PATH="/workspace/cuteafd/scripts:$PATH"
  cd /workspace/cuteafd
fi
exec "$@"
