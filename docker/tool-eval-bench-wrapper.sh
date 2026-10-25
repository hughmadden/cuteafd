#!/bin/sh
# Installed as /opt/cuteafd/bin/tool-eval-bench by Dockerfile.release: the image's
# lock-verified tool-eval-bench source (third_party/tool-eval-bench.lock.json), run
# without a build step. Its dependencies come from docker/release-python-lock.json.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/opt/cuteafd/third_party/tool-eval-bench/src:/opt/cuteafd/lib/tool-eval-bench${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -P -c 'import sys; sys.argv[0] = "tool-eval-bench"; from tool_eval_bench.cli.dispatch import main; sys.exit(main())' "$@"
