#!/usr/bin/env python3
"""The launch family table: which family a checkpoint is, and which decoder
layers carry routed experts.

  checkpoint-family.py CONFIG_JSON   -> "FAMILY MODEL_TYPE FIRST_LAYER LAST_LAYER"

FAMILY is the long family id (deepseek_v41, deepseek_v4, glm5, glm5_flash,
mimo_v2, qwen4). The Spark ranks serve routed experts on [FIRST, LAST]
(LAST -1: through the end); MTP/nextn layers past the decoder stay off them for
GLM 5.3 Flash and Qwen 3.8, whose serve paths do not run their MTP heads.
"""
from __future__ import annotations

import json
import sys

FAMILIES = {
    "deepseek_v41": "deepseek_v41",
    "deepseek_v4": "deepseek_v4",
    "glm_moe_dsa": "glm5",
    "glm5_next": "glm5_flash",
    "mimo_v2_flash": "mimo_v2",
    "mimo_v2": "mimo_v2",
    "qwen4_exp": "qwen4",
}


def describe(config: dict) -> tuple[str, str, int, int]:
    text = config.get("text_config", config)
    kind = config.get("model_type", "?")
    family = FAMILIES.get(kind)
    if family is None:
        raise SystemExit(f"unsupported model_type {kind!r}; known: {', '.join(sorted(FAMILIES))}")
    first, last = 0, -1
    if family in ("glm5", "glm5_flash"):
        types = text.get("mlp_layer_types")
        first = types.index("sparse") if types else text.get("first_k_dense_replace", 0)
    elif family == "mimo_v2":
        first = text["moe_layer_freq"].index(1)
    if family in ("glm5_flash", "qwen4"):
        last = text["num_hidden_layers"] - 1
    return family, kind, first, last


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        print(*describe(json.load(handle)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
