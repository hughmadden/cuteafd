#!/usr/bin/env python3
"""Build a Spark (SM121, GB10) EXL3 package on an SM120 host.

Runs ``package_v41_exl3_aot.py`` with its arguments after describing a GB10
compile target: Torch reports compute 12.1, 48 SMs and the GB10 name; CuTe
DSL compiles for ``sm_121a``; Triton (route preparation) targets 121. The
real SM120 device still hosts the exporter's buffer layout pass, which only
sizes buffers. B12x's offline compiler workers use the same target
description. Check a cross build against a native one before trusting it
(the Trellis objects of an existing native package must be byte-identical).

  exl3_cross_sm121.py build --role spark --geometry glm ... (package tool args)
"""
from __future__ import annotations

import sys

import torch

GB10 = {"major": 12, "minor": 1, "multi_processor_count": 48, "name": "NVIDIA GB10"}


def describe_gb10() -> None:
    real = torch.cuda.get_device_properties(0)
    if (real.major, real.minor) != (12, 0):
        raise SystemExit("cross SM121 builds run on an SM120 host")
    fields = {name: getattr(real, name) for name in dir(real) if not name.startswith("_")}
    fields.update(GB10)
    target = type("CrossTargetProperties", (), fields)()
    torch.cuda.get_device_properties = lambda device=None: target
    torch.cuda.get_device_capability = lambda device=None: (12, 1)
    torch.cuda.get_device_name = lambda device=None: GB10["name"]

    from cutlass.cutlass_dsl import CuTeDSL

    CuTeDSL._get_dsl().envar.arch = "sm_121a"

    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver

    active = driver.active
    active.get_current_target = lambda: GPUTarget("cuda", 121, 32)
    active.get_device_capability = lambda device=None: (12, 1)


def main() -> None:
    describe_gb10()
    import package_v41_exl3_aot

    sys.argv = [sys.argv[0], *sys.argv[1:]]
    package_v41_exl3_aot.main()


if __name__ == "__main__":
    main()
