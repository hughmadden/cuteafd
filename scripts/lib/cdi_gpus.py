"""GPUs reach containers through CDI (`--device nvidia.com/gpu=NAME`), never `--gpus`.

The legacy `--gpus` hook adds device rules behind systemd's back, so a
`systemctl daemon-reload` strips them from running containers (NVML dies).
A CDI device is passed by Docker itself and survives the reload. Shell twin:
scripts/lib/cdi-gpus.sh.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

SPECS = (Path('/var/run/cdi/nvidia.yaml'), Path('/etc/cdi/nvidia.yaml'))


def cdi_names(specs: Iterable[Path] = SPECS) -> set[str]:
    names: set[str] = set()
    for spec in specs:
        try:
            text = spec.read_text()
        except OSError:
            continue
        names.update(m.group(1) for m in re.finditer(r'^\s*-?\s*name:\s*["\']?([^"\'\s]+)["\']?\s*$', text, re.M))
    return names


def cdi_gpu_args(gpus: Iterable[object] | str, specs: Iterable[Path] = SPECS) -> list[str]:
    """`--device nvidia.com/gpu=X` per GPU (index, GPU-uuid, or 'all'); raises if the spec lacks one."""
    wanted = [g.strip() for g in (gpus.split(',') if isinstance(gpus, str) else map(str, gpus))]
    if not wanted or not all(wanted):
        raise ValueError('empty GPU request')
    have = cdi_names(specs)
    missing = [] if os.environ.get('CUTEAFD_CDI_CHECK') == 'off' else [g for g in wanted if g not in have]  # off: script tests only
    if missing:
        raise RuntimeError(f'CDI device nvidia.com/gpu={",".join(missing)} is not in /var/run/cdi/nvidia.yaml; '
                           'run: sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml')
    return [arg for g in wanted for arg in ('--device', f'nvidia.com/gpu={g}')]
