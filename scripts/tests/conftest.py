"""Fixtures shared by every script test."""
from __future__ import annotations

import pytest

# Startup files a shell runs before its first command: every non-interactive bash runs
# $BASH_ENV, an interactive POSIX shell $ENV. NGC-based images export
# BASH_ENV=/etc/bash.bashrc and ENV=/etc/shinit_v2, and that startup code calls
# `nvidia-smi -q -d COMPUTE`. The launcher tests put a bash-script `nvidia-smi` stub first
# on PATH, so inside such an image every stub call ran the startup code again and called
# itself, without end. No script test needs a startup file, so none gets one.
SHELL_STARTUP_VARIABLES = ("BASH_ENV", "ENV")


@pytest.fixture(autouse=True)
def no_shell_startup_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the shell startup variables for the duration of every test.

    Helpers that pass `{**os.environ, ...}`, `dict(os.environ)`, `os.environ.copy()` or
    no `env` at all read the environment when they launch, so their shells start without
    them; helpers that build their environment from scratch never carried them."""
    for name in SHELL_STARTUP_VARIABLES:
        monkeypatch.delenv(name, raising=False)
