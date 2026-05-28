"""Tests for the ``RELIQUARY_ENVIRONMENT_NAME`` env-var override on the
``mine`` / ``validate`` CLI commands.

The historical default for ``--environment`` is the constant
``reliquary.constants.ENVIRONMENT_NAME``, which means flipping the
prod env requires editing constants.py and redeploying. After the v2.3
cutover this turned the OpenMathInstruct rollout into a coordinated
multi-stage deploy when it should be a one-line env-file change.

Adding ``RELIQUARY_ENVIRONMENT_NAME`` lets operators flip the active
env on validator + miners with just a systemd restart, no code push.
The constant remains the fallback so any forgotten env var still
resolves to the canonical default.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


def _reload_cli_main():
    """Reload the CLI module so typer.Option defaults are re-evaluated
    against the current process environment. The Option default value
    is captured at function-decoration time (module load), so the only
    way to test different env-var states is to reload the module.
    """
    # Drop the cached module so the next import re-runs the decorators.
    sys.modules.pop("reliquary.cli.main", None)
    return importlib.import_module("reliquary.cli.main")


def _get_environment_option_default(cli_module, command_name: str) -> str:
    """Reach into the typer command's parameter list and pull out the
    Option default for ``--environment``. Typer stores the click params
    on the registered command's ``params`` list; the Option default is
    on the ``default`` attribute of the matching one.
    """
    # Find the command registered with typer
    for cmd in cli_module.app.registered_commands:
        if cmd.callback.__name__ == command_name:
            # The callback's signature carries the typer Option default
            # via inspect: each parameter's default is the typer.Option
            # instance, which has a ``.default`` attribute.
            import inspect
            sig = inspect.signature(cmd.callback)
            return sig.parameters["environment"].default.default
    raise AssertionError(f"command {command_name!r} not found in app")


def test_mine_environment_defaults_to_constant_when_unset(monkeypatch):
    """When ``RELIQUARY_ENVIRONMENT_NAME`` is not in the environment the
    ``--environment`` default falls back to ``ENVIRONMENT_NAME`` from
    constants. This preserves the existing CI / fresh-host behaviour
    (the constant is the canonical declaration; the env var is a per-
    deploy override knob)."""
    monkeypatch.delenv("RELIQUARY_ENVIRONMENT_NAME", raising=False)
    cli = _reload_cli_main()
    from reliquary.constants import ENVIRONMENT_NAME
    assert _get_environment_option_default(cli, "mine") == ENVIRONMENT_NAME


def test_validate_environment_defaults_to_constant_when_unset(monkeypatch):
    """Same fallback on the trainer/validator subcommand."""
    monkeypatch.delenv("RELIQUARY_ENVIRONMENT_NAME", raising=False)
    cli = _reload_cli_main()
    from reliquary.constants import ENVIRONMENT_NAME
    assert _get_environment_option_default(cli, "validate") == ENVIRONMENT_NAME


def test_mine_environment_picks_up_env_var(monkeypatch):
    """Setting ``RELIQUARY_ENVIRONMENT_NAME=foo`` makes the miner CLI
    default to ``foo``. This is the prod-deploy ergonomic: flip the
    env file and restart, no code push."""
    monkeypatch.setenv("RELIQUARY_ENVIRONMENT_NAME", "openmathinstruct")
    cli = _reload_cli_main()
    assert _get_environment_option_default(cli, "mine") == "openmathinstruct"


def test_validate_environment_picks_up_env_var(monkeypatch):
    """Same on the trainer/validator subcommand — same env var name,
    same precedence rule, so a single ``RELIQUARY_ENVIRONMENT_NAME=foo``
    flips both validator + miner without per-process plumbing."""
    monkeypatch.setenv("RELIQUARY_ENVIRONMENT_NAME", "openmathinstruct")
    cli = _reload_cli_main()
    assert _get_environment_option_default(cli, "validate") == "openmathinstruct"


def test_env_var_takes_precedence_over_constant(monkeypatch):
    """If the operator chose a value, that wins over the constant —
    otherwise the override would be useless. Pin this with an env value
    that is definitely not the constant default."""
    monkeypatch.setenv("RELIQUARY_ENVIRONMENT_NAME", "math-legacy-fake")
    cli = _reload_cli_main()
    assert _get_environment_option_default(cli, "mine") == "math-legacy-fake"
    assert _get_environment_option_default(cli, "validate") == "math-legacy-fake"


@pytest.fixture(autouse=True)
def _cleanup_module_cache():
    """Make sure each test re-imports cleanly — leaving a side-effecting
    typer.Option in sys.modules across tests would let one test's env-
    var setting leak into another."""
    yield
    sys.modules.pop("reliquary.cli.main", None)
