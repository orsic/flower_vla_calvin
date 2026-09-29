"""Regression test for libero_venv.py's file-path import fallback.

flower.evaluation.libero_venv is reused by the MimicGen eval, which may run in a
container where `libero.libero.envs`'s package __init__ fails to import (it pulls in
robosuite-1.4.0-era code, incompatible with MimicGen's newer robosuite pin) or where
`libero` isn't installed at all. venv.py itself has no robosuite dependency, so the
module should fall back to loading it directly by file path and still expose the same
five names.
"""

import builtins
import importlib
import sys


def test_module_loads_via_fallback_when_libero_envs_package_import_fails(monkeypatch):
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "libero.libero.envs.venv" or name.startswith("libero.libero.envs"):
            raise ImportError("simulated: libero.libero.envs fails under MimicGen's robosuite pin")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    sys.modules.pop("flower.evaluation.libero_venv", None)

    try:
        module = importlib.import_module("flower.evaluation.libero_venv")
        for name in (
            "BaseVectorEnv",
            "CloudpickleWrapper",
            "DummyEnvWorker",
            "DummyVectorEnv",
            "SubprocEnvWorker",
        ):
            assert hasattr(module, name), f"missing {name} after fallback import"
    finally:
        # monkeypatch.undo() first: it only reverts automatically *after* this test
        # function returns, so re-importing under `finally` while still patched would
        # just take the fallback branch again and leave a fallback-bound module cached
        # for every later test in this process (confirmed as an actual regression:
        # tests/test_batched_eval.py's spawn-based tests failed to pickle
        # _libero_envs_venv.CloudpickleWrapper once that happened).
        monkeypatch.undo()
        sys.modules.pop("flower.evaluation.libero_venv", None)
        sys.modules.pop("_libero_envs_venv", None)
        importlib.import_module("flower.evaluation.libero_venv")
