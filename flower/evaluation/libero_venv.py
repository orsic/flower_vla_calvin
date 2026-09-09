"""Start-method-aware LIBERO vector env factory.

`libero.libero.envs.venv.SubprocVectorEnv` creates its worker subprocesses via
`multiprocessing.context.Process`, i.e. the process-global default start
method (fork on Linux). `flower_eval_libero.py` has historically fallen back
to `DummyVectorEnv` (all envs stepped sequentially in the parent process)
because of that, attributing the failure to CUDA already being initialized in
the parent.

Empirically (see the plan / commit this file lands with) the real constraint
is broader: a forked child that tries to create its own EGL rendering context
fails with EGL_BAD_ALLOC whenever the parent process has already touched
GL/EGL — which happens as soon as `robosuite`/`libero.libero.envs` is
imported, independent of CUDA. Reusing a `forkserver` (fork children from a
preloaded, otherwise-idle control process) does not sidestep this: even with
GL modules excluded from the preload list, forking several workers back to
back fires their `eglCreateContext` calls at nearly the same instant, and the
NVIDIA driver's context creation is not safe under that concurrency — verified
to fail here at N=10 concurrent forkserver children.

`spawn` (each worker is a genuinely fresh interpreter, launched via fork+exec)
sidesteps both problems: no inherited GL state, and process startup naturally
staggers the workers' EGL init calls enough to avoid the race. This mirrors
seeker's AsyncVectorEnv (github.com/zheyu-zhuang/seeker,
seeker/env_runner/async_vector_env.py), which forces spawn for the same
reason:

    if os.getenv("MUJOCO_GL") != "osmesa":
        mp.set_start_method('spawn', force=True)

We use an explicit `ctx` passed into `Process(...)`/`Pipe()` instead of
seeker's `set_start_method(force=True)`, so we don't mutate process-global
multiprocessing state (other code, e.g. torch's dataloader workers, may rely
on the default context).
"""
import multiprocessing as mp
from typing import Callable, List, Optional, Union

import numpy as np

from libero.libero.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    DummyVectorEnv,
    SubprocEnvWorker,
    _worker,
)


class _CtxSubprocEnvWorker(SubprocEnvWorker):
    """SubprocEnvWorker whose child Process/Pipe come from an explicit mp context.

    Only __init__ differs from the parent: using `ctx` instead of the bare
    `multiprocessing.context.Process`/`Pipe`, which otherwise resolve to the
    process-global default start method (fork on Linux). share_memory is
    unused by our eval loop and unsupported here.
    """

    def __init__(self, env_fn: Callable, ctx: mp.context.BaseContext) -> None:
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = False
        self.buffer = None
        args = (self.parent_remote, self.child_remote, CloudpickleWrapper(env_fn), self.buffer)
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        # Skip SubprocEnvWorker.__init__ (it rebuilds the pipe via the
        # default-context Process); go straight to EnvWorker.__init__.
        super(SubprocEnvWorker, self).__init__(env_fn)


class _CtxSubprocVectorEnv(BaseVectorEnv):
    """SubprocVectorEnv variant that spawns workers from a given mp context."""

    def __init__(self, env_fns: List[Callable], start_method: str, **kwargs) -> None:
        ctx = mp.get_context(start_method)
        worker_fn = lambda fn: _CtxSubprocEnvWorker(fn, ctx)
        super().__init__(env_fns, worker_fn, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs,
    ) -> np.ndarray:
        ids = self._wrap_id(id)
        obs_list = [self.workers[i].set_init_state(init_state[j]) for j, i in enumerate(ids)]
        return np.stack(obs_list)


def make_libero_venv(env_fns: List[Callable], start_method: str = "spawn"):
    """Build a LIBERO vector env over `env_fns` using the given start method.

    start_method:
      "dummy" -> DummyVectorEnv (sequential in-process; the old fallback).
      "spawn" -> subprocess workers from freshly spawned interpreters
                 (recommended: EGL-safe and avoids the concurrent-context-
                 creation race a fork-based start method hits; see module
                 docstring).
    """
    if start_method == "dummy":
        return DummyVectorEnv(env_fns)
    if start_method != "spawn":
        raise ValueError(f"Unknown env_start_method: {start_method!r}")
    return _CtxSubprocVectorEnv(env_fns, start_method=start_method)
