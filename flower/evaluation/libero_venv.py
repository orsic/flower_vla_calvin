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

Workers are also rebuildable in place (`.rebuild()` below): a batch that only
changes which bddl each slot loads (LIBERO-Plus's per-batch task mix) does not
need a fresh process/EGL context, so the eval loop reuses one venv across
batches instead of paying spawn + EGL init on every one -- see the plan /
commit this lands with for the measurement that motivated it.
"""
import gc
import multiprocessing as mp
from multiprocessing import connection
from typing import Callable, List, Optional, Union

import numpy as np

from libero.libero.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    DummyEnvWorker,
    DummyVectorEnv,
    SubprocEnvWorker,
)


def _rebuildable_worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    """Subprocess command loop for `_CtxSubprocEnvWorker`.

    A trimmed copy of `libero.libero.envs.venv._worker`: drops the `obs_bufs`
    share-memory path (`_CtxSubprocEnvWorker` always passes `share_memory=False`,
    dead here) and adds a `rebuild` command that swaps the worker's env in
    place instead of exiting the process.
    """
    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                p.send(env.step(data))
            elif cmd == "reset":
                # Upstream `_worker` branches here on whether reset() returned
                # (obs, info) vs. bare obs, only to re-pack the same value
                # (that branching matters solely for the obs_bufs share-memory
                # path we dropped above) -- so it collapses to a plain send.
                p.send(env.reset(**data))
            elif cmd == "rebuild":
                env.close()
                del env
                gc.collect()
                env = data.data()
                # A freshly spawned worker's numpy RNG is entropy-seeded (new
                # interpreter); reseed the same way so a slot with no explicit
                # init state (flower_eval_libero.py's initial_states is None
                # case) keeps drawing from a fresh stream instead of
                # continuing the outgoing env's stream. This is only the
                # steady state until flower_eval_libero.seed_and_reset() reseeds
                # the worker per episode (right before its next reset()) --
                # it exists so a rebuild without an explicit seed still starts
                # from a non-degenerate stream rather than the outgoing env's.
                np.random.seed(None)
                p.send(True)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                p.send(env.set_init_state(data))
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class _CtxSubprocEnvWorker(SubprocEnvWorker):
    """SubprocEnvWorker whose child Process/Pipe come from an explicit mp context.

    Uses `ctx` instead of the bare `multiprocessing.context.Process`/`Pipe`,
    which otherwise resolve to the process-global default start method (fork
    on Linux). share_memory is unused by our eval loop and unsupported here.
    Runs `_rebuildable_worker` (not upstream's `_worker`) so `.rebuild()` can
    swap the env without tearing down the process.
    """

    def __init__(self, env_fn: Callable, ctx: mp.context.BaseContext) -> None:
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = False
        self.buffer = None
        args = (self.parent_remote, self.child_remote, CloudpickleWrapper(env_fn))
        self.process = ctx.Process(target=_rebuildable_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        # Skip SubprocEnvWorker.__init__ (it rebuilds the pipe via the
        # default-context Process); go straight to EnvWorker.__init__.
        super(SubprocEnvWorker, self).__init__(env_fn)

    def send_rebuild(self, env_fn: Callable) -> None:
        self.parent_remote.send(["rebuild", CloudpickleWrapper(env_fn)])

    def recv_rebuild(self) -> None:
        self.parent_remote.recv()


class _RebuildableDummyEnvWorker(DummyEnvWorker):
    """DummyEnvWorker whose env can be swapped in place (see `.rebuild()`)."""

    def send_rebuild(self, env_fn: Callable) -> None:
        self.env.close()
        self.env = env_fn()

    def recv_rebuild(self) -> None:
        pass


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

    def rebuild(self, env_fns: List[Callable]) -> None:
        """Swap the envs of workers `0..len(env_fns)-1` in place.

        Sent to all targeted workers before waiting on any reply, so the
        rebuilds (process-local: env.close() + reconstruct) run concurrently
        instead of serially -- the same send-all/recv-all shape `step`/`reset`
        use. `env_fns` may be shorter than the pool (e.g. a ragged last
        batch); the surplus workers keep their current env untouched.
        """
        n = len(env_fns)
        for worker, fn in zip(self.workers[:n], env_fns):
            worker.send_rebuild(fn)
        for worker in self.workers[:n]:
            worker.recv_rebuild()


class _RebuildableDummyVectorEnv(DummyVectorEnv):
    """DummyVectorEnv variant whose workers can rebuild their env in place."""

    def __init__(self, env_fns: List[Callable], **kwargs) -> None:
        BaseVectorEnv.__init__(self, env_fns, _RebuildableDummyEnvWorker, **kwargs)

    def rebuild(self, env_fns: List[Callable]) -> None:
        n = len(env_fns)
        for worker, fn in zip(self.workers[:n], env_fns):
            worker.send_rebuild(fn)
        for worker in self.workers[:n]:
            worker.recv_rebuild()


def make_libero_venv(env_fns: List[Callable], start_method: str = "spawn"):
    """Build a LIBERO vector env over `env_fns` using the given start method.

    start_method:
      "dummy" -> DummyVectorEnv (sequential in-process; the old fallback).
      "spawn" -> subprocess workers from freshly spawned interpreters
                 (recommended: EGL-safe and avoids the concurrent-context-
                 creation race a fork-based start method hits; see module
                 docstring).

    The returned env supports `.rebuild(new_env_fns)` to swap in a new set of
    envs (e.g. a new batch's tasks) without tearing down its workers.
    """
    if start_method == "dummy":
        return _RebuildableDummyVectorEnv(env_fns)
    if start_method != "spawn":
        raise ValueError(f"Unknown env_start_method: {start_method!r}")
    return _CtxSubprocVectorEnv(env_fns, start_method=start_method)
