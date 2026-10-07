"""Rerun (.rrd) recording of LIBERO evaluation rollouts.

Opt-in (``rerun.enabled``, off by default in every eval config), this writes one Rerun
recording per eval invocation covering the first ``rerun.max_episodes`` episodes::

    /<modality combo>/
        input/rgb_static     3rd-person camera, JPEG
        input/rgb_gripper    wrist camera, JPEG
        input/proprio        9-D joint positions + gripper state, named series
        input/present        0/1 per modality -- what actually reached the model
        output/action        the 7-D action the model emitted, named series
        output/success       the outcome, one point per episode
        info                 task name, instruction, modality combo, rollout_seed,
                             steps_taken, success -- as structured fields
        summary              the same, as readable text for a viewer panel

**The episode is a position on the timeline, not a level in the entity path.** Every
episode logs to the same entity paths, on one ``frame`` timeline where episode *i*
occupies the fixed slot ``[i * max_steps, (i + 1) * max_steps)``. One time slider
therefore scrolls continuously through every recorded rollout in a single set of views
(see scripts/rerun_blueprint.py). Rerun resolves data as "latest value at time T on the
active timeline", so an episode can live either in the path or on the timeline, not
both -- putting it on the timeline is what makes a single scrubbing view possible.

Fixed-width slots (rather than packing episodes end to end) keep two recordings
frame-aligned: a full-modality arm and a withheld-modality arm run the same episodes in
the same order, so frame F lands on the same episode of both even though their episodes
end at different steps. Their combos differ, so they occupy sibling branches and can be
viewed stacked, scrubbed by one cursor.

Both camera streams and proprio are read straight from the environment observation, so
they are recorded whether or not the model consumed them -- a modality withheld by
``eval_modalities.*=False`` still shows its frames, and the withholding is recorded
separately as ``input/present``. That keeps the withheld arm of a comparison watchable
instead of blank.

The ``input/present`` flags are derived from ``EvaluateLibero.eval_modalities`` and its
``uses_proprio``, i.e. exactly the source of ``result.csv``'s ``use_*`` columns, so a
recording and the CSV never disagree. (On a fixed-ablation checkpoint -- one trained with
``model.modalities.*`` -- the model's own ``modality_mask`` can withhold more than
``eval_modalities`` says; that pre-existing discrepancy is inherited from the CSV columns
rather than re-derived here.)

**Camera frames are vertically flipped before logging, for display only.** robosuite
renders bottom-up, so the raw ``agentview_image``/``robot0_eye_in_hand_image`` arrays --
the ones the model is actually fed, and the ones the training data was rendered in --
are upside down to a human. Flipping makes the viewer legible at the cost of the
recorded image no longer being byte-identical to the model input: it is that input
mirrored vertically. Nothing on the inference path sees the flip.

Recordings are written through an explicit ``rr.RecordingStream`` rather than ``rr.init()``
globals: eval spawns worker processes and MuJoCo subprocesses, and a global recording
stream there is a trap. ``rerun`` itself is imported lazily, only when recording is
enabled, so an image built before rerun-sdk was added still runs evals unchanged.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Rerun application id. The recording id (see RerunRecorder) is what groups recordings:
# two .rrd files sharing one recording id merge into a single recording when opened
# together, which is how an all-modalities arm and a withheld-modality arm end up as
# sibling branches under one time cursor.
APPLICATION_ID = "flower_eval"

# The single timeline. Episode i occupies [i * frame_stride, (i+1) * frame_stride).
TIMELINE = "frame"

# Short modality names, the same ones scripts/compare_eval_csvs.py (MODALITY_COLUMNS) and
# scripts/pid_modality.py (MODALITY_ORDER) use, so a combo branch reads the same way as a
# --modalities-a spec.
MODALITY_ORDER: Tuple[str, ...] = ("static", "wrist", "lang", "proprio")

ACTION_NAMES: Tuple[str, ...] = ("x", "y", "z", "rx", "ry", "rz", "gripper")
PROPRIO_NAMES: Tuple[str, ...] = tuple(f"joint_{i}" for i in range(7)) + (
    "gripper_0",
    "gripper_1",
)

# Env observation key -> entity name, mirroring obs_translation.translate_obs_space's
# agentview_image -> rgb_static / robot0_eye_in_hand_image -> rgb_gripper mapping.
_CAMERAS: Tuple[Tuple[str, str], ...] = (
    ("agentview_image", "rgb_static"),
    ("robot0_eye_in_hand_image", "rgb_gripper"),
)


def modality_flags(
    eval_modalities: Optional[Dict[str, bool]], uses_proprio: bool
) -> Dict[str, bool]:
    """Present-modality flags keyed by short name, from EvaluateLibero's own fields.

    ``uses_proprio`` (not ``eval_modalities["proprio"]``) is what reached the model: a
    checkpoint trained with ``use_proprio=False`` never receives proprioception however
    ``eval_modalities`` is set.
    """
    eval_modalities = eval_modalities or {}
    return {
        "static": bool(eval_modalities.get("rgb_static", True)),
        "wrist": bool(eval_modalities.get("rgb_gripper", True)),
        "lang": bool(eval_modalities.get("language", True)),
        "proprio": bool(uses_proprio),
    }


def modality_label(flags: Dict[str, bool]) -> str:
    """'+'-joined present modalities, e.g. 'static+lang+proprio'. Matches run.sh's
    modality_label() and scripts/compare_eval_csvs.py's --modalities spelling."""
    present = [name for name in MODALITY_ORDER if flags.get(name)]
    return "+".join(present) if present else "none"


def combo_entity_path(combo: str) -> str:
    """Root entity path for one modality combo: ``/<combo>``.

    Escaped per Rerun's entity-path grammar, so the '+' in a combo name survives as
    ``\\+`` on the wire (the viewer renders it back as '+'). That escape is part of the
    path a blueprint origin or a dataframe query has to spell, which is why this is
    public rather than inlined -- see scripts/rerun_blueprint.py.
    """
    import rerun as rr

    return "/" + rr.escape_entity_path_part(str(combo))


def episode_frame(episode_slot: int, frame_stride: int, step: int = 0) -> int:
    """Position of a within-episode step on the global timeline."""
    return int(episode_slot) * int(frame_stride) + int(step)


def _proprio_vector(obs: Any) -> Optional[np.ndarray]:
    """joint positions + gripper state, or None when the env didn't render them.

    Same concatenation as obs_translation.translate_obs_space's robot_obs.
    """
    if not hasattr(obs, "get"):
        return None
    joints = obs.get("robot0_joint_pos")
    gripper = obs.get("robot0_gripper_qpos")
    if joints is None or gripper is None:
        return None
    return np.concatenate(
        [
            np.asarray(joints, dtype=float).ravel(),
            np.asarray(gripper, dtype=float).ravel(),
        ]
    )


def _for_display(frame: Any) -> np.ndarray:
    """Vertically flip a robosuite camera frame so it is the right way up in the viewer.

    Display only -- see the module docstring. The model is fed the unflipped array.
    """
    return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[::-1])


class RerunRecorder:
    """Records rollout episodes to a Rerun .rrd file. Disabled by default and a cheap
    no-op in that state -- every method returns immediately and ``rerun`` is never
    imported.

    Call order, once per batch of episodes:
        begin_batch(keys, languages)               # after the env warmup steps
        log_step(step, obs, actions, active_ids)   # before each env.step
        end_batch(dones, steps_taken, seeds)       # after the rollout loop
    and ``close()`` once when the evaluation finishes.

    Episodes are capped at ``max_episodes`` counted in work order, consistently across
    both batching modes, so two evals of the same suite that differ only in
    ``eval_modalities`` record the identical episode set, in the same order, into the
    same frame slots. (This deliberately differs from ``num_videos``, which caps per task
    in ``evaluate_task`` but globally in ``evaluate_work_list``.)
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]],
        log_dir: Optional[str],
        eval_modalities: Optional[Dict[str, bool]],
        uses_proprio: bool,
        libero_variant: str = "orig",
        benchmark_name: str = "",
        max_steps: int = 0,
    ) -> None:
        cfg = dict(cfg or {})
        self.max_episodes: int = int(cfg.get("max_episodes") or 0)
        self.jpeg_quality: int = int(cfg.get("jpeg_quality") or 40)
        self.flags: Dict[str, bool] = modality_flags(eval_modalities, uses_proprio)
        self.label: str = modality_label(self.flags)
        self.combo: str = str(cfg.get("combo_name") or self.label)
        # Every episode gets the same slot width, so two arms whose episodes end at
        # different steps still line up episode-for-episode on the frame timeline.
        self.frame_stride: int = int(cfg.get("frame_stride") or max_steps or 1) + 1
        self.path: Optional[str] = None

        self._rr: Any = None
        self._stream: Any = None
        self._root: str = ""
        self._n_started: int = 0
        # Per batch slot: the episode's frame slot, or None when past the episode cap.
        self._slots: List[Optional[int]] = []
        self._keys: List[Tuple[str, int]] = []
        self._languages: List[str] = []
        self._annotated: set = set()

        if not cfg.get("enabled", False) or self.max_episodes <= 0:
            return

        import rerun as rr  # lazy: only a recording run needs rerun-sdk installed

        self._rr = rr
        self._root = combo_entity_path(self.combo)
        path = cfg.get("path") or os.path.join(log_dir or ".", "rollouts.rrd")
        path = os.path.abspath(str(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        recording_id = cfg.get("recording_id") or f"{libero_variant}_{benchmark_name}"
        self._stream = rr.RecordingStream(
            APPLICATION_ID, recording_id=str(recording_id)
        )
        self._stream.save(path)
        self.path = path

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    @property
    def root(self) -> str:
        """Entity path every episode of this recorder logs under."""
        return self._root or combo_entity_path(self.combo)

    def episode_frame(self, episode_slot: int, step: int = 0) -> int:
        """Global-timeline frame for a step of the episode in the given slot."""
        return episode_frame(episode_slot, self.frame_stride, step)

    def begin_batch(
        self,
        keys: Sequence[Tuple[str, int]],
        languages: Union[str, Sequence[str]],
    ) -> None:
        """Claim a frame slot for each batch slot, up to the episode cap.

        ``keys`` is one ``(task_name, episode_idx)`` per slot; ``languages`` is either one
        instruction broadcast across the batch (per-task batching) or one per slot
        (cross-task batching) -- the same dispatch process_env_obs_batch does.

        The per-episode ``info``/``summary`` records are written by end_batch instead,
        back at the episode's own first frame, because they carry the outcome. Rerun
        accepts out-of-order time logging, and latest-at then resolves them correctly
        from anywhere inside the episode.
        """
        if self._stream is None:
            return
        self._slots = []
        self._keys = list(keys)
        self._languages = []
        for k in range(len(keys)):
            self._languages.append(
                languages
                if isinstance(languages, str)
                else (languages[k] if k < len(languages) else "")
            )
            if self._n_started >= self.max_episodes:
                self._slots.append(None)
                continue
            self._slots.append(self._n_started)
            self._n_started += 1

    def _annotate(self, entity: str, names: Sequence[str]) -> None:
        """Name a multi-series entity's lines, once, on its first sample."""
        if entity in self._annotated:
            return
        self._annotated.add(entity)
        self._stream.log(entity, self._rr.SeriesLines(names=list(names)), static=True)

    def log_step(
        self,
        step: int,
        obs: Sequence[Any],
        actions: Any,
        active_ids: Sequence[int],
    ) -> None:
        """Log one rollout step for every still-running recorded slot.

        Called before ``env.step``, so the observation logged at step *t* is the one the
        step-*t* action was produced from. Slots that have already finished are absent
        from ``active_ids`` and stop producing rows.
        """
        if self._stream is None:
            return
        recorded = [
            k for k in active_ids if k < len(self._slots) and self._slots[k] is not None
        ]
        if not recorded:
            return

        rr = self._rr
        root = self._root
        present = [float(self.flags[name]) for name in MODALITY_ORDER]

        for k in recorded:
            # Per slot, not once per call: concurrently-rolling episodes sit in
            # different slots and therefore at different frames for the same step.
            self._stream.set_time(
                TIMELINE, sequence=self.episode_frame(self._slots[k], step)
            )
            slot_obs = obs[k]
            for obs_key, entity in _CAMERAS:
                frame = slot_obs.get(obs_key) if hasattr(slot_obs, "get") else None
                if frame is None:
                    continue
                self._stream.log(
                    f"{root}/input/{entity}",
                    rr.Image(_for_display(frame), color_model="RGB").compress(
                        jpeg_quality=self.jpeg_quality
                    ),
                )
            proprio = _proprio_vector(slot_obs)
            if proprio is not None:
                self._annotate(f"{root}/input/proprio", PROPRIO_NAMES)
                self._stream.log(f"{root}/input/proprio", rr.Scalars(proprio.tolist()))
            self._annotate(f"{root}/input/present", MODALITY_ORDER)
            self._stream.log(f"{root}/input/present", rr.Scalars(present))
            self._annotate(f"{root}/output/action", ACTION_NAMES)
            self._stream.log(
                f"{root}/output/action",
                rr.Scalars(np.asarray(actions[k], dtype=float).ravel().tolist()),
            )

    def end_batch(
        self,
        dones: Sequence[bool],
        steps_taken: Sequence[int],
        seeds: Optional[Sequence[int]] = None,
    ) -> None:
        """Record each recorded episode's outcome and identity, then release the batch."""
        if self._stream is None:
            return
        rr = self._rr
        root = self._root
        for k, slot in enumerate(self._slots):
            if slot is None:
                continue
            success = int(bool(dones[k]))
            steps = int(steps_taken[k])
            seed = int(seeds[k]) if seeds is not None else -1
            task_name, episode_idx = self._keys[k]
            language = self._languages[k]

            # Identity + outcome at the episode's FIRST frame, so latest-at resolves it
            # for every frame inside the episode -- scrubbing always tells you which
            # rollout you are looking at and how it ended.
            self._stream.set_time(TIMELINE, sequence=self.episode_frame(slot))
            self._stream.log(
                f"{root}/info",
                rr.AnyValues(
                    task_name=str(task_name),
                    episode_idx=int(episode_idx),
                    episode_slot=int(slot),
                    language=str(language),
                    modalities=self.label,
                    success=success,
                    steps_taken=steps,
                    rollout_seed=seed,
                    **{
                        f"present_{name}": int(self.flags[name])
                        for name in MODALITY_ORDER
                    },
                ),
            )
            self._stream.log(
                f"{root}/summary",
                rr.TextDocument(
                    f"**episode {slot}** — {'SUCCESS' if success else 'FAILURE'}"
                    f" in {steps} steps\n\n"
                    f"- modalities: `{self.label}`\n"
                    f"- task: `{task_name}`\n"
                    f'- instruction: "{language}"\n'
                    f"- episode_idx: {episode_idx} · rollout_seed: {seed}\n",
                    media_type="text/markdown",
                ),
            )

            # One point per episode, at the frame it ended on.
            self._stream.set_time(TIMELINE, sequence=self.episode_frame(slot, steps))
            self._stream.log(f"{root}/output/success", rr.Scalars(float(success)))
        self._slots = []
        self._keys = []
        self._languages = []

    def close(self) -> None:
        """Flush the recording. Safe to call more than once."""
        if self._stream is None:
            return
        self._stream.flush()
        self._stream = None
        self._rr = None
