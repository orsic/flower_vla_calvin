#!/usr/bin/env python3
"""Model-independent severity for LIBERO-Plus perturbations.

task_classification.json's difficulty_level (1-5) is a coarse upstream annotation, not
a physical measurement -- it disagrees with the true magnitude e.g. 67% of the time for
Robot Initial States (0.1 rad band vs. difficulty_level: only 129/393 agree), and is
outright non-monotone for Objects Layout because that category pools two physically
different sub-mechanisms (distractor count and target displacement) into one label.

Every perturbation family instead encodes its exact magnitude in the task/BDDL name
itself -- the same substrings LIBERO-Plus's own env_wrapper.py and new_init.py parse at
rollout time to configure the perturbation -- so severity is recoverable purely by
parsing columns already in result.csv (task_name, bddl_file, problem_folder), with no
re-evaluation needed. Two categories have no intrinsic magnitude and are reported as
categorical sub-types instead: Language Instructions (the rewrite index is an unordered
LLM-rewrite id) and Background Textures (a texture identity, not a scalar).

Every parser below returns None when its category's expected substring is absent, so
calling the wrong parser on a row (or a row from an unrelated category) fails safe
rather than returning a bogus number.
"""
import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Tuple

# --- Camera Viewpoints -------------------------------------------------------
# task_name ends "..._view_{h}_{v}_{s}_{er}_{ev}_initstate_{N}"; h/v/er/ev are degrees
# wrapped into [0, 360) (e.g. 358 encodes -2 degrees), s/100 is a distance scale.
# Robot Initial States and Sensor Noise task names carry this same nominal
# "_view_0_0_100_0_0_..." prefix (their own perturbation is layered on top of it), so
# this regex matches all three categories -- callers dispatch by task_category first.
_VIEW_RE = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)")
_INITSTATE_RE = re.compile(r"_initstate_(\d+)")
_NOISE_RE = re.compile(r"_noise_(\d+)$")
_ADD_RE = re.compile(r"_add_(\d+)$")
_LEVEL_RE = re.compile(r"_level(\d+)_sample(\d+)$")
_BACKGROUND_RE = re.compile(r"_(table|tb)_(\d+)$")
_LIGHT_PROBLEM_RE = re.compile(r"([a-z]+)_light_sync_modified_(\d+)$")

# Nominal camera pose is per-scene (libero_*_manipulation.py's pos_av/quat_av), but the
# *relative* rotation applied by _setup_camera doesn't depend on it -- see
# camera_severity's docstring -- so no per-scene table is needed here.


def _signed_deg(d: int) -> int:
    """Wrap a 0-360 filename value to the signed [-180, 180) rotation it encodes."""
    return ((d + 180) % 360) - 180


def camera_severity(task_name: str) -> Optional[Tuple[float, float]]:
    """(rotation_deg, distance_scale) for a Camera Viewpoints task_name, else None.

    _setup_camera (libero_kitchen_tabletop_manipulation.py:318-345, identical in each
    of the 6 problems/*.py files) applies, to the nominal camera quaternion, in world
    frame: Ry(-vertical) first, then Rz(horizon), then Rz(end_point_rot), then
    Ry(-end_point_vertical). The resulting *relative* rotation
    Ry(-ev).Rz(er).Rz(h).Ry(-v) is independent of the nominal pose (any per-scene
    Rz(h)/Ry(v) constants cancel out of R_final . R_nominal^-1), so this is safe to
    compute without knowing which scene the task belongs to. distance_scale is a pure
    post-rotation translation (scale_distance_from_pivot), reported separately since it
    isn't commensurate with a rotation angle.
    """
    m = _VIEW_RE.search(task_name)
    if m is None:
        return None
    h, v, s, er, ev, _initstate = (int(x) for x in m.groups())
    h, v, er, ev = (_signed_deg(x) for x in (h, v, er, ev))

    from scipy.spatial.transform import Rotation

    delta = (
        Rotation.from_euler("y", -ev, degrees=True)
        * Rotation.from_euler("z", er, degrees=True)
        * Rotation.from_euler("z", h, degrees=True)
        * Rotation.from_euler("y", -v, degrees=True)
    )
    rotation_deg = math.degrees(delta.magnitude())
    return rotation_deg, s / 100.0


# --- Robot Initial States ----------------------------------------------------


def robot_initial_state_severity(task_name: str) -> Optional[float]:
    """||delta qpos|| in radians for a Robot Initial States task_name, else None.

    new_init.py generates MountedPanda{N}/OnTheGroundPanda{N} for N in 1..500, in five
    bands of 100, each offsetting the nominal home qpos by a unit random vector scaled
    by {0.1, 0.2, 0.3, 0.4, 0.5} rad (new_init.py:10-160: `perturbed_qpos =
    original_qpos + perturbation * {0.1..0.5}`, perturbation already unit-normalized).
    ControlEnv swaps in robot "N" whenever int(init_state) != 0
    (env_wrapper.py:216-217), and N is exactly the filename's _initstate_ index -- 0
    (the nominal robot, no swap) maps to 0.0 rad.
    """
    m = _INITSTATE_RE.search(task_name)
    if m is None:
        return None
    n = int(m.group(1))
    return 0.0 if n == 0 else 0.1 * math.ceil(n / 100)


# --- Sensor Noise -------------------------------------------------------------

# (corruption_fn_name, band_offset); env_wrapper.py:288-308 dispatches noise index M
# (1-50) to one of 5 corruptions at severity = M - offset (always 1-10).
_NOISE_BANDS = [
    (10, "motion_blur"),
    (20, "gaussian_blur"),
    (30, "zoom_blur"),
    (40, "fog"),
    (50, "glass_blur"),
]


def sensor_noise_severity(task_name: str) -> Optional[Tuple[str, int]]:
    """(corruption_type, severity 1-10) for a Sensor Noise task_name, else None."""
    m = _NOISE_RE.search(task_name)
    if m is None:
        return None
    idx = int(m.group(1))
    for band_end, name in _NOISE_BANDS:
        if idx <= band_end:
            return name, idx - (band_end - 10)
    raise ValueError(f"noise index {idx} out of the expected 1-50 range")


# --- Objects Layout -----------------------------------------------------------


def objects_layout_severity(task_name: str) -> Optional[Tuple[str, int]]:
    """('distractors', n) or ('displacement', level) for an Objects Layout
    task_name, else None -- the category has two physically distinct sub-mechanisms,
    each with its own severity, which is why difficulty_level (a single pooled label)
    is non-monotone for this category.

    n_distractors = ceil(A/6): verified by counting `add_object_region` blocks in the
    BDDL across the _add_1.._add_30 range (add_1..6 -> 1 distractor's worth of regions
    ... add_25..30 -> 5). 'displacement' is the filename's levelK index directly
    (target region center moves ~0.01 m further from the base task per level, measured
    from the perturbed BDDL's (:ranges ...) vs. the base task's).
    """
    m = _ADD_RE.search(task_name)
    if m is not None:
        return "distractors", math.ceil(int(m.group(1)) / 6)
    m = _LEVEL_RE.search(task_name)
    if m is not None:
        return "displacement", int(m.group(1))
    return None


# --- Background Textures (no magnitude -- categorical sub-type only) ---------


def background_texture_subtype(task_name: str) -> Optional[str]:
    """'table' (1 surface retextured) or 'tb' (2 surfaces) for a Background Textures
    task_name, else None. T itself just indexes an alphabetically-sorted texture
    asset list -- a texture identity, not a scalar -- so there is no severity axis
    here, only this structural sub-type."""
    m = _BACKGROUND_RE.search(task_name)
    return m.group(1) if m is not None else None


# --- Light Conditions ----------------------------------------------------------

# BDDL problem name -> base (unperturbed) scene style file holding the nominal light
# values light_severity diffs against. Covers every scene prefix LIBERO-Plus's light
# generator produced (verified against libero_10/spatial/object/goal's Light
# Conditions BDDLs: {floor, kitchen, liv, study, tabletop}, all present).
_LIGHT_BASE_STYLE = {
    "kitchen": "libero_kitchen_tabletop_base_style.xml",
    "floor": "libero_floor_base_style.xml",
    "liv": "libero_living_room_tabletop_base_style.xml",
    "study": "libero_study_base_style.xml",
    "tabletop": "libero_tabletop_base_style.xml",
    "coffee": "libero_coffee_table_base_style.xml",
}


def _parse_lights(xml_path: str) -> Dict[str, Dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    lights = {}
    for el in root.iter("light"):
        name = el.get("name")
        if name is None:
            continue
        lights[name] = {
            "diffuse": tuple(float(x) for x in el.get("diffuse", "0 0 0").split()),
            "dir": tuple(float(x) for x in el.get("dir", "0 0 -1").split()),
            "specular": tuple(float(x) for x in el.get("specular", "0 0 0").split()),
            "castshadow": el.get("castshadow", "false").lower() == "true",
        }
    return lights


def _euclid(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _angle_deg(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    cos_t = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))
    return math.degrees(math.acos(cos_t))


def light_severity(bddl_path: str, libero_plus_root: str) -> Optional[Dict[str, Any]]:
    """Composite photometric distance from nominal for a Light Conditions task, else
    None if the BDDL's problem name doesn't reference a light-perturbed scene.

    The BDDL's (problem ...) name embeds the exact perturbed-scene index, e.g.
    "..._kitchen_light_sync_modified_1017" -> assets/scenes/lights/
    kitchen_light_sync_modified_1017.xml (verified: this direct construction resolves
    for every Light Conditions task across all 4 LIBERO-Plus suites, 0 misses -- no
    generation-log lookup needed). Only the light *fixture names shared* between the
    perturbed scene and its base style are diffed (kitchen scenes perturb
    light_kitchen/light1/light2; every other scene perturbs light1/light2 only), on:
    diffuse and specular (RGB triples, Euclidean distance), dir (angle between
    vectors), and castshadow (flipped or not). The composite `severity` is the mean of
    the four components each normalized to roughly [0, 1] (diffuse/specular by
    sqrt(3), the max possible RGB distance in a [0,1]^3 cube; dir by 180 degrees;
    shadow flip is already 0/1).
    """
    from libero.libero.envs.bddl_utils import get_problem_info

    problem_name = get_problem_info(bddl_path)["problem_name"]
    m = _LIGHT_PROBLEM_RE.search(problem_name)
    if m is None:
        return None
    prefix, idx = m.groups()

    base_style = _LIGHT_BASE_STYLE.get(prefix)
    if base_style is None:
        raise ValueError(f"no base style scene registered for light prefix '{prefix}'")

    assets_dir = os.path.join(libero_plus_root, "libero", "libero", "assets")
    variant_xml = os.path.join(assets_dir, "scenes", "lights", f"{prefix}_light_sync_modified_{idx}.xml")
    base_xml = os.path.join(assets_dir, "scenes", base_style)

    variant_lights = _parse_lights(variant_xml)
    nominal_lights = _parse_lights(base_xml)
    shared = sorted(set(variant_lights) & set(nominal_lights))
    if not shared:
        raise ValueError(f"no shared <light> fixture names between {variant_xml} and {base_xml}")

    diffuse_delta = max(_euclid(variant_lights[n]["diffuse"], nominal_lights[n]["diffuse"]) for n in shared)
    dir_delta_deg = max(_angle_deg(variant_lights[n]["dir"], nominal_lights[n]["dir"]) for n in shared)
    specular_delta = max(_euclid(variant_lights[n]["specular"], nominal_lights[n]["specular"]) for n in shared)
    shadow_flipped = any(variant_lights[n]["castshadow"] != nominal_lights[n]["castshadow"] for n in shared)

    severity = (
        diffuse_delta / math.sqrt(3)
        + dir_delta_deg / 180.0
        + specular_delta / math.sqrt(3)
        + float(shadow_flipped)
    ) / 4.0

    return {
        "diffuse_delta": diffuse_delta,
        "dir_delta_deg": dir_delta_deg,
        "specular_delta": specular_delta,
        "shadow_flipped": shadow_flipped,
        "severity": severity,
    }
