#!/usr/bin/env python
"""Generate a Rerun blueprint (.rbl) for eval rollout recordings.

One row per modality combo found in the recordings, each row being

    [ 3rd person image | 1st person image | action chart over proprio chart ]

and a thin summary strip on top saying which rollout the cursor is in. Because
flower/evaluation/rerun_recorder.py puts the episode on the `frame` timeline rather than
in the entity path, this one set of views covers every recorded rollout: drag the time
slider to scroll through them.

Usage:
    python scripts/rerun_blueprint.py OUT.rbl RECORDING.rrd [RECORDING.rrd ...]
    python scripts/rerun_blueprint.py OUT.rbl --combo static+wrist+lang+proprio ...

    rerun OUT.rbl RECORDING.rrd [RECORDING.rrd ...]

Recordings sharing a recording id merge into one recording in the viewer, so passing a
full-modality arm and a withheld-modality arm together stacks them as two rows under a
single time cursor.
"""

import argparse
import sys
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
from flower.evaluation.rerun_recorder import (  # noqa: E402
    APPLICATION_ID,
    combo_entity_path,
)


def combos_in(recordings):
    """Modality combos present in the given .rrd files, all-modalities first.

    Read back from the recordings rather than taken on faith so the blueprint can never
    name a row that has no data behind it. Entity paths come back escaped; the first
    path part IS the combo root, so it is used as-is.
    """
    found = []
    for path in recordings:
        rec = rr.dataframe.load_recording(str(path))
        for col in rec.schema().component_columns():
            parts = col.entity_path.strip("/").split("/")
            if len(parts) < 2 or parts[0].startswith("__"):
                continue
            root = "/" + parts[0]
            if root not in found:
                found.append(root)
    # Longest combo (most modalities) first, so the full-modality arm is the top row.
    return sorted(found, key=lambda r: (-r.count("+"), r))


def arm_row(root):
    """The requested layout for one modality combo: two images, then stacked charts."""
    label = root.lstrip("/").replace("\\+", "+")
    return rrb.Horizontal(
        rrb.Spatial2DView(origin=f"{root}/input/rgb_static", name=f"3rd person · {label}"),
        rrb.Spatial2DView(origin=f"{root}/input/rgb_gripper", name=f"1st person · {label}"),
        rrb.Vertical(
            rrb.TimeSeriesView(origin=f"{root}/output/action", name=f"action · {label}"),
            rrb.TimeSeriesView(origin=f"{root}/input/proprio", name=f"proprio · {label}"),
        ),
        column_shares=[1, 1, 2],
    )


def build(roots):
    rows = [arm_row(root) for root in roots]
    summaries = rrb.Horizontal(
        *[
            rrb.TextDocumentView(origin=f"{root}/summary", name="rollout")
            for root in roots
        ]
    )
    # Thin strip on top so the current rollout's task/instruction/outcome is always
    # visible while scrubbing; drop this line and the first row_share to remove it.
    return rrb.Blueprint(
        rrb.Vertical(summaries, *rows, row_shares=[0.5] + [2] * len(rows)),
        collapse_panels=False,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="path to write the .rbl blueprint to")
    ap.add_argument("recordings", nargs="*", help=".rrd files to read the combos from")
    ap.add_argument("--combo", action="append", default=[],
                    help="name a combo explicitly instead of reading the recordings "
                         "(repeatable, e.g. --combo static+wrist+lang+proprio)")
    args = ap.parse_args(argv)

    if args.combo:
        roots = [combo_entity_path(c) for c in args.combo]
    elif args.recordings:
        roots = combos_in(args.recordings)
    else:
        ap.error("pass at least one .rrd recording, or one --combo")
    if not roots:
        ap.error(f"no modality combos found in {args.recordings}")

    build(roots).save(APPLICATION_ID, args.out)
    print(f"wrote {args.out} with {len(roots)} row(s):")
    for root in roots:
        print("   ", root.lstrip("/").replace("\\+", "+"))
    print(f"\n  rerun {args.out} " + " ".join(str(r) for r in args.recordings))


if __name__ == "__main__":
    main()
