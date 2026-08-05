"""Adds the pitch itself — velocity, movement, spin, release point — to every swing model
as a control. Pitchers throw different stuff at 0-2 than at 2-0, so without this a swing's
reaction to nastier pitching gets scored as the hitter choosing to adjust.

It matters a lot: a hitter's raw situational movement repeats at r=0.616 with location
controls alone, and only 0.394 once the pitch is accounted for. Most of it was reaction.

Source: data/pitch_chars.parquet, written by src/extract_pitch_chars.R (sabRmetrics).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Parsimonious and non-redundant: vx0/vy0/vz0 and ax/ay/az are dropped because
# release_speed and pfx_x/pfx_z are functions of them. Squares on the three primary axes
# let the swing respond nonlinearly to velocity and break. Within a (batter, stand) unit
# the stance is fixed, so pfx_x needs no handedness mirror.
PITCH_CHARS = ["release_speed", "pfx_x", "pfx_z", "release_spin_rate", "spin_axis",
               "extension", "release_pos_x", "release_pos_z", "arm_angle"]
PITCH_SQUARES = ["release_speed", "pfx_x", "pfx_z"]
JOIN_KEYS = ["game_pk", "batter_id", "balls", "strikes", "px_key", "pz_key"]


def join_pitch_chars(swings, verbose=True):
    """Attach pitch characteristics; drop the ~3% of swings that do not match."""
    chars = pd.read_parquet(ROOT / "data" / "pitch_chars.parquet")
    swings = swings.copy()
    swings["px_key"] = swings["plate_x"].round(2)
    swings["pz_key"] = swings["plate_z"].round(2)
    for k in JOIN_KEYS:
        if chars[k].dtype != swings[k].dtype:
            chars[k] = chars[k].astype(swings[k].dtype)
    merged = swings.merge(chars[JOIN_KEYS + PITCH_CHARS], on=JOIN_KEYS, how="left")
    before = len(merged)
    merged = merged.dropna(subset=PITCH_CHARS).reset_index(drop=True)
    if verbose:
        print(f"  pitch chars joined: {len(merged):,}/{before:,} swings "
              f"({100 * len(merged) / before:.1f}%)", flush=True)
    return merged


def control_matrix(group):
    """Within-unit standardised characteristics plus squares on velocity and both breaks.

    Standardised so the block is conditioned comparably to the location surface and the
    dummy blocks; lstsq is otherwise sensitive to spin rate's ~2000-unit scale.
    """
    raw = group[PITCH_CHARS].to_numpy(float)
    sd = raw.std(axis=0)
    z = (raw - raw.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
    return np.column_stack([z, z[:, [PITCH_CHARS.index(c) for c in PITCH_SQUARES]] ** 2])
