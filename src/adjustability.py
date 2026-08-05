"""How much a hitter changes his swing for the situation, once you account for where the pitch
was. Location forces most of any swing, so this measures only what is left over that still
moves with the count, the baserunners and the pitch type.

Single-regression build. Per (batter, stand) unit (2024-25, >= MIN_SWINGS), for each of 3 volitional
trait dials (bat_speed, swing_length, swing_path_tilt) we fit ONE regression of the dial on

    dial ~ pitch-location surface  +  situation (count, game state, pitch type)

and read adjustability off as the INCREMENTAL variance the situation explains once location is already
in the model — adjusted R^2 of the full model minus the location-only model — averaged over the 3
dials and floored at 0. 0 = the situation adds nothing to a swing already pinned by where the pitch
is; higher = his swing systematically moves with the situation.

Why one joint regression per hitter (not a pooled league model): a pooled model shares one situation
coefficient across everyone, so it reflects a hitter's situation MIX, not his responsiveness. Per-hitter
responsiveness needs hitter x situation terms, which by Frisch-Waugh-Lovell is exactly this per-unit
joint fit. It replaces the earlier two-stage build (global location residualization, then a separate
per-hitter situation R^2): one model is easier to read and fits each hitter's OWN location relationship
rather than a single league-average surface.

The two attack angles are left out of the dials — they are the most location-forced. adjustability is
an UNSIGNED magnitude (a variance share, not a direction).

The headline is the whole situation over a location-only baseline. Each per-axis number is that axis's
UNIQUE contribution — the variance it adds on top of location AND the other two situation axes — so
adj_count holds pitch type fixed and is not two-strike pitch-mix relabeled:

  adjustability  = all of the situation (count + game state + pitch), over location-only  -- headline
  adj_count      = unique add of the count      (balls, strikes),         net of location + game state + pitch
  adj_gamestate  = unique add of the game state (base state, outs),         net of location + count + pitch
  adj_pitch      = unique add of the pitch type (FB / breaking / offspeed), net of location + count + game state

Input:  data/swings_model.parquet
Output: data/adjustability.parquet   (headline column: adjustability)

Run:  python src/adjustability.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KEY = ["batter_id", "batter_stand"]
SEASONS = [2024, 2025]
REF_PATH = ROOT / "src" / "adjustability_reference.json"
REF_SEASONS = [2024, 2025]
MIN_SWINGS = 400
DIALS = ["bat_speed", "swing_length", "swing_path_tilt"]
PITCH_GROUP = {
    "FF": "FB", "SI": "FB", "FC": "FB",
    "SL": "brk", "CU": "brk", "KC": "brk", "ST": "brk", "SV": "brk", "SC": "brk", "KN": "brk",
    "CH": "off", "FS": "off", "FO": "off",
}


def add_context(df):
    pull = np.where(df["batter_stand"].to_numpy() == "L", 1.0, -1.0)
    df["px"] = df["plate_x"] * pull
    df["pz"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    df = df.dropna(subset=["px", "pz"] + DIALS).copy()
    df["pitch_group"] = df["pitch_type"].map(PITCH_GROUP).fillna("other")
    on_first  = df["on_1b_id"].notna()
    on_second = df["on_2b_id"].notna()
    on_third  = df["on_3b_id"].notna()
    df["base_state"] = np.where(on_second | on_third, "risp", np.where(on_first, "on1", "empty"))
    return df


def location_design(group):
    """Pitch-location nuisance surface: intercept + linear + quadratic + interaction in (px, pz)."""
    px = group.px.to_numpy(float)
    pz = group.pz.to_numpy(float)
    return np.column_stack([np.ones(len(group)), px, pz, px ** 2, pz ** 2, px * pz])


def adjusted_r_squared(y, X):
    """Adjusted R^2 of y on design X (X INCLUDES its own intercept column). Not clipped here — the
    caller differences two adjusted_r_squared's and floors the increment."""
    n_obs, n_cols = X.shape
    n_predictors = n_cols - 1  # predictors excluding the intercept
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    sum_sq_error = float(((y - X @ coefficients) ** 2).sum())
    sum_sq_total = float(((y - y.mean()) ** 2).sum())
    if sum_sq_total <= 0 or n_obs - n_predictors - 1 <= 0:
        return 0.0
    # Penalty adjusts for the number of predictors so adding useless ones doesn't inflate R^2
    penalty = (n_obs - 1) / (n_obs - n_predictors - 1)
    return 1 - (sum_sq_error / sum_sq_total) * penalty


def situation_dummies(group, cols):
    return pd.get_dummies(group[cols].astype(str), drop_first=True).to_numpy(float)


def build_situation_design(location_matrix, group, situation_cols):
    """Stack location surface with situation dummy columns. Returns just location if cols is empty."""
    if not situation_cols:
        return location_matrix
    return np.column_stack([location_matrix, situation_dummies(group, situation_cols)])


def dial_r_squared_with_situation(location_matrix, group, dial_outcomes, situation_cols):
    """Per-dial adjusted R^2 of [location + these situation cols]."""
    X = build_situation_design(location_matrix, group, situation_cols)
    return {dial: adjusted_r_squared(dial_outcomes[dial], X) for dial in DIALS}


def mean_r_squared_gain(full_model_r_squared, baseline_r_squared):
    """Mean over dials of (full model - baseline model) adjusted R^2, each floored at 0."""
    gains = [max(0.0, full_model_r_squared[dial] - baseline_r_squared[dial]) for dial in DIALS]
    return float(np.mean(gains))


def resolve_reference(scores_df):
    """Return (mean, std, sorted_array) using the frozen 2024-25 baseline.

    Pegging the scale to a fixed 2024-25 baseline keeps adjustability_plus / adjustability_pctile
    comparable across seasons — same approach as repertoire_reference.json. Delete the file to re-peg.
    Holds only league-level aggregates (no PII) so it lives in-repo and is committed."""
    if REF_PATH.exists():
        ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
        baseline_mean   = ref["adjustability_mean"]
        baseline_std    = ref["adjustability_std"]
        baseline_sorted = np.array(ref["adjustability_sorted"], float)
        return baseline_mean, baseline_std, baseline_sorted
    baseline_mean   = float(scores_df["adjustability"].mean())
    baseline_std    = float(scores_df["adjustability"].std())
    baseline_sorted = [round(float(v), 4) for v in sorted(scores_df["adjustability"])]
    ref = {
        "reference_seasons":    REF_SEASONS,
        "n_reference_units":    len(scores_df),
        "adjustability_mean":   baseline_mean,
        "adjustability_std":    baseline_std,
        "adjustability_sorted": baseline_sorted,
    }
    REF_PATH.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(f"Built + froze adjustability reference -> {REF_PATH} (seasons {REF_SEASONS}, {len(scores_df)} units)")
    return baseline_mean, baseline_std, np.array(baseline_sorted, float)


def main():
    df = add_context(pd.read_parquet(
        DATA / "swings_model.parquet",
        columns=["game_year", "batter_full_name"] + KEY + ["balls", "strikes", "outs_when_up",
                 "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type", "pitcher_throws",
                 "on_1b_id", "on_2b_id", "on_3b_id"] + DIALS).query("game_year in @SEASONS"))

    axes = {
        "count":     ["balls", "strikes"],
        "gamestate": ["base_state", "outs_when_up"],
        "pitch":     ["pitch_group"],
        "platoon":   ["pitcher_throws"],
    }
    all_situation_cols = [col for axis_cols in axes.values() for col in axis_cols]

    rows = []
    for (batter_id, stand), group in df.groupby(KEY, sort=False):
        if len(group) < MIN_SWINGS:
            continue
        location_matrix = location_design(group)
        dial_outcomes   = {dial: group[dial].to_numpy(float) for dial in DIALS}

        full_model_r_squared = dial_r_squared_with_situation(
            location_matrix, group, dial_outcomes, all_situation_cols
        )
        location_only_r_squared = dial_r_squared_with_situation(
            location_matrix, group, dial_outcomes, []
        )

        # headline = whole situation over location-only; per-axis = unique add net of location + the OTHER axes
        row = {
            "batter_id":     batter_id,
            "batter_stand":  stand,
            "label":         group["batter_full_name"].iloc[0],
            "n_swings":      len(group),
            "adjustability": round(mean_r_squared_gain(full_model_r_squared, location_only_r_squared), 4),
        }
        for axis_name, axis_cols in axes.items():
            other_situation_cols = [col for other_axis, other_cols in axes.items()
                                    if other_axis != axis_name for col in other_cols]
            baseline_r_squared = dial_r_squared_with_situation(
                location_matrix, group, dial_outcomes, other_situation_cols
            )
            row[f"adj_{axis_name}"] = round(mean_r_squared_gain(full_model_r_squared, baseline_r_squared), 4)
        rows.append(row)

    scores_df = pd.DataFrame(rows).sort_values("adjustability", ascending=False).reset_index(drop=True)
    baseline_mean, baseline_std, baseline_sorted = resolve_reference(scores_df)

    # Scale to 50 + 10·z, clipped to [0, 100], matching the Swing+ / Repertoire+ convention
    z_scores = (scores_df["adjustability"] - baseline_mean) / baseline_std
    scores_df["adjustability_plus"] = (50 + 10 * z_scores).clip(0, 100).round(1)
    scores_df["adjustability_pctile"] = (
        np.searchsorted(baseline_sorted, scores_df["adjustability"].to_numpy(), side="right")
        / len(baseline_sorted) * 100
    ).round(1)

    scores_df.to_parquet(DATA / "adjustability.parquet", index=False)
    print(f"{len(scores_df)} hitters, >= {MIN_SWINGS} swings, {SEASONS}")
    print(f"adjustability: mean {scores_df.adjustability.mean():.3f}, median {scores_df.adjustability.median():.3f}, "
          f"max {scores_df.adjustability.max():.3f}")
    print(f"per-axis means -> count {scores_df.adj_count.mean():.3f}  gamestate {scores_df.adj_gamestate.mean():.3f}  "
          f"pitch {scores_df.adj_pitch.mean():.3f}  platoon {scores_df.adj_platoon.mean():.3f}")
    print("\ntop 8:\n", scores_df.head(8)[["label", "batter_stand", "n_swings", "adjustability",
                                            "adj_count", "adj_gamestate", "adj_pitch", "adj_platoon"]].to_string(index=False))
    print("\nbottom 5:\n", scores_df.tail(5)[["label", "batter_stand", "adjustability",
                                              "adj_count", "adj_gamestate", "adj_pitch", "adj_platoon"]].to_string(index=False))


if __name__ == "__main__":
    main()
