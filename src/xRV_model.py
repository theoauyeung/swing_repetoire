"""Bespoke per-swing expected run value (xRV) via a swing-outcome decomposition.

Conditioned on a competitive swing, the pitch either goes into play, is fouled, or is missed.
Three XGBoost models estimate that outcome tree; a run-value layer (from the count/linear-weight
CSVs) turns the outcome probabilities into an expected run value:

    xRV(swing) =  P(BIP)        * ( V_bip - ERV(b,s) )                        # ball in play
               + (1 - P(BIP)) * [  P(foul | not BIP) * rv_foul(b,s)           # foul
                                 + (1 - P(foul | not BIP)) * rv_whiff(b,s) ]  # whiff

Models (all conditioned on the SAME pre-swing predictors: pitch/situation context + the 5 swing
shape features; NO post-contact mediators -- exit velo / launch angle excluded per design):
  p_bip  : P(ball in play | swing)                        -- XGBClassifier
  p_foul : P(foul | swing, not in play)                   -- XGBClassifier  (whiff = 1 - p_foul)
  v_bip  : E[linear-weight run value of the batted ball]  -- XGBRegressor    ("xwOBACON" piece)

Run-value layer (RE24-style delta = value of resulting state - value of the count you left), both
sides in the AVERAGE-PA-centered frame (avg PA ~= 0) so they are directly comparable:
  ERV(b,s)      count run expectancy            (count_values.csv)
  lw(outcome)   linear_weights.csv `lw_raw`      (avg-PA-centered; an out in play is ~ -0.25, not 0)
  rv_whiff(b,s) = ERV(b,s+1) - ERV(b,s)  for s<2 ;  lw_K - ERV(b,2)  at 2 strikes (K)
  rv_foul(b,s)  = ERV(b,s+1) - ERV(b,s)  for s<2 ;  0                at 2 strikes

Hyperparameters are FIXED (selected by experiments/sweep.py on a 2024-train / 2025-val split); this
script does no tuning -- it trains on 2024-25, holds out 2026, assembles xRV, and validates the
run-value tables against delta_run_exp.

Input:  data/swings_model.parquet, data/{count_values,count_transitions,linear_weights}.csv
Output: data/xrv_models/{p_bip,p_foul,v_bip}.json, data/xrv_swings.parquet, data/xrv_report.md

Run:  python src/xRV_model.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_DIR = DATA / "xrv_models"
HELD_OUT_SEASON = 2026
SEED = 7


SHAPE_FEATURES = ["swing_path_tilt", "swing_length", "bat_speed",
                  "vert_attack_angle", "horz_attack_angle_pull"]
CONTEXT = [ "same_hand", "plate_x_pull", "plate_z_norm", "pitch_type"]
FEATURES = CONTEXT + SHAPE_FEATURES

HIT_OUTCOMES = {"home_run": "home_run", "triple": "triple", "double": "double", "single": "single"}

# ---- fixed hyperparameters from experiments/sweep.py (2024 train / 2025 val).
PARAMS = {
    "p_bip":  dict(max_depth=4, learning_rate=0.08, min_child_weight=20, subsample=0.6,
                   colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=5.0, gamma=0.5, n_estimators=1336),
    "p_foul": dict(max_depth=5, learning_rate=0.05, min_child_weight=50, subsample=0.7,
                   colsample_bytree=0.9, reg_lambda=0.0, reg_alpha=5.0, gamma=0.0, n_estimators=1413),
    "v_bip":  dict(max_depth=3, learning_rate=0.1, min_child_weight=1, subsample=0.9,
                   colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=1.0, gamma=1.0, n_estimators=412),
}


def load_run_value_tables():
    """ERV(b,s), the avg-PA-centered batted-ball value map, and the per-count whiff/foul RV deltas."""
    count_values_df = pd.read_csv(DATA / "count_values.csv")

    # Build the count-run-expectancy dict keyed by (balls, strikes) tuple
    count_run_expectancy = {}
    for row in count_values_df.itertuples():
        count_key = (int(row.balls), int(row.strikes))
        count_run_expectancy[count_key] = float(row.expected_run_value)

    linear_weight_by_outcome = pd.read_csv(DATA / "linear_weights.csv").set_index("outcome_type")["lw_raw"].to_dict()
    strikeout_linear_weight  = float(linear_weight_by_outcome["strikeout"])

    rv_whiff = {}
    rv_foul  = {}
    for (balls, strikes) in count_run_expectancy:
        if strikes < 2:
            # Non-terminal strike: advances the count by one strike for both outcomes
            next_count_erv = count_run_expectancy[(balls, strikes + 1)]
            current_erv    = count_run_expectancy[(balls, strikes)]
            rv_whiff[(balls, strikes)] = next_count_erv - current_erv
            rv_foul[(balls, strikes)]  = next_count_erv - current_erv
        else:
            # 2-strike terminal: whiff = strikeout (linear weight); foul = count stays the same
            rv_whiff[(balls, strikes)] = strikeout_linear_weight - count_run_expectancy[(balls, strikes)]
            rv_foul[(balls, strikes)]  = 0.0

    return {
        "erv":     count_run_expectancy,
        "lw_val":  linear_weight_by_outcome,
        "rv_whiff": rv_whiff,
        "rv_foul":  rv_foul,
    }


def bip_value_target(pa_outcome, linear_weight_by_outcome):
    """Batted-ball run value: hits -> their linear weight, everything else -> out_in_play (~ -0.25)."""
    canonical_outcome = pa_outcome.map(lambda outcome: HIT_OUTCOMES.get(outcome, "out_in_play"))
    run_value         = canonical_outcome.map(linear_weight_by_outcome).astype(float)
    return run_value


def build_features(swings_df):
    """Attach the shared predictor columns (context + shape) on a copy."""
    enriched = swings_df.copy()
    # plate_x is ABSOLUTE (catcher frame), so pull-side/inside location needs a real per-hand flip:
    # verified vs bearing, inside(pull-side) = negative plate_x for RHH, positive for LHH -> flip RHH.
    # + = pull-side / inside for both. (Sign is tree-invariant, so this does not change fitted xRV.)
    pull_side_multiplier    = np.where(enriched["batter_stand"] == "L", 1.0, -1.0)
    enriched["plate_x_pull"] = enriched["plate_x"] * pull_side_multiplier
    strike_zone_height       = enriched["sz_top"] - enriched["sz_bot"]
    enriched["plate_z_norm"] = (enriched["plate_z"] - enriched["sz_bot"]) / strike_zone_height
    enriched["same_hand"]    = (enriched["batter_stand"] == enriched["pitcher_throws"]).astype(int)
    enriched["pitch_type"]   = enriched["pitch_type"].astype("category")
    return enriched


def assemble_xrv(swings_df, prob_bip, prob_foul, value_bip, run_value_tables):
    """Combine the three model outputs and the run-value tables into per-swing expected xRV."""
    count_states         = list(zip(swings_df["balls"], swings_df["strikes"]))
    count_run_expectancy = np.array([run_value_tables["erv"][key]      for key in count_states])
    whiff_run_value      = np.array([run_value_tables["rv_whiff"][key] for key in count_states])
    foul_run_value       = np.array([run_value_tables["rv_foul"][key]  for key in count_states])
    bip_component     = prob_bip * (value_bip - count_run_expectancy)
    non_bip_component = (1.0 - prob_bip) * (prob_foul * foul_run_value + (1.0 - prob_foul) * whiff_run_value)
    return bip_component + non_bip_component


def compute_neutral_run_values(swings_df, run_value_tables):
    """
    Count-weighted average run values for a count-neutral xRV.

    The ML models (p_bip, p_foul, v_bip) already have no count in their CONTEXT.
    The only count-sensitivity in xRV is the run-value layer: rv_whiff at 2 strikes
    is lw_K − ERV(b,2) ≈ −0.25 to −0.45, versus a small advance cost at 0−1 strikes.
    Replacing per-count values with their empirical frequency-weighted averages yields
    xrv_neutral — a count-invariant swing quality measure suitable as a Swing+ control
    when comparing 2-strike and early-count outcomes.

    Weights = empirical count frequency across all swings in swings_df.
    Returns scalar dict: erv_neutral, rv_whiff_neutral, rv_foul_neutral.
    """
    count_counts = swings_df.groupby(["balls", "strikes"]).size()
    total        = count_counts.sum()
    weights      = (count_counts / total).to_dict()

    erv_neutral       = sum(weights.get((b, s), 0.0) * v
                            for (b, s), v in run_value_tables["erv"].items())
    rv_whiff_neutral  = sum(weights.get((b, s), 0.0) * v
                            for (b, s), v in run_value_tables["rv_whiff"].items())
    rv_foul_neutral   = sum(weights.get((b, s), 0.0) * v
                            for (b, s), v in run_value_tables["rv_foul"].items())
    return {
        "erv_neutral":      float(erv_neutral),
        "rv_whiff_neutral": float(rv_whiff_neutral),
        "rv_foul_neutral":  float(rv_foul_neutral),
    }


def assemble_xrv_neutral(prob_bip, prob_foul, value_bip, neutral_rv):
    """Count-neutral xRV: same ML models, league-average run values instead of per-count."""
    erv   = neutral_rv["erv_neutral"]
    r_wh  = neutral_rv["rv_whiff_neutral"]
    r_fo  = neutral_rv["rv_foul_neutral"]
    return (prob_bip * (value_bip - erv)
            + (1.0 - prob_bip) * (prob_foul * r_fo + (1.0 - prob_foul) * r_wh))


def realized_run_value(swings_df, run_value_tables):
    """The tables' *realized* per-swing run value (actual outcome), for validation vs delta_run_exp."""
    count_states         = list(zip(swings_df["balls"], swings_df["strikes"]))
    count_run_expectancy = np.array([run_value_tables["erv"][key]      for key in count_states])
    whiff_run_value      = np.array([run_value_tables["rv_whiff"][key] for key in count_states])
    foul_run_value       = np.array([run_value_tables["rv_foul"][key]  for key in count_states])
    batted_ball_lw       = bip_value_target(swings_df["pa_outcome"], run_value_tables["lw_val"]).to_numpy()
    is_bip   = swings_df["is_bip"].to_numpy() == 1
    is_whiff = swings_df["is_whiff"].to_numpy() == 1
    return np.where(is_bip, batted_ball_lw - count_run_expectancy,
           np.where(is_whiff, whiff_run_value, foul_run_value))


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    run_value_tables = load_run_value_tables()
    all_swings   = build_features(pd.read_parquet(DATA / "swings_model.parquet"))
    train_swings = all_swings[all_swings["game_year"] != HELD_OUT_SEASON]
    test_swings  = all_swings[all_swings["game_year"] == HELD_OUT_SEASON]
    print(f"Swings: {len(all_swings):,}  (train {len(train_swings):,} / {HELD_OUT_SEASON} test {len(test_swings):,})")

    # ---- train the three models on 2024-25 with fixed params; score on held-out 2026 ----------
    models = {}
    report = {}
    for model_name, model_params in PARAMS.items():
        is_classifier = model_name != "v_bip"
        if model_name == "p_bip":                                    # all competitive swings
            train_subset  = train_swings
            test_subset   = test_swings
            train_labels  = train_swings["is_bip"]
            test_labels   = test_swings["is_bip"]
        elif model_name == "p_foul":                                 # non-BIP swings; contact == foul
            train_subset  = train_swings[train_swings.is_bip == 0]
            test_subset   = test_swings[test_swings.is_bip == 0]
            train_labels  = train_subset["is_contact"]
            test_labels   = test_subset["is_contact"]
        else:                                                        # BIP swings; run value of the batted ball
            train_subset  = train_swings[train_swings.is_bip == 1]
            test_subset   = test_swings[test_swings.is_bip == 1]
            train_labels  = bip_value_target(train_subset["pa_outcome"], run_value_tables["lw_val"])
            test_labels   = bip_value_target(test_subset["pa_outcome"],  run_value_tables["lw_val"])

        ModelClass = XGBClassifier if is_classifier else XGBRegressor
        model = ModelClass(tree_method="hist", enable_categorical=True, n_jobs=-1, random_state=SEED,
                           eval_metric=("logloss" if is_classifier else "rmse"), **model_params)
        model.fit(train_subset[FEATURES], train_labels)
        model.save_model(MODEL_DIR / f"{model_name}.json")
        models[model_name] = model

        if is_classifier:
            test_probs = model.predict_proba(test_subset[FEATURES])[:, 1]
            report[model_name] = {"logloss": log_loss(test_labels, test_probs),
                                  "auc":     roc_auc_score(test_labels, test_probs)}
        else:
            test_preds = model.predict(test_subset[FEATURES])
            report[model_name] = {"rmse": root_mean_squared_error(test_labels, test_preds),
                                  "r2":   r2_score(test_labels, test_preds)}
        metrics_str = "  ".join(f"{key}={value:.4f}" for key, value in report[model_name].items())
        print(f"[{model_name}] 2026: {metrics_str}")

    # ---- assemble per-swing xRV on the full set ----------------------------------------------
    prob_bip_all   = models["p_bip"].predict_proba(all_swings[FEATURES])[:, 1]
    prob_foul_all  = models["p_foul"].predict_proba(all_swings[FEATURES])[:, 1]
    value_bip_all  = models["v_bip"].predict(all_swings[FEATURES])

    output_df = all_swings[["play_id", "batter_id", "batter_stand", "game_year",
                             "balls", "strikes", "delta_run_exp"]].copy()
    output_df["p_bip"]       = prob_bip_all
    output_df["p_foul"]      = prob_foul_all
    output_df["v_bip"]       = value_bip_all
    output_df["xrv"]         = assemble_xrv(all_swings, prob_bip_all, prob_foul_all, value_bip_all, run_value_tables)
    output_df["realized_rv"] = realized_run_value(all_swings, run_value_tables)

    # Standard grade: count-inclusive run-value layer (suitable for Swing+ leaderboard).
    xrv_mean    = output_df["xrv"].mean()
    xrv_std     = output_df["xrv"].std()
    output_df["xrv_grade"] = (50 + 10 * (output_df["xrv"] - xrv_mean) / xrv_std).clip(0, 100)

    # Count-neutral grade: same ML models, league-average run values instead of per-count.
    # Use this as the Swing+ control in between-batter regressions that also use count-stratified
    # outcomes (e.g. twostrike_rv_penalty), so the control doesn't absorb the count effect.
    neutral_rv = compute_neutral_run_values(train_swings, run_value_tables)
    xrv_neutral = assemble_xrv_neutral(prob_bip_all, prob_foul_all, value_bip_all, neutral_rv)
    xrv_neutral_mean = xrv_neutral.mean()
    xrv_neutral_std  = xrv_neutral.std()
    output_df["xrv_neutral"]       = xrv_neutral
    output_df["xrv_grade_neutral"] = (50 + 10 * (xrv_neutral - xrv_neutral_mean) / xrv_neutral_std).clip(0, 100)

    output_df.to_parquet(DATA / "xrv_swings.parquet", index=False)
    print(f"  xrv_grade         mean={output_df.xrv_grade.mean():.2f}  std={output_df.xrv_grade.std():.2f}")
    print(f"  xrv_grade_neutral mean={output_df.xrv_grade_neutral.mean():.2f}  std={output_df.xrv_grade_neutral.std():.2f}")

    # ---- validation gate: do the tables' realized run values reproduce delta_run_exp? ---------
    has_delta_run_exp  = output_df["delta_run_exp"].notna()
    validation_actual  = output_df.loc[has_delta_run_exp, "delta_run_exp"]
    validation_realized = output_df.loc[has_delta_run_exp, "realized_rv"]
    validation_corr = float(np.corrcoef(validation_realized, validation_actual)[0, 1])
    validation_rmse = root_mean_squared_error(validation_actual, validation_realized)
    print(f"[validation] realized_rv vs delta_run_exp: corr={validation_corr:.4f} rmse={validation_rmse:.4f}")

    write_report(report, run_value_tables, output_df, validation_corr, validation_rmse)
    print(f"Wrote {MODEL_DIR}/*.json, data/xrv_swings.parquet, data/xrv_report.md")


def write_report(report, run_value_tables, output_df, validation_corr, validation_rmse):
    linear_weights = run_value_tables["lw_val"]
    lines = [
        "# xRV model report\n",
        "Per-swing expected run value from a 3-model swing-outcome decomposition + count/linear-weight "
        "run-value tables. Fixed hyperparameters (experiments/sweep.py); train 2024-25, test 2026.\n",
        f"- Run-value frame (avg-PA-centered `lw_raw`): single={linear_weights['single']:.3f}, "
        f"HR={linear_weights['home_run']:.3f}, out_in_play={linear_weights['out_in_play']:.3f}, "
        f"K={linear_weights['strikeout']:.3f}\n",
        "## Model performance (2026 held-out)",
    ]
    for model_name, model_metrics in report.items():
        metrics_str = ", ".join(f"{key}={value:.4f}" for key, value in model_metrics.items())
        lines.append(f"- **{model_name}**: {metrics_str}  `{PARAMS[model_name]}`")
    lines += [
        "\n## Validation gate vs delta_run_exp",
        f"- realized_rv (actual outcome) vs delta_run_exp: corr **{validation_corr:.4f}**, rmse {validation_rmse:.4f}",
        "> The realized run value must track delta_run_exp before xRV is used downstream "
        "(research-design.md). Aggregate hitter xRV vs pitch_values.ipv is still TODO.\n",
        "## xRV distribution\n",
        output_df["xrv"].describe().round(4).to_frame().to_markdown(),
    ]
    (DATA / "xrv_report.md").write_text("\n".join(lines), encoding="utf-8")
    (MODEL_DIR / "reports.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
