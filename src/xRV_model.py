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
CONTEXT = ["balls", "strikes", "same_hand", "plate_x_pull", "plate_z_norm", "pitch_type"]
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
    cv = pd.read_csv(DATA / "count_values.csv")
    erv = {(int(r.balls), int(r.strikes)): float(r.expected_run_value) for r in cv.itertuples()}
    lw_val = pd.read_csv(DATA / "linear_weights.csv").set_index("outcome_type")["lw_raw"].to_dict()
    lw_k = float(lw_val["strikeout"])

    rv_whiff, rv_foul = {}, {}
    for (b, s) in erv:
        if s < 2:                                     # non-terminal strike advances the count
            rv_whiff[(b, s)] = rv_foul[(b, s)] = erv[(b, s + 1)] - erv[(b, s)]
        else:                                         # 2-strike: whiff -> K (terminal); foul -> no change
            rv_whiff[(b, s)] = lw_k - erv[(b, s)]
            rv_foul[(b, s)] = 0.0
    return {"erv": erv, "lw_val": lw_val, "rv_whiff": rv_whiff, "rv_foul": rv_foul}


def bip_value_target(pa_outcome, lw_val):
    """Batted-ball run value: hits -> their linear weight, everything else -> out_in_play (~ -0.25)."""
    return pa_outcome.map(lambda o: HIT_OUTCOMES.get(o, "out_in_play")).map(lw_val).astype(float)


def build_features(df):
    """Attach the shared predictor columns (context + shape) on a copy."""
    out = df.copy()
    # plate_x is ABSOLUTE (catcher frame), so pull-side/inside location needs a real per-hand flip:
    # verified vs bearing, inside(pull-side) = negative plate_x for RHH, positive for LHH -> flip RHH.
    # + = pull-side / inside for both. (Sign is tree-invariant, so this does not change fitted xRV.)
    side = np.where(out["batter_stand"] == "L", 1.0, -1.0)
    out["plate_x_pull"] = out["plate_x"] * side                 # exogenous pitch location; de-confounds the shape angles
    out["plate_z_norm"] = (out["plate_z"] - out["sz_bot"]) / (out["sz_top"] - out["sz_bot"])
    out["same_hand"] = (out["batter_stand"] == out["pitcher_throws"]).astype(int)   # platoon: 1 = same-side matchup
    out["pitch_type"] = out["pitch_type"].astype("category")
    return out


def assemble_xrv(df, p_bip, p_foul, v_bip, rv):
    """Combine the three model outputs and the run-value tables into per-swing expected xRV."""
    bs = list(zip(df["balls"], df["strikes"]))
    erv = np.array([rv["erv"][k] for k in bs])
    rvw = np.array([rv["rv_whiff"][k] for k in bs])
    rvf = np.array([rv["rv_foul"][k] for k in bs])
    return p_bip * (v_bip - erv) + (1.0 - p_bip) * (p_foul * rvf + (1.0 - p_foul) * rvw)


def realized_run_value(df, rv):
    """The tables' *realized* per-swing run value (actual outcome), for validation vs delta_run_exp."""
    bs = list(zip(df["balls"], df["strikes"]))
    erv = np.array([rv["erv"][k] for k in bs])
    rvw = np.array([rv["rv_whiff"][k] for k in bs])
    rvf = np.array([rv["rv_foul"][k] for k in bs])
    lw = bip_value_target(df["pa_outcome"], rv["lw_val"]).to_numpy()
    return np.where(df["is_bip"].to_numpy() == 1, lw - erv,
           np.where(df["is_whiff"].to_numpy() == 1, rvw, rvf))


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    rv = load_run_value_tables()
    df = build_features(pd.read_parquet(DATA / "swings_model.parquet"))
    train = df[df["game_year"] != HELD_OUT_SEASON]
    test = df[df["game_year"] == HELD_OUT_SEASON]
    print(f"Swings: {len(df):,}  (train {len(train):,} / {HELD_OUT_SEASON} test {len(test):,})")

    # ---- train the three models on 2024-25 with fixed params; score on held-out 2026 ----------
    models, report = {}, {}
    for name, params in PARAMS.items():
        is_clf = name != "v_bip"
        if name == "p_bip":                                    # all competitive swings
            tr, te, ytr, yte = train, test, train["is_bip"], test["is_bip"]
        elif name == "p_foul":                                 # non-BIP swings; contact == foul
            tr, te = train[train.is_bip == 0], test[test.is_bip == 0]
            ytr, yte = tr["is_contact"], te["is_contact"]
        else:                                                  # BIP swings; run value of the batted ball
            tr, te = train[train.is_bip == 1], test[test.is_bip == 1]
            ytr, yte = bip_value_target(tr["pa_outcome"], rv["lw_val"]), bip_value_target(te["pa_outcome"], rv["lw_val"])

        Mk = XGBClassifier if is_clf else XGBRegressor
        m = Mk(tree_method="hist", enable_categorical=True, n_jobs=-1, random_state=SEED,
               eval_metric=("logloss" if is_clf else "rmse"), **params)
        m.fit(tr[FEATURES], ytr)
        m.save_model(MODEL_DIR / f"{name}.json")
        models[name] = m
        if is_clf:
            p = m.predict_proba(te[FEATURES])[:, 1]
            report[name] = {"logloss": log_loss(yte, p), "auc": roc_auc_score(yte, p)}
        else:
            p = m.predict(te[FEATURES])
            report[name] = {"rmse": root_mean_squared_error(yte, p), "r2": r2_score(yte, p)}
        print(f"[{name}] 2026: " + "  ".join(f"{k}={v:.4f}" for k, v in report[name].items()))

    # ---- assemble per-swing xRV on the full set ----------------------------------------------
    pb = models["p_bip"].predict_proba(df[FEATURES])[:, 1]
    pf = models["p_foul"].predict_proba(df[FEATURES])[:, 1]
    vb = models["v_bip"].predict(df[FEATURES])
    out = df[["play_id", "batter_id", "batter_stand", "game_year", "balls", "strikes", "delta_run_exp"]].copy()
    out["p_bip"], out["p_foul"], out["v_bip"] = pb, pf, vb
    out["xrv"] = assemble_xrv(df, pb, pf, vb, rv)
    out["realized_rv"] = realized_run_value(df, rv)
    # 0-100 swing grade: z-score the expected xRV across all swings, center at 50 (10 pts / SD),
    # clip to [0, 100]. 50 = league-average swing; >50 better, <50 worse.
    z = (out["xrv"] - out["xrv"].mean()) / out["xrv"].std()
    out["xrv_grade"] = (50 + 10 * z).clip(0, 100)
    out.to_parquet(DATA / "xrv_swings.parquet", index=False)

    # ---- validation gate: do the tables' realized run values reproduce delta_run_exp? ---------
    mask = out["delta_run_exp"].notna()
    corr = float(np.corrcoef(out.loc[mask, "realized_rv"], out.loc[mask, "delta_run_exp"])[0, 1])
    rmse = root_mean_squared_error(out.loc[mask, "delta_run_exp"], out.loc[mask, "realized_rv"])
    print(f"[validation] realized_rv vs delta_run_exp: corr={corr:.4f} rmse={rmse:.4f}")

    write_report(report, rv, out, corr, rmse)
    print(f"Wrote {MODEL_DIR}/*.json, data/xrv_swings.parquet, data/xrv_report.md")


def write_report(report, rv, out, corr, rmse):
    L = ["# xRV model report\n",
         "Per-swing expected run value from a 3-model swing-outcome decomposition + count/linear-weight "
         "run-value tables. Fixed hyperparameters (experiments/sweep.py); train 2024-25, test 2026.\n",
         f"- Run-value frame (avg-PA-centered `lw_raw`): single={rv['lw_val']['single']:.3f}, "
         f"HR={rv['lw_val']['home_run']:.3f}, out_in_play={rv['lw_val']['out_in_play']:.3f}, "
         f"K={rv['lw_val']['strikeout']:.3f}\n",
         "## Model performance (2026 held-out)"]
    for name, r in report.items():
        L.append(f"- **{name}**: " + ", ".join(f"{k}={v:.4f}" for k, v in r.items()) +
                 f"  `{PARAMS[name]}`")
    L += ["\n## Validation gate vs delta_run_exp",
          f"- realized_rv (actual outcome) vs delta_run_exp: corr **{corr:.4f}**, rmse {rmse:.4f}",
          "> The realized run value must track delta_run_exp before xRV is used downstream "
          "(research-design.md). Aggregate hitter xRV vs pitch_values.ipv is still TODO.\n",
          "## xRV distribution\n", out["xrv"].describe().round(4).to_frame().to_markdown()]
    (DATA / "xrv_report.md").write_text("\n".join(L), encoding="utf-8")
    (MODEL_DIR / "reports.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
