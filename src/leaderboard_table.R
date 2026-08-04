# leaderboard_table.R
# Aesthetic Swing+ / Repertoire+ / Swing+-by-shape leaderboards using gt + mlbplotR (MLBAM
# headshots). Mirrors the pitcher-leaderboard style from the disruption-tax project.
#
# Two modes:
#   Rscript src/leaderboard_table.R              # build all 6 leaderboards (top/bottom x 3)
#   Rscript src/leaderboard_table.R "Arraez"     # per-hitter drill-down: that batter's shapes only
#
# Required packages:
#   install.packages(c("arrow","dplyr","gt","gtExtras","mlbplotR","scales","webshot2"))
#
# batter_id is the MLBAM id, which is what mlbplotR keys headshots on.
# Coloring: diverging blue(low) -> white -> red(high), with the domain fixed to the FULL qualified
# pool for each metric (not just the shown rows), so the Top N reads all-red and the Bottom N
# all-blue -- and a drill-down's shapes sit on the same league scale.
# Writes results/plots/{swingplus,repertoire}_leaderboard_gt.png + *_bottom_gt.png + by-cluster pair,
# or shape_breakdown_<name>_gt.png in drill mode.

suppressMessages({
  library(arrow)
  library(dplyr)
  library(gt)
  library(gtExtras)
  library(mlbplotR)
  library(scales)
})

TOP_N              <- 10
MIN_SWINGS         <- 750   # Swing+ leaderboard: >= 300 competitive swings in 2024-25
MIN_CLUSTER_SWINGS <- 100   # by-shape: >= 100 swings in the (batter, stand, cluster) shape
PLOTS              <- "results/plots"
dir.create(PLOTS, showWarnings = FALSE, recursive = TRUE)

# Route a figure filename to its results/plots/<category> subfolder.
fig_path <- function(name) {
  sub <- if (startsWith(name, "repertoire")) "repertoire"
    else if (startsWith(name, "adjustability") || startsWith(name, "adj_")) "adjustability"
    else if (grepl("^(swingplus_leaderboard|swingplus_bottom|swingplus_by_cluster|shape_breakdown)", name)) "swing_plus"
    else "predictiveness"
  d <- file.path(PLOTS, sub)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  file.path(d, name)
}

PAL_COLS <- c("#2166ac", "#f7f7f7", "#d73027")   # low = blue -> high = red

args  <- commandArgs(trailingOnly = TRUE)
DRILL <- if (length(args) >= 1 && nzchar(args[[1]])) args[[1]] else NA_character_

save_png <- function(tbl, path) {
  tryCatch(
    { gtsave(tbl, path, vwidth = 1200, vheight = 1200); cat("Saved", path, "\n") },
    error = function(e) cat("PNG export skipped:", conditionMessage(e), "\n")
  )
}

# One shared builder. `value_col` is the colored metric, `pal` a col_numeric closure whose domain is
# fixed to the full pool, `labels` a named list of pretty headers, `pct_col` (optional) a proportion
# column rendered as a percent.
make_leaderboard <- function(df, value_col, pal, labels, align_cols,
                             title, subtitle, out, footnote = NULL, width = 800, pct_col = NULL) {
  tbl <- df |>
    gt() |>
    gt_theme_538(quiet = TRUE) |>
    gt_fmt_mlb_headshot(columns = batter_id, height = 34) |>
    cols_label(.list = labels) |>
    data_color(columns = all_of(value_col), fn = pal) |>
    cols_align(align = "center", columns = all_of(align_cols)) |>
    sub_missing(missing_text = "-")
  if (!is.null(pct_col)) tbl <- tbl |> fmt_percent(columns = all_of(pct_col), decimals = 1)
  tbl <- tbl |>
    tab_header(title = md(title), subtitle = md(subtitle))
  if (!is.null(footnote))
    tbl <- tbl |> tab_footnote(footnote = footnote, locations = cells_column_labels(all_of(value_col)))
  tbl <- tbl |>
    tab_options(table.font.size = 13, data_row.padding = px(3), table.width = px(width))
  save_png(tbl, out)
}

# Pre-aggregated by src/leaderboard_preprocess.py — all small files, no large parquet reads in R.
names_df <- read_parquet("data/r_batter_names.parquet")

# ── Swing+ by swing-shape pool (shared by the by-shape leaderboards AND the drill-down) ──────────
# Clusters are per-hitter and NOT comparable across hitters; this ranks individual shapes by value.
# UsageProp = share of the unit's (stance) swings in this shape (denominator incl. sub-100 clusters).

cl_pool <- read_parquet("data/r_swing_plus_shape.parquet") |>
  filter(Swings >= MIN_CLUSTER_SWINGS) |>
  left_join(read_parquet("data/batter_repertoire.parquet",
                         col_select = c("batter_id", "batter_stand", "label")),
            by = c("batter_id", "batter_stand")) |>
  left_join(read_parquet("data/shape_cards.parquet",
                         col_select = c("batter_id", "batter_stand", "cluster", "archetype_detailed")),
            by = c("batter_id", "batter_stand", "cluster")) |>
  arrange(desc(SwingPlus)) |>
  mutate(Rank = row_number())

pal_cl <- col_numeric(PAL_COLS, domain = range(cl_pool$SwingPlus))
cl_labels <- list(Rank = "#", batter_id = "", label = "Batter", cluster = "Cluster",
                  archetype_detailed = "Archetype (· situation)", UsageProp = "% of swings", SwingPlus = "Swing+")
cl_align  <- c("Rank", "cluster", "UsageProp", "SwingPlus")
cl_cols   <- c("Rank", "batter_id", "label", "cluster", "archetype_detailed", "UsageProp", "SwingPlus")
cl_foot   <- "'% of swings' = share of the hitter's (stance) swings in this shape. Cluster 0 = primary swing. Clusters are per-hitter; this ranks individual shapes, not batters."

# ── Drill mode: one hitter's shapes ranked by value, colored on the league scale ────────────────
if (!is.na(DRILL)) {
  d <- cl_pool |>
    filter(grepl(DRILL, label, ignore.case = TRUE)) |>
    arrange(desc(SwingPlus)) |>
    mutate(Rank = row_number())
  if (nrow(d) == 0) {
    cat("No batter matching:", DRILL, "\n")
    quit(save = "no", status = 0)
  }
  nm   <- paste(sort(unique(d$label)), collapse = " / ")   # actual hitter name(s), not the search term
  slug <- gsub("(^_|_$)", "", gsub("[^a-z0-9]+", "_", tolower(DRILL)))
  out  <- fig_path(paste0("shape_breakdown_", slug, "_gt.png"))
  make_leaderboard(d |> select(all_of(cl_cols)), "SwingPlus", pal_cl, cl_labels, cl_align,
                   sprintf("**Swing shapes by value - %s**", nm),
                   sprintf("Each of the hitter's shapes ranked by Swing+  &middot;  color = league scale (all %d shapes)", nrow(cl_pool)),
                   out, footnote = cl_foot, width = 900, pct_col = "UsageProp")
  quit(save = "no", status = 0)
}

# ── Swing+ (batter) ─────────────────────────────────────────────────────────────

sp_pool <- read_parquet("data/r_swing_plus_batter.parquet") |>
  filter(Swings >= MIN_SWINGS) |>
  left_join(names_df, by = "batter_id") |>
  arrange(desc(SwingPlus)) |>
  mutate(Rank = row_number())

pal_sp <- col_numeric(PAL_COLS, domain = range(sp_pool$SwingPlus))
sp_labels <- list(Rank = "#", batter_id = "", batter_full_name = "Batter",
                  Swings = "Swings", SwingPlus = "Swing+")
sp_align  <- c("Rank", "Swings", "SwingPlus")
sp_cols   <- c("Rank", "batter_id", "batter_full_name", "Swings", "SwingPlus")
sp_foot   <- "Swing+ = batter mean of xrv_grade (per-swing xRV z-scored, 50 + 10z, clipped 0-100)."
sp_sub    <- sprintf("Mean per-swing xRV, 0-100 scale (50 = league-average)  &middot;  &ge;%d swings  &middot;  color spans all %d qualified batters",
                     MIN_SWINGS, nrow(sp_pool))

make_leaderboard(head(sp_pool, TOP_N) |> select(all_of(sp_cols)),
                 "SwingPlus", pal_sp, sp_labels, sp_align,
                 "**Swing+ Leaderboard**", sp_sub,
                 fig_path("swingplus_leaderboard_gt.png"), footnote = sp_foot, width = 760)

make_leaderboard(tail(sp_pool, TOP_N) |> select(all_of(sp_cols)),
                 "SwingPlus", pal_sp, sp_labels, sp_align,
                 "**Swing+ Leaderboard**", sp_sub,
                 fig_path("swingplus_bottom_gt.png"), footnote = sp_foot, width = 760)

# ── Repertoire+ (unit = batter x stand) ─────────────────────────────────────────

unit_swing_plus <- read_parquet("data/r_swing_plus_unit.parquet")

rep_pool <- read_parquet("data/repertoire_scores.parquet",
                         col_select = c("batter_id", "batter_stand", "label", "k",
                                        "repertoire_plus", "repertoire_pctile")) |>
  left_join(unit_swing_plus, by = c("batter_id", "batter_stand")) |>
  mutate(RepertoirePlus = round(repertoire_plus, 1)) |>
  arrange(desc(RepertoirePlus)) |>
  mutate(Rank = row_number())

pal_rep <- col_numeric(PAL_COLS, domain = range(rep_pool$RepertoirePlus))
rep_labels <- list(Rank = "#", batter_id = "", label = "Batter", batter_stand = "R/L",
                   k = "Shapes (k)", RepertoirePlus = "Repertoire+", SwingPlus = "Swing+")
rep_align  <- c("Rank", "batter_stand", "k", "RepertoirePlus", "SwingPlus")
rep_cols   <- c("Rank", "batter_id", "label", "batter_stand", "k", "RepertoirePlus", "SwingPlus")
rep_sub    <- sprintf("Repertoire width: usage-weighted shape spread × effective # of shapes (50 = league-average)  &middot;  color spans all %d units",
                      nrow(rep_pool))

make_leaderboard(head(rep_pool, TOP_N) |> select(all_of(rep_cols)),
                 "RepertoirePlus", pal_rep, rep_labels, rep_align,
                 "**Repertoire+ Leaderboard**", rep_sub,
                 fig_path("repertoire_leaderboard_gt.png"), width = 820)

make_leaderboard(tail(rep_pool, TOP_N) |> select(all_of(rep_cols)),
                 "RepertoirePlus", pal_rep, rep_labels, rep_align,
                 "**Repertoire+ Leaderboard**", rep_sub,
                 fig_path("repertoire_bottom_gt.png"), width = 820)

# ── Swing+ by shape (top / bottom), reusing cl_pool + pal_cl from above ──────────────────────────

cl_sub <- sprintf("Value of a single swing shape (>=%d swings)  &middot;  color spans all %d qualified shapes",
                  MIN_CLUSTER_SWINGS, nrow(cl_pool))

make_leaderboard(head(cl_pool, TOP_N) |> select(all_of(cl_cols)),
                 "SwingPlus", pal_cl, cl_labels, cl_align,
                 "**Swing+ by Shape**", cl_sub,
                 fig_path("swingplus_by_cluster_gt.png"), footnote = cl_foot, width = 900, pct_col = "UsageProp")

make_leaderboard(tail(cl_pool, TOP_N) |> select(all_of(cl_cols)),
                 "SwingPlus", pal_cl, cl_labels, cl_align,
                 "**Swing+ by Shape**", cl_sub,
                 fig_path("swingplus_by_cluster_bottom_gt.png"), footnote = cl_foot, width = 900, pct_col = "UsageProp")

# ── Adjustability (unit = batter x stand) ────────────────────────────────────────
# How much a hitter reshapes his swing by situation, net of pitch location (adjusted-R^2 magnitude on
# the trait dials; see src/adjustability.py). Ranked by composite `adjustability` (v4: count-led,
# corr adj_count 0.72 > adj_pitch 0.57). Count and pitch axes shown as breakdown columns.

adj_pool <- read_parquet("data/adjustability.parquet",
                         col_select = c("batter_id", "batter_stand", "label", "n_swings",
                                        "adjustability_plus", "adj_count", "adj_pitch")) |>
  mutate(AdjPlus = round(adjustability_plus, 1),
         Count   = round(adj_count, 3),
         Pitch   = round(adj_pitch, 3)) |>
  arrange(desc(AdjPlus)) |>
  mutate(Rank = row_number())

pal_adj <- col_numeric(PAL_COLS, domain = range(adj_pool$AdjPlus))
adj_labels <- list(Rank = "#", batter_id = "", label = "Batter", batter_stand = "R/L",
                   AdjPlus = "Adjustability+", Count = "Count axis", Pitch = "Pitch axis")
adj_align  <- c("Rank", "batter_stand", "AdjPlus", "Count", "Pitch")
adj_cols   <- c("Rank", "batter_id", "label", "batter_stand", "AdjPlus", "Count", "Pitch")
adj_sub    <- sprintf("Situational swing change, net of pitch location (50 = league-average)  &middot;  &ge;%d swings 2024-25  &middot;  color spans all %d qualified units",
                      400, nrow(adj_pool))

make_leaderboard(head(adj_pool, TOP_N) |> select(all_of(adj_cols)),
                 "AdjPlus", pal_adj, adj_labels, adj_align,
                 "**Adjustability+ Leaderboard**", adj_sub,
                 fig_path("adjustability_leaderboard_gt.png"), width = 820)

make_leaderboard(tail(adj_pool, TOP_N) |> select(all_of(adj_cols)),
                 "AdjPlus", pal_adj, adj_labels, adj_align,
                 "**Adjustability+ Leaderboard**", adj_sub,
                 fig_path("adjustability_bottom_gt.png"), width = 820)

# ── Adjustability value: counterfactual season runs (unit = batter x stand) ──────────────────────
# Reads data/adjustability_value.parquet (written by src/adjustability_value.py).
# runs_total = season runs from situational swing changes vs the de-situated counterfactual:
# the swing the hitter would have made on the same pitch with count, base state and matchup held
# at his own mean. Priced by xRV at the real count. Positive = adjusting earns you runs.
# 93% of units are positive, but the bottom 20 are all negative (-1.68 to -0.12), so the trailers
# table really is units losing runs, not merely gaining the least.
# Platoon runs are near zero for switch-hitter units: within a single stance a switch hitter faces
# almost no same-hand pitching, so there is little platoon variation left to de-situate.

pol_raw <- read_parquet("data/adjustability_value.parquet",
                        col_select = c("batter_id", "label", "batter_stand",
                                       "runs_total", "runs_count",
                                       "runs_gamestate", "runs_platoon",
                                       "marginal_runs_per_alpha",
                                       "adjustability_plus", "swing_plus")) |>
  filter(!is.na(runs_total)) |>
  mutate(
    Runs        = round(runs_total, 1),
    RunsCount   = round(runs_count, 1),
    RunsGS      = round(runs_gamestate, 1),
    RunsPlatoon = round(runs_platoon, 1),
    Marginal    = round(marginal_runs_per_alpha, 1),
    Adj         = round(adjustability_plus, 1),
    Swing       = round(swing_plus, 1)
  ) |>
  arrange(desc(Runs)) |>
  mutate(Rank = row_number())

n_pol    <- nrow(pol_raw)
pal_pol  <- col_numeric(PAL_COLS, domain = range(pol_raw$Runs, na.rm = TRUE))

pol_cols   <- c("Rank", "batter_id", "label", "batter_stand",
                "Runs", "RunsCount", "RunsGS", "RunsPlatoon", "Marginal", "Adj", "Swing")
pol_labels <- list(Rank = "#", batter_id = "", label = "Batter", batter_stand = "R/L",
                   Runs = "Total Runs", RunsCount = "Count",
                   RunsGS = "Game St", RunsPlatoon = "Platoon",
                   Marginal = "Marginal", Adj = "Adj+", Swing = "Swing+")
pol_align  <- c("Rank", "batter_stand", "Runs", "RunsCount", "RunsGS", "RunsPlatoon",
                "Marginal", "Adj", "Swing")
pol_foot   <- paste0(
  "Season runs from situational swing changes vs a de-situated counterfactual: the swing ",
  "the hitter would have made on the same pitch with count, base state and matchup held at ",
  "his own mean. Priced by xRV at the real count. Pitch type is a control, never ",
  "de-situated. 2024-25, min 400 swings. Marginal = runs per unit of extra situational ",
  "responsiveness at current behavior. ",
  "Platoon is near zero for switch-hitter units, which face almost no same-hand pitching ",
  "within a stance.")

top_pol <- pol_raw |> head(TOP_N)
bot_pol <- pol_raw |> tail(TOP_N) |> arrange(Runs) |> mutate(Rank = row_number())

make_leaderboard(top_pol |> select(all_of(pol_cols)),
                 "Runs", pal_pol, pol_labels, pol_align,
                 "**Counterfactual Adjustment Runs — Leaders**",
                 sprintf("Season runs gained from situational swing changes  &middot;  n=%d  &middot;  color = Total Runs", n_pol),
                 fig_path("adjustability_value_top_gt.png"), footnote = pol_foot, width = 1050)

make_leaderboard(bot_pol |> select(all_of(pol_cols)),
                 "Runs", pal_pol, pol_labels, pol_align,
                 "**Counterfactual Adjustment Runs — Trailers**",
                 sprintf("Units losing the most runs to situational swing changes  &middot;  same pool (n=%d)  &middot;  color = Total Runs", n_pol),
                 fig_path("adjustability_value_bottom_gt.png"), footnote = pol_foot, width = 1050)

cat("done\n")
