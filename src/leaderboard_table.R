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
    else if (startsWith(name, "adjustability")) "adjustability"
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

# ── ADJValue leaderboard (unit = batter x stand) ─────────────────────────────────────────────────
# Reads data/twostrike_penalties.parquet (written by the adjustability_results.ipynb computation cell).
# ADJValue = z-score composite of two_strike_rv_delta and -two_strike_whiff_delta (both flipped so
# higher = better). Ranked by ADJValue; adjustability shown alongside as context.

if (file.exists("data/twostrike_penalties.parquet")) {
  val_raw <- read_parquet("data/twostrike_penalties.parquet",
                          col_select = c("batter_id", "batter_stand", "label",
                                         "adj_count", "two_strike_rv_delta",
                                         "two_strike_whiff_delta", "swing_plus")) |>
    left_join(read_parquet("data/adjustability.parquet",
                            col_select = c("batter_id", "batter_stand", "adjustability_plus")),
              by = c("batter_id", "batter_stand")) |>
    filter(!is.na(adjustability_plus)) |>
    mutate(
      rv_z      = as.numeric(scale(two_strike_rv_delta)),
      whiff_z   = -as.numeric(scale(two_strike_whiff_delta)),
      raw_val   = (rv_z + whiff_z) / 2,
      ADJValue  = round(pmax(0, pmin(100, 50 + 10 * as.numeric(scale(raw_val)))), 1),
      AdjPlus   = round(adjustability_plus, 1),
      MatchedRV = round(two_strike_rv_delta, 4),
      Whiff2K   = round(two_strike_whiff_delta, 3),
      SwingPlus = round(swing_plus, 1)
    ) |>
    arrange(desc(ADJValue)) |>
    mutate(Rank = row_number())

  n_pool      <- nrow(val_raw)
  pal_adjval  <- col_numeric(PAL_COLS, domain = range(val_raw$ADJValue))

  val_cols   <- c("Rank", "batter_id", "label", "batter_stand", "ADJValue", "AdjPlus", "MatchedRV", "Whiff2K", "SwingPlus")
  val_labels <- list(Rank = "#", batter_id = "", label = "Batter", batter_stand = "R/L",
                     ADJValue = "ADJValue+", AdjPlus = "Adjustability+",
                     MatchedRV = "2K Δ run value", Whiff2K = "2K Δ whiff", SwingPlus = "Swing+")
  val_align  <- c("Rank", "batter_stand", "ADJValue", "AdjPlus", "MatchedRV", "Whiff2K", "SwingPlus")
  val_foot   <- paste0(
    "n=", n_pool, " qualified units (≥400 swings, 2024-25). ",
    "ADJValue+ = 50+10·z composite of 2K matched run value + 2K whiff resilience (50 = avg; higher = better).")

  top_val <- val_raw |> head(TOP_N)
  bot_val <- val_raw |> tail(TOP_N) |> arrange(ADJValue) |> mutate(Rank = row_number())

  make_leaderboard(top_val |> select(all_of(val_cols)),
                   "ADJValue", pal_adjval, val_labels, val_align,
                   "**ADJValue+ Leaderboard — Leaders**",
                   sprintf("Most value from adjustability (50 = avg)  &middot;  n=%d  &middot;  color = ADJValue+", n_pool),
                   fig_path("adjustability_value_top_gt.png"), footnote = val_foot, width = 950)

  make_leaderboard(bot_val |> select(all_of(val_cols)),
                   "ADJValue", pal_adjval, val_labels, val_align,
                   "**ADJValue+ Leaderboard — Worst Performers**",
                   sprintf("Least value from adjustability  &middot;  same pool (n=%d)  &middot;  color = ADJValue+", n_pool),
                   fig_path("adjustability_value_bottom_gt.png"), footnote = val_foot, width = 950)

  # ── Residuals: high Adj+ / low ADJValue+ and low Adj+ / high ADJValue+ (top 5 each) ────────────
  val_fit  <- lm(ADJValue ~ AdjPlus, data = val_raw)
  val_res  <- val_raw |> mutate(ADJResid = round(residuals(val_fit), 1))

  pal_resid   <- col_numeric(PAL_COLS, domain = range(val_res$ADJResid))
  resid_cols  <- c("Rank", "batter_id", "label", "batter_stand", "AdjPlus", "ADJValue", "ADJResid", "MatchedRV", "Whiff2K", "SwingPlus")
  resid_labels <- list(Rank = "#", batter_id = "", label = "Batter", batter_stand = "R/L",
                       AdjPlus = "Adjustability+", ADJValue = "ADJValue+", ADJResid = "Residual",
                       MatchedRV = "2K Δ run value", Whiff2K = "2K Δ whiff", SwingPlus = "Swing+")
  resid_align  <- c("Rank", "batter_stand", "AdjPlus", "ADJValue", "ADJResid", "MatchedRV", "Whiff2K", "SwingPlus")
  resid_foot   <- paste0("Residual = actual ADJValue+ minus expected from linear fit on Adjustability+. n=", n_pool, " pool.")

  high_adj_low_val <- val_res |> arrange(ADJResid)       |> head(5) |> mutate(Rank = row_number())
  low_adj_high_val <- val_res |> arrange(desc(ADJResid)) |> head(5) |> mutate(Rank = row_number())

  make_leaderboard(high_adj_low_val |> select(all_of(resid_cols)),
                   "ADJResid", pal_resid, resid_labels, resid_align,
                   "**High Adjustability+, Low ADJValue+**",
                   sprintf("Skilled adjusters not translating to outcomes  &middot;  n=%d pool", n_pool),
                   fig_path("adjustability_resid_high_adj_low_val_gt.png"), footnote = resid_foot, width = 1000)

  make_leaderboard(low_adj_high_val |> select(all_of(resid_cols)),
                   "ADJResid", pal_resid, resid_labels, resid_align,
                   "**Low Adjustability+, High ADJValue+**",
                   sprintf("Two-strike outcomes without the measured skill  &middot;  n=%d pool", n_pool),
                   fig_path("adjustability_resid_low_adj_high_val_gt.png"), footnote = resid_foot, width = 1000)
} else {
  cat("Skipping ADJValue leaderboard: data/twostrike_penalties.parquet not found.\n",
      "Run the ‘Two-strike outcome gap’ cell in adjustability_results.ipynb first.\n")
}

cat("done\n")
