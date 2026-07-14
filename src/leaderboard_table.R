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
# pool for each metric (not just the shown rows), so the Top 25 reads all-red and the Bottom 25
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

TOP_N              <- 25
MIN_SWINGS         <- 300   # Swing+ leaderboard: >= 300 competitive swings in 2024-25
MIN_CLUSTER_SWINGS <- 100   # by-shape: >= 100 swings in the (batter, stand, cluster) shape
PLOTS              <- "results/plots"
dir.create(PLOTS, showWarnings = FALSE, recursive = TRUE)

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
                             title, subtitle, footnote, out, width = 800, pct_col = NULL) {
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
    tab_header(title = md(title), subtitle = md(subtitle)) |>
    tab_footnote(footnote = footnote, locations = cells_column_labels(all_of(value_col))) |>
    tab_options(table.font.size = 13, data_row.padding = px(3), table.width = px(width))
  save_png(tbl, out)
}

names_df <- read_parquet("data/swings_model.parquet",
                         col_select = c("batter_id", "batter_full_name")) |>
  distinct(batter_id, .keep_all = TRUE)

xrv_grade_by_play <- read_parquet("data/xrv_swings.parquet",
                                  col_select = c("play_id", "game_year", "xrv_grade"))

# ── Swing+ by swing-shape pool (shared by the by-shape leaderboards AND the drill-down) ──────────
# Clusters are per-hitter and NOT comparable across hitters; this ranks individual shapes by value.
# UsageProp = share of the unit's (stance) swings in this shape (denominator incl. sub-100 clusters).

cl_pool <- read_parquet("data/cluster_assignments.parquet",
                        col_select = c("play_id", "batter_id", "batter_stand", "cluster")) |>
  inner_join(xrv_grade_by_play, by = "play_id") |>
  filter(game_year %in% c(2024, 2025)) |>
  group_by(batter_id, batter_stand, cluster) |>
  summarise(Swings = n(), SwingPlus = round(mean(xrv_grade), 1), .groups = "drop_last") |>
  mutate(UsageProp = Swings / sum(Swings)) |>
  ungroup() |>
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
  out  <- file.path(PLOTS, paste0("shape_breakdown_", slug, "_gt.png"))
  make_leaderboard(d |> select(all_of(cl_cols)), "SwingPlus", pal_cl, cl_labels, cl_align,
                   sprintf("**Swing shapes by value - %s**", nm),
                   sprintf("Each of the hitter's shapes ranked by Swing+  &middot;  color = league scale (all %d shapes)", nrow(cl_pool)),
                   cl_foot, out, width = 900, pct_col = "UsageProp")
  quit(save = "no", status = 0)
}

# ── Swing+ (batter) ─────────────────────────────────────────────────────────────

sp_pool <- read_parquet("data/xrv_swings.parquet",
                        col_select = c("batter_id", "game_year", "xrv_grade")) |>
  filter(game_year %in% c(2024, 2025)) |>
  group_by(batter_id) |>
  summarise(Swings = n(),
            SwingPlus = round(mean(xrv_grade), 1),
            .groups = "drop") |>
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
                 "**Swing+ Leaderboard**", sp_sub, sp_foot,
                 file.path(PLOTS, "swingplus_leaderboard_gt.png"), width = 760)

make_leaderboard(tail(sp_pool, TOP_N) |> select(all_of(sp_cols)),
                 "SwingPlus", pal_sp, sp_labels, sp_align,
                 "**Swing+ Leaderboard**", sp_sub, sp_foot,
                 file.path(PLOTS, "swingplus_bottom_gt.png"), width = 760)

# ── Repertoire+ (unit = batter x stand) ─────────────────────────────────────────

unit_swing_plus <- read_parquet("data/cluster_assignments.parquet",
                                col_select = c("play_id", "batter_id", "batter_stand")) |>
  inner_join(xrv_grade_by_play, by = "play_id") |>
  filter(game_year %in% c(2024, 2025)) |>
  group_by(batter_id, batter_stand) |>
  summarise(SwingPlus = round(mean(xrv_grade), 1), .groups = "drop")

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
rep_foot   <- "Repertoire+ is geometry only (no value). Swing+ is the unit's mean swing quality - an independent axis."
rep_sub    <- sprintf("Repertoire width: usage-weighted spread of a hitter's swing shapes (50 = league-average)  &middot;  color spans all %d units",
                      nrow(rep_pool))

make_leaderboard(head(rep_pool, TOP_N) |> select(all_of(rep_cols)),
                 "RepertoirePlus", pal_rep, rep_labels, rep_align,
                 "**Repertoire+ Leaderboard**", rep_sub, rep_foot,
                 file.path(PLOTS, "repertoire_leaderboard_gt.png"), width = 820)

make_leaderboard(tail(rep_pool, TOP_N) |> select(all_of(rep_cols)),
                 "RepertoirePlus", pal_rep, rep_labels, rep_align,
                 "**Repertoire+ Leaderboard**", rep_sub, rep_foot,
                 file.path(PLOTS, "repertoire_bottom_gt.png"), width = 820)

# ── Swing+ by shape (top / bottom), reusing cl_pool + pal_cl from above ──────────────────────────

cl_sub <- sprintf("Value of a single swing shape (>=%d swings)  &middot;  color spans all %d qualified shapes",
                  MIN_CLUSTER_SWINGS, nrow(cl_pool))

make_leaderboard(head(cl_pool, TOP_N) |> select(all_of(cl_cols)),
                 "SwingPlus", pal_cl, cl_labels, cl_align,
                 "**Swing+ by Shape &mdash; Top 25**", cl_sub, cl_foot,
                 file.path(PLOTS, "swingplus_by_cluster_gt.png"), width = 900, pct_col = "UsageProp")

make_leaderboard(tail(cl_pool, TOP_N) |> select(all_of(cl_cols)),
                 "SwingPlus", pal_cl, cl_labels, cl_align,
                 "**Swing+ by Shape &mdash; Bottom 25**", cl_sub, cl_foot,
                 file.path(PLOTS, "swingplus_by_cluster_bottom_gt.png"), width = 900, pct_col = "UsageProp")

cat("done\n")
