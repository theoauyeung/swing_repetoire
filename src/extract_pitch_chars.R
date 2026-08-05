# Pulls how each pitch actually moved — velocity, break, spin, release point — from Baseball
# Savant. Our main swing file only knows where the pitch crossed the plate and what it was called,
# which is not enough to tell a hitter reacting to a nasty slider from one choosing to adjust.
#
# Comes from Savant rather than the club database because that database is not reachable here.
#
# Join key: game_pk + batter_id + balls + strikes + rounded plate_x/plate_z. at_bat_number
# and pitch_number are in extract.py's SQL but were never persisted to the parquet, and
# re-running extract.py needs the DB.
#
# Run: Rscript src/extract_pitch_chars.R   (slow: full-season Savant pulls)

suppressPackageStartupMessages({
  library(sabRmetrics); library(arrow); library(dplyr)
})

SEASONS <- 2024:2026
OUT <- "data/pitch_chars.parquet"

KEEP <- c("game_id", "batter_id", "balls", "strikes", "plate_x", "plate_z",
          "release_speed", "release_spin_rate", "extension",
          "release_pos_x", "release_pos_y", "release_pos_z",
          "pfx_x", "pfx_z", "ax", "ay", "az", "vx0", "vy0", "vz0",
          "spin_axis", "effective_speed", "arm_angle")

pull_season <- function(yr) {
  message("  ", yr, " ...")
  d <- try(sabRmetrics::download_baseballsavant(
    start_date = paste0(yr, "-02-01"), end_date = paste0(yr, "-12-01")), silent = TRUE)
  if (inherits(d, "try-error") || !nrow(d)) {
    message("    no data for ", yr, " — skipped"); return(NULL)
  }
  have <- intersect(KEEP, names(d))
  missing <- setdiff(KEEP, names(d))
  if (length(missing)) message("    absent this season: ", paste(missing, collapse = ", "))
  d |> select(all_of(have)) |> mutate(game_year = yr)
}

message("Downloading Baseball Savant pitch characteristics")
raw <- bind_rows(lapply(SEASONS, pull_season))

out <- raw |>
  rename(game_pk = game_id) |>
  mutate(px_key = round(plate_x, 2), pz_key = round(plate_z, 2)) |>
  select(-plate_x, -plate_z) |>
  # The rounded-coordinate key is near-unique but not guaranteed; drop the rare collisions
  # rather than join them ambiguously.
  add_count(game_pk, batter_id, balls, strikes, px_key, pz_key, name = "key_n") |>
  filter(key_n == 1) |>
  select(-key_n)

message(sprintf("%s rows, %d columns", format(nrow(out), big.mark = ","), ncol(out)))
write_parquet(out, OUT)
message("wrote ", OUT)
