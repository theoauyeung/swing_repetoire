#!/usr/bin/env python3
"""Generate DML process flowchart for adjustability_value methodology (swing-level DML)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results" / "plots" / "dml_flowchart.png"

C_BLUE   = "#1F77B4"
C_ORANGE = "#D45500"
C_GRAY   = "#5E5E5E"
C_GREEN  = "#2CA02C"
C_LIGHT  = "#C0C0C0"
C_DARK   = "#2A2A2A"
C_TEAL   = "#17BECF"
WHITE    = "#FFFFFF"

fig, ax = plt.subplots(figsize=(11, 14), facecolor=WHITE)
ax.set_facecolor(WHITE)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def draw_box(cx, cy, w, h, title, sub=None, fc=C_LIGHT, tc=WHITE, fs=10.5, sub_dy=0.020):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=fc, edgecolor=WHITE, linewidth=2.5,
        transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.add_patch(p)
    if sub:
        ax.text(cx, cy + 0.022, title,
                ha="center", va="center", fontsize=fs, fontweight="bold",
                color=tc, transform=ax.transAxes, zorder=3)
        ax.text(cx, cy - sub_dy, sub,
                ha="center", va="center", fontsize=fs - 1.5, color=tc,
                alpha=0.90, linespacing=1.4, transform=ax.transAxes, zorder=3)
    else:
        ax.text(cx, cy, title,
                ha="center", va="center", fontsize=fs, fontweight="bold",
                color=tc, transform=ax.transAxes, zorder=3)


def arrow(x0, y0, x1, y1, color, lw=2.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                zorder=1,
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    mutation_scale=15, connectionstyle="arc3,rad=0.0",
                ))


def badge(cx, cy, w, h, text):
    ax.text(cx + w / 2 - 0.006, cy + h / 2 + 0.006, text,
            ha="right", va="bottom", fontsize=7.5, style="italic",
            color="#555555", transform=ax.transAxes, zorder=4)


# ── Layout constants ─────────────────────────────────────────────────────────
XL, XC, XR = 0.17, 0.50, 0.83
H    = 0.080   # standard box height
HB   = 0.130   # confounders (4 sub-lines)
HTR  = 0.115   # treatment construction (3 sub-lines)
WS   = 0.30    # side column width
WC   = 0.25    # confounders width
WTR  = 0.42    # treatment construction width (wider to fit 3-line sub)
WI   = 0.58    # input bar width
WO   = 0.54    # output bar width

YIN  = 0.916
YTRT = 0.788   # treatment construction (left-lane pre-step)
YCOL = 0.628   # T | Confounders | Y
YXGB = 0.464
YRES = 0.312
YOLS = 0.172
YOUT = 0.047

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(XC, 0.982,
        "Double Machine Learning — Does Adjustability Improve Outcomes?",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=C_DARK, transform=ax.transAxes)

ax.text(XC, 0.966,
        "Swing-level  ·  2024–25  ·  ~573k swings  ·  ≥ 400 per (batter, stand)",
        ha="center", va="center", fontsize=9, color=C_GRAY,
        transform=ax.transAxes)

# ── Input ────────────────────────────────────────────────────────────────────
draw_box(XC, YIN, WI, H,
         "Input Data",
         "swings_model.parquet  ·  bat_speed · swing_length · swing_path_tilt · location · situation",
         fc=C_LIGHT, tc=C_DARK)

# ── Treatment Construction (new pre-step, left lane) ─────────────────────────
draw_box(XL, YTRT, WTR, HTR,
         "Treatment Construction",
         "5-fold within-batter cross-fitting per dial\n"
         "dial ~ location surface + situation dummies\n"
         "T = mean |fitted shift / dial SD|  (clipped ± 1.5 SD)",
         fc=C_TEAL, tc=WHITE, fs=10.0, sub_dy=0.038)
badge(XL, YTRT, WTR, HTR, "prevents self-contamination")

# ── T | Confounders | Y ──────────────────────────────────────────────────────
draw_box(XL, YCOL, WS, H,
         "Shift magnitude  (T)",
         "per-swing · unsigned\n"
         "composite + 4 per-axis variants",
         fc=C_BLUE, tc=WHITE, sub_dy=0.022)

draw_box(XC, YCOL, WC, HB,
         "Confounders + Batter FE",
         "location surface\n"
         "count · game state · pitch type\n"
         "pitcher quality\n"
         "within-batter demeaning",
         fc=C_GRAY, tc=WHITE, fs=9.0, sub_dy=0.042)

draw_box(XR, YCOL, WS, H,
         "Outcome  (Y)",
         "delta_run_exp  ·  is_whiff\n"
         "all swings  or  two-strike subset",
         fc=C_ORANGE, tc=WHITE, sub_dy=0.022)

# ── XGBoost nuisance models ──────────────────────────────────────────────────
draw_box(XL, YXGB, WS, H,
         "XGBoost: predict T",
         "from confounders",
         fc=C_BLUE, tc=WHITE)
ax.text(XL + WS / 2 - 0.006, YXGB - H / 2 - 0.006,
        "GroupKFold on batter_id  ·  5 folds",
        ha="right", va="top", fontsize=7.5, style="italic",
        color="#555555", transform=ax.transAxes, zorder=4)

draw_box(XR, YXGB, WS, H,
         "XGBoost: predict Y",
         "from confounders",
         fc=C_ORANGE, tc=WHITE)
ax.text(XR + WS / 2 - 0.006, YXGB - H / 2 - 0.006,
        "GroupKFold on batter_id  ·  5 folds",
        ha="right", va="top", fontsize=7.5, style="italic",
        color="#555555", transform=ax.transAxes, zorder=4)

# ── Residuals ────────────────────────────────────────────────────────────────
draw_box(XL, YRES, WS, H,
         "T residual",
         "shift unexplained\nby confounders + batter FE",
         fc=C_BLUE, tc=WHITE)

draw_box(XR, YRES, WS, H,
         "Y residual",
         "outcome unexplained\nby confounders + batter FE",
         fc=C_ORANGE, tc=WHITE)

# ── OLS ─────────────────────────────────────────────────────────────────────
draw_box(XC, YOLS, 0.44, H,
         "OLS Regression",
         "Y residual ~ T residual",
         fc=C_DARK, tc=WHITE)

# ── Output ───────────────────────────────────────────────────────────────────
draw_box(XC, YOUT, WO, H,
         "θ — standardized causal effect",
         "± clustered sandwich SE (by batter)  ·  p-value",
         fc=C_GREEN, tc=WHITE, fs=11.0)

# ── Arrows ───────────────────────────────────────────────────────────────────
# Input → Treatment Construction (left lane)
arrow(XC - WI / 2 * 0.52, YIN - H / 2, XL, YTRT + HTR / 2, C_TEAL)
# Input → Confounders (center)
arrow(XC,                  YIN - H / 2, XC, YCOL + HB / 2,  C_GRAY)
# Input → Y (right)
arrow(XC + WI / 2 * 0.52, YIN - H / 2, XR, YCOL + H / 2,   C_ORANGE)

# Treatment Construction → T
arrow(XL, YTRT - HTR / 2, XL, YCOL + H / 2, C_TEAL)

# T → XGBoost T
arrow(XL, YCOL - H / 2, XL, YXGB + H / 2, C_BLUE)
# Y → XGBoost Y
arrow(XR, YCOL - H / 2, XR, YXGB + H / 2, C_ORANGE)

# Confounders → both XGBoost models (diagonal)
arrow(XC - WC / 2, YCOL - HB * 0.28, XL + WS / 2, YXGB + H * 0.28, C_GRAY, lw=1.5)
arrow(XC + WC / 2, YCOL - HB * 0.28, XR - WS / 2, YXGB + H * 0.28, C_GRAY, lw=1.5)

# XGBoost → Residuals
arrow(XL, YXGB - H / 2, XL, YRES + H / 2, C_BLUE)
arrow(XR, YXGB - H / 2, XR, YRES + H / 2, C_ORANGE)

# Residuals → OLS
arrow(XL + WS * 0.30, YRES - H / 2, XC - 0.10, YOLS + H / 2, C_BLUE)
arrow(XR - WS * 0.30, YRES - H / 2, XC + 0.10, YOLS + H / 2, C_ORANGE)

# OLS → Output
arrow(XC, YOLS - H / 2, XC, YOUT + H / 2, C_GREEN, lw=2.5)

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=WHITE)
print(f"Saved → {OUT}")
