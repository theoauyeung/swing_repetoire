#!/usr/bin/env python3
"""Generate DML process flowchart for adjustability_value methodology."""
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
WHITE    = "#FFFFFF"

fig, ax = plt.subplots(figsize=(11, 13), facecolor=WHITE)
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
H   = 0.090   # standard box height
HB  = 0.120   # taller for confounders (3 sub-lines)
WS  = 0.30    # side column width
WC  = 0.25    # confounders width
WI  = 0.58    # input bar
WO  = 0.46    # output bar

YIN  = 0.895
YCOL = 0.735
YXGB = 0.560
YRES = 0.385
YOLS = 0.215
YOUT = 0.062

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(XC, 0.968,
        "Double Machine Learning — Does Adjustability Improve Outcomes?",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=C_DARK, transform=ax.transAxes)

# ── Boxes ────────────────────────────────────────────────────────────────────
draw_box(XC, YIN, WI, H,
         "Input Data",
         "2024–25 MLB swings  ·  ≥ 400 swings per batter-stance",
         fc=C_LIGHT, tc=C_DARK)

draw_box(XL, YCOL, WS, H,
         "Adjustability  (T)",
         "count · pitch-type · gamestate",
         fc=C_BLUE, tc=WHITE)

draw_box(XC, YCOL, WC, HB,
         "Confounders",
         "Swing+  ·  Repertoire+\npitcher quality  ·  whiff rate\nhandedness  ·  log swings",
         fc=C_GRAY, tc=WHITE, fs=9.5, sub_dy=0.028)

draw_box(XR, YCOL, WS, H,
         "Outcome  (Y)",
         "run value  ·  whiff rate",
         fc=C_ORANGE, tc=WHITE)

draw_box(XL, YXGB, WS, H,
         "XGBoost: predict T",
         "from confounders",
         fc=C_BLUE, tc=WHITE)
badge(XL, YXGB, WS, H, "5-fold cross-fitting")

draw_box(XR, YXGB, WS, H,
         "XGBoost: predict Y",
         "from confounders",
         fc=C_ORANGE, tc=WHITE)
badge(XR, YXGB, WS, H, "5-fold cross-fitting")

draw_box(XL, YRES, WS, H,
         "T residual",
         "adjustability unexplained\nby confounders",
         fc=C_BLUE, tc=WHITE)

draw_box(XR, YRES, WS, H,
         "Y residual",
         "outcome unexplained\nby confounders",
         fc=C_ORANGE, tc=WHITE)

draw_box(XC, YOLS, 0.44, H,
         "OLS Regression",
         "Y residual ~ T residual",
         fc=C_DARK, tc=WHITE)

draw_box(XC, YOUT, WO, H,
         "θ — standardized causal effect",
         "± sandwich SE  ·  p-value",
         fc=C_GREEN, tc=WHITE, fs=11.5)

# ── Arrows ───────────────────────────────────────────────────────────────────
# Input → Treatment / Confounders / Outcome
arrow(XC - WI / 2 * 0.52, YIN - H / 2, XL, YCOL + H / 2,           C_BLUE)
arrow(XC,                  YIN - H / 2, XC, YCOL + HB / 2,           C_GRAY)
arrow(XC + WI / 2 * 0.52, YIN - H / 2, XR, YCOL + H / 2,            C_ORANGE)

# Treatment / Outcome → XGBoost
arrow(XL, YCOL - H / 2, XL, YXGB + H / 2, C_BLUE)
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
