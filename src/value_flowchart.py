#!/usr/bin/env python3
"""Draws the diagram of how adjustment value is put together, from raw swings to the runs
leaderboard and the prescriptions. Presentation only — it reads nothing and changes no numbers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "plots" / "adjustability" / "value_flowchart.png"

C_GRAY = "#5E5E5E"
C_TEAL = "#17858C"
C_BLUE = "#1F77B4"
C_MUTE = "#9A9A9A"
C_ORANGE = "#D45500"
C_DARK = "#2A2A2A"
C_GREEN = "#2C8C3C"
C_PURPLE = "#6A4C93"
WHITE = "#FFFFFF"

fig, ax = plt.subplots(figsize=(12, 15.5), facecolor=WHITE)
ax.set_facecolor(WHITE)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(cx, cy, w, h, title, sub=None, fc=C_MUTE, fs=10.5, sub_fs=None, gap=0.011):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=WHITE, linewidth=2.5,
        transform=ax.transAxes, clip_on=False, zorder=2,
    ))
    if sub:
        n = sub.count("\n") + 1
        top = cy + gap + 0.0075 * (n - 1)
        ax.text(cx, top, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=WHITE,
                transform=ax.transAxes, zorder=3)
        ax.text(cx, top - 0.020, sub, ha="center", va="top",
                fontsize=sub_fs or (fs - 2.0), color=WHITE, alpha=0.92,
                linespacing=1.45, transform=ax.transAxes, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=WHITE, transform=ax.transAxes, zorder=3)


def arrow(x0, y0, x1, y1, color, lw=2.0, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction", zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14,
                                connectionstyle=f"arc3,rad={rad}"))


def note(x, y, text, ha="center", color="#666666", fs=8.0):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, style="italic",
            color=color, transform=ax.transAxes, zorder=4)


Y_IN = 0.930
Y_SHAPE = 0.812
Y_ARM = 0.663
Y_XRV = 0.533
Y_HEAD = 0.418
Y_DEC = 0.298
Y_PRESC = 0.170
Y_VAL = 0.052

ax.text(0.5, 0.982,
        "Adjustment value  ·  2024–25  ·  553,459 swings  ·  466 hitter-stances  ·  ≥ 400 swings each",
        ha="center", va="center", fontsize=9.5, color=C_GRAY, transform=ax.transAxes)

# ── inputs ───────────────────────────────────────────────────────────────────
box(0.27, Y_IN, 0.38, 0.062, "swings_model.parquet",
    "competitive swings · 5 shape dials · situation", fc=C_GRAY, fs=10.0)
box(0.73, Y_IN, 0.38, 0.062, "pitch_chars.parquet",
    "velocity · movement · spin · release", fc=C_GRAY, fs=10.0)

# ── shape model ──────────────────────────────────────────────────────────────
box(0.50, Y_SHAPE, 0.76, 0.108,
    "Per-hitter shape model   (one OLS per unit, per dial)",
    "shape  ~  location surface + pitch_group + pitch characteristics\n"
    "+ count dummies + gamestate dummies + platoon dummies\n"
    "5-fold cross-fitted within unit — both arms share a fold",
    fc=C_TEAL, fs=11.0, sub_fs=9.3)
note(0.50, Y_SHAPE - 0.074,
     "the pitch is a CONTROL, never de-situated — a hitter cannot choose his swing before identifying it")

# ── three arms ───────────────────────────────────────────────────────────────
box(0.175, Y_ARM, 0.30, 0.098, "His actual policy",
    "fitted shape on the\npitch he really saw", fc=C_BLUE, fs=10.5)
box(0.50, Y_ARM, 0.30, 0.098, "De-situated arm",
    "situation dummies set to\ntheir unit MEAN\n(mean-preserving)", fc=C_MUTE, fs=10.5, sub_fs=8.6)
box(0.825, Y_ARM, 0.30, 0.098, "Replacement policy",
    "a random peer's CENTERED\nsituation → shape block,\naveraged over 10 draws", fc=C_ORANGE,
    fs=10.5, sub_fs=8.6)
note(0.805, Y_ARM - 0.064, "same hitter, same pitches —", color="#8A4400")
note(0.805, Y_ARM - 0.077, "only the deviation is swapped", color="#8A4400")

# ── xRV ──────────────────────────────────────────────────────────────────────
box(0.50, Y_XRV, 0.62, 0.082, "Score every arm through xRV",
    "p_bip · p_foul · v_bip  +  count-aware run-value layer", fc=C_DARK, fs=11.0)

# ── headline + diagnostic ────────────────────────────────────────────────────
box(0.375, Y_HEAD, 0.55, 0.084, "runs_total   —   adjustment edge",
    "xRV(his policy) − xRV(replacement's)\n52.4% positive · two-sided by construction",
    fc=C_GREEN, fs=11.5, sub_fs=9.0)
box(0.835, Y_HEAD, 0.26, 0.084, "runs_vs_desituated",
    "adjustment MAGNITUDE\ndiagnostic · 96% positive", fc=C_MUTE, fs=9.5, sub_fs=8.2)

# ── axis decomposition ───────────────────────────────────────────────────────
for x, t, s in [(0.135, "count", "+0.026"),
                (0.375, "gamestate", "−0.002"),
                (0.615, "platoon", "+0.102"),
                (0.860, "interaction", "+0.013")]:
    box(x, Y_DEC, 0.21, 0.062, f"runs_{t}", s, fc=C_BLUE, fs=9.5, sub_fs=8.5)
note(0.50, Y_DEC - 0.046, "the three axes plus their cross-term close the identity by construction")

# ── prescriptive layers ──────────────────────────────────────────────────────
box(0.19, Y_PRESC, 0.32, 0.092, "α scan — how much",
    "rescale his own direction\nleague-capped at Q90\nsplit-half r = 0.30", fc=C_PURPLE,
    fs=10.0, sub_fs=8.5)
box(0.50, Y_PRESC, 0.26, 0.092, "Gradients — which dial",
    "±0.1 SD finite difference\nper situation cell\nVAA r = 0.68", fc=C_PURPLE,
    fs=10.0, sub_fs=8.5)
box(0.815, Y_PRESC, 0.32, 0.092, "Policy search — what to do",
    "mean-preserving contrasts\ntwo feasibility rails\npriced by re-scoring", fc=C_PURPLE,
    fs=10.0, sub_fs=8.5)

# ── validation ───────────────────────────────────────────────────────────────
box(0.50, Y_VAL, 0.84, 0.062, "Validation   ·   verdict: QUALIFIED",
    "placebo  ·  predicts rv_per_swing (p=0.03) but not wOBA (p=0.17)  ·  split-half r = 0.59",
    fc=C_GRAY, fs=10.0, sub_fs=8.8)

# ── arrows ───────────────────────────────────────────────────────────────────
arrow(0.27, Y_IN - 0.031, 0.38, Y_SHAPE + 0.054, C_GRAY, 1.6)
arrow(0.73, Y_IN - 0.031, 0.62, Y_SHAPE + 0.054, C_GRAY, 1.6)

arrow(0.34, Y_SHAPE - 0.054, 0.175, Y_ARM + 0.049, C_TEAL, 1.8)
arrow(0.50, Y_SHAPE - 0.054, 0.50, Y_ARM + 0.049, C_TEAL, 1.8)
arrow(0.66, Y_SHAPE - 0.054, 0.825, Y_ARM + 0.049, C_TEAL, 1.8)

arrow(0.175, Y_ARM - 0.049, 0.28, Y_XRV + 0.041, C_BLUE, 1.8)
arrow(0.50, Y_ARM - 0.049, 0.50, Y_XRV + 0.041, C_MUTE, 1.8)
arrow(0.825, Y_ARM - 0.049, 0.72, Y_XRV + 0.041, C_ORANGE, 1.8)

arrow(0.42, Y_XRV - 0.041, 0.375, Y_HEAD + 0.042, C_GREEN, 2.4)
arrow(0.70, Y_XRV - 0.041, 0.835, Y_HEAD + 0.042, C_MUTE, 1.5)

for x in (0.135, 0.375, 0.615, 0.860):
    arrow(0.375, Y_HEAD - 0.042, x, Y_DEC + 0.031, C_BLUE, 1.3)

arrow(0.375, Y_HEAD - 0.042, 0.19, Y_PRESC + 0.046, C_PURPLE, 1.8, rad=0.22)
arrow(0.375, Y_HEAD - 0.042, 0.50, Y_PRESC + 0.046, C_PURPLE, 1.8, rad=-0.18)
arrow(0.50, Y_PRESC, 0.815, Y_PRESC, C_PURPLE, 1.5)

arrow(0.19, Y_PRESC - 0.046, 0.30, Y_VAL + 0.031, C_GRAY, 1.5)
arrow(0.815, Y_PRESC - 0.046, 0.70, Y_VAL + 0.031, C_GRAY, 1.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=WHITE)
print(f"Saved → {OUT}")
