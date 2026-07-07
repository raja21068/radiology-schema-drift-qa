"""
natstyle.py — a small publication-grade styling layer for the RadAudit-CN figures.

Design goals (Nature Medicine house style):
  * Arial-metric sans (Liberation Sans) at small point sizes
  * de-spined axes (left + bottom only), hairline spines in mid-grey
  * no chartjunk: no boxes, no heavy grids, outward minor ticks
  * a restrained, meaning-bearing palette: pre-transition vs post-transition
    are ALWAYS the same two colours across every panel and every figure
  * direct labelling over legend boxes wherever it fits
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ----------------------------------------------------------------------
# Fonts: Liberation Sans is metric-compatible with Arial (Nature standard)
# ----------------------------------------------------------------------
for _p in [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass

# ----------------------------------------------------------------------
# Palette — locked across all figures
# ----------------------------------------------------------------------
INK      = "#22272E"   # near-black for text, axes, key strokes
PRE      = "#2F5D8A"   # pre-transition era (slate blue)
POST     = "#D1603D"   # post-transition era (terracotta)
PRE_L    = "#9DBAD4"   # light tint of PRE for fills / secondary
POST_L   = "#EBB89F"   # light tint of POST
GREY     = "#8A9099"   # neutral / context series
GREY_L   = "#C9CDD2"   # faint neutral (reference bars, baselines)
GOLD     = "#C9A227"   # sparingly: a third categorical accent
RULE     = "#586069"   # change-point rule / annotation grey
GOOD     = "#3E7C5A"   # occasional positive accent (green)

# A short categorical ramp when >2 unbanded series are needed
CYCLE = [PRE, POST, GREY, GOOD, GOLD]


def set_style():
    """Apply global rcParams. Call once at import time in each figure script."""
    mpl.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.3,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,          # embed as TrueType (editable, no type-3)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # axes
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        # ticks
        "xtick.color": RULE,
        "ytick.color": RULE,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # lines / patches
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        # figure
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_tag(ax, letter, x=-0.16, y=1.06, fontsize=9):
    """Bold lower-case panel letter in the Nature position."""
    ax.text(x, y, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=fontsize, va="top", ha="left", color=INK)


def despine(ax, left=True, bottom=True):
    """Hairline only on the requested spines; trim the rest."""
    for side in ("top", "right", "left", "bottom"):
        keep = (side == "left" and left) or (side == "bottom" and bottom)
        ax.spines[side].set_visible(keep)


def change_rule(ax, x, label=None, y=1.0, color=RULE, lw=0.9, label_dx=0.0):
    """Vertical change-point rule with an optional small word label at top."""
    ax.axvline(x, ls=(0, (4, 3)), lw=lw, color=color, zorder=1)
    if label:
        ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"),
                    xytext=(3 + label_dx, -2), textcoords="offset points",
                    fontsize=6.0, color=color, va="top", ha="left",
                    style="italic")


def era_shade(ax, x_split, xmin, xmax, alpha=0.06):
    """Faint background bands for the pre / post eras."""
    ax.axvspan(xmin, x_split, color=PRE,  alpha=alpha, lw=0, zorder=0)
    ax.axvspan(x_split, xmax, color=POST, alpha=alpha, lw=0, zorder=0)


def end_label(ax, x, y, text, color, dx=4, dy=0, fontsize=6.3, weight="bold"):
    """Direct line label at the series end (replaces a legend entry)."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=fontsize, va="center", ha="left",
                fontweight=weight)


def save(fig, path_noext, also_png=True):
    fig.savefig(f"{path_noext}.pdf")
    if also_png:
        fig.savefig(f"{path_noext}.png", dpi=600)
    plt.close(fig)
