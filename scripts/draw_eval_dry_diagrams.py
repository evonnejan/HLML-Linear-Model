"""Generate slide-friendly PNG diagrams for the eval_dry pipeline.

Outputs (high-DPI, 16:9, English labels, soft matplotlib default palette):
  docs/figures/eval_dry/eval_dry_pipeline.png
  docs/figures/eval_dry/eval_dry_window_timeline.png

Constants are hard-coded to match the current eval_dry.py + training setup:
  seq_len         = 60   (best run we're documenting uses sl=60)
  pred_len        = 15
  POST_WINDOW     = 60   (Data_From_SQL_4.py POST_WINDOW_MINUTES)
  min_dry_minutes = 180  (post_rain group upper bound = 180)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- hard-coded constants (matches current eval_dry + best run) ---
SEQ_LEN = 60
PRED_LEN = 15
POST_WINDOW = 60
POST_RAIN_MAX_MSR = 180

# --- output paths ---
OUT_DIR = Path("docs/figures/eval_dry")


# =========================================================================== #
#  Figure 1: end-to-end pipeline                                              #
# =========================================================================== #

def draw_pipeline_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_aspect("auto")
    ax.axis("off")

    # Color palette (soft matplotlib defaults)
    c_input = "#E0E0E0"      # gray
    c_shared = "#AEC6E5"     # soft blue (C0-ish lighter)
    c_loop = "#F4B183"       # soft orange (C1-ish lighter)
    c_output = "#A9D08E"     # soft green (C2-ish lighter)
    edge_normal = "#444444"
    edge_loop = "#C00000"    # dashed border for loop

    title_kw = dict(ha="center", va="top", fontsize=14, fontweight="bold")
    body_kw = dict(ha="left", va="top", fontsize=11)

    def add_box(x, y, w, h, color, edge_color, lw, dashed=False):
        style = "round,pad=0.6"
        bbox = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=style,
            linewidth=lw,
            facecolor=color,
            edgecolor=edge_color,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(bbox)

    def add_arrow(x0, y0, x1, y1):
        arr = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="->,head_length=8,head_width=6",
            linewidth=2,
            color=edge_normal,
        )
        ax.add_patch(arr)

    # ----- Inputs (column 1) -----
    add_box(2, 14, 16, 38, c_input, edge_normal, 1.5)
    ax.text(10, 50, "Inputs", **title_kw)
    ax.text(3.5, 46, "• train CSV\n  (water_level_rain_gate_all.csv)", **body_kw)
    ax.text(3.5, 39.5, "• all_minute_wide.csv", **body_kw)
    ax.text(3.5, 36, "• rain_segments_meta.csv", **body_kw)
    ax.text(3.5, 32, "• run_args.json", **body_kw)
    ax.text(3.5, 28, "• checkpoint.pth\n• checkpoint_alt.pth", **body_kw)

    # ----- Shared processing (column 2) -----
    add_box(23, 8, 26, 44, c_shared, edge_normal, 1.5)
    ax.text(36, 50, "Shared processing", **title_kw)
    steps_shared = [
        "1.  Fit scaler from train CSV\n     (StandardScaler on train rain windows)",
        "2.  Label each minute:\n     • in_rain_window\n     • minutes_since_last_rain (msr)",
        f"3.  Find dry blocks\n     in_rain_window=False, length ≥ 180 min",
        "4.  Per-(dry-segment) gate ffill\n     (mirrors training per-rain-segment ffill)",
        f"5.  Slice sliding windows\n     stride=1, input={SEQ_LEN}, target={PRED_LEN}\n     → bundle (x, y_true, persist, meta)",
    ]
    y_cursor = 46
    for s in steps_shared:
        ax.text(24.5, y_cursor, s, fontsize=10, va="top", ha="left")
        y_cursor -= 7.4

    # ----- Per-checkpoint loop (column 3) -----
    add_box(54, 8, 24, 44, c_loop, edge_loop, 2.0, dashed=True)
    ax.text(66, 50, "Per-checkpoint loop", **title_kw)
    ax.text(66, 46.5, "(runs N times,  N = 1 or 2)", ha="center", fontsize=10, fontstyle="italic")
    steps_loop = [
        "1.  Load checkpoint (state_dict)",
        "2.  Batched inference\n     (DataLoader, batch=256)",
        "3.  Inverse-scale predictions\n     (back to HL01 units)",
        "4.  Compute metrics per group\n     overall / per-horizon / per-segment\n     + persistence baseline",
        "5.  Save CSVs, NPZ, figures",
    ]
    y_cursor = 42
    for s in steps_loop:
        ax.text(55.5, y_cursor, s, fontsize=10, va="top", ha="left")
        y_cursor -= 6.8

    # ----- Outputs (column 4) -----
    add_box(83, 14, 15, 38, c_output, edge_normal, 1.5)
    ax.text(90.5, 50, "Outputs", **title_kw)
    ax.text(83.8, 46, "runs/<setting>/eval_dry/", fontsize=11, fontweight="bold", va="top")
    ax.text(84.5, 42, "├── best_mse/", fontsize=10, va="top", family="monospace")
    ax.text(84.5, 39.5, "│    ├── dry_metrics.csv", fontsize=9.5, va="top", family="monospace")
    ax.text(84.5, 37.5, "│    ├── per_horizon.csv", fontsize=9.5, va="top", family="monospace")
    ax.text(84.5, 35.5, "│    ├── per_segment.csv", fontsize=9.5, va="top", family="monospace")
    ax.text(84.5, 33.5, "│    ├── predictions.npz", fontsize=9.5, va="top", family="monospace")
    ax.text(84.5, 31.5, "│    └── figures/  (PNG)", fontsize=9.5, va="top", family="monospace")
    ax.text(84.5, 28, "└── best_corr/ (same)", fontsize=10, va="top", family="monospace")
    ax.text(84.5, 22, "+ eval_dry.log\n   (full run log)", fontsize=10, va="top")

    # ----- Arrows between columns -----
    add_arrow(18.5, 33, 22.5, 33)
    add_arrow(49.5, 30, 53.5, 30)
    add_arrow(78.5, 30, 82.5, 30)

    # ----- Title -----
    fig.suptitle("eval_dry.py  —  dry-segment inference evaluation pipeline",
                 fontsize=17, fontweight="bold", y=0.97)

    # ----- Legend strip at bottom -----
    legend_handles = [
        mpatches.Patch(facecolor=c_input, edgecolor=edge_normal, label="Inputs"),
        mpatches.Patch(facecolor=c_shared, edgecolor=edge_normal, label="Shared (run once)"),
        mpatches.Patch(facecolor=c_loop, edgecolor=edge_loop, linestyle="--", label="Per-checkpoint loop"),
        mpatches.Patch(facecolor=c_output, edgecolor=edge_normal, label="Outputs"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=11,
              frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# =========================================================================== #
#  Figure 2: window slicing timeline                                          #
# =========================================================================== #

def draw_window_timeline_figure(out_path: Path) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 9), dpi=180,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.2]},
        sharex=True,
    )
    ax_top, ax_mid, ax_bot = axes

    # X axis range in msr (minutes since SegmentEnd)
    x_min, x_max = -30, 240
    earliest_anchor = POST_WINDOW + 1 + SEQ_LEN  # = 121 for sl=60

    # Colors
    c_rain = "#F4B6B6"        # pink
    c_postwin = "#FFC58A"     # orange
    c_dry = "#BFD9F2"         # light blue
    c_input = "#7AB8E0"       # blue
    c_target = "#8BC78B"      # green
    c_anchor = "#222222"
    c_legal = "#1E8E3E"       # green outline
    c_illegal = "#C82626"     # red outline

    # ----------- helper: draw color bands -----------
    def draw_bands(ax):
        ax.axhspan(0, 1, xmin=(x_min - x_min) / (x_max - x_min),
                   xmax=(0 - x_min) / (x_max - x_min), facecolor=c_rain, alpha=0.55)
        ax.axhspan(0, 1, xmin=(0 - x_min) / (x_max - x_min),
                   xmax=(POST_WINDOW - x_min) / (x_max - x_min), facecolor=c_postwin, alpha=0.55)
        ax.axhspan(0, 1, xmin=(POST_WINDOW - x_min) / (x_max - x_min),
                   xmax=1, facecolor=c_dry, alpha=0.35)

    # ============== Top subplot: overview ==============
    ax_top.set_xlim(x_min, x_max)
    ax_top.set_ylim(0, 1)
    draw_bands(ax_top)

    # vertical dividers at WinEnd, dry start (60→61), earliest anchor, post_rain boundary
    # Labels placed BELOW the band so they don't collide with the subplot title.
    for x, label in [
        (0, "SegEnd\n(msr=0)"),
        (POST_WINDOW, "WinEnd\n(msr=60)"),
        (earliest_anchor, f"earliest legal anchor\n(msr={earliest_anchor})"),
        (POST_RAIN_MAX_MSR, f"post_rain boundary\n(msr=180)"),
    ]:
        ax_top.axvline(x, color="black", linestyle=":", linewidth=1)
        ax_top.text(x, -0.08, label, ha="center", va="top", fontsize=10)

    # band labels (centred vertically in the band)
    ax_top.text((x_min + 0) / 2, 0.65, "Rain segment\n(SegStart … SegEnd)",
                ha="center", va="center", fontsize=11, fontweight="bold")
    ax_top.text(POST_WINDOW / 2, 0.65, "Post-window\n(training buffer\n+60 min)",
                ha="center", va="center", fontsize=10, fontweight="bold")
    ax_top.text((POST_WINDOW + POST_RAIN_MAX_MSR) / 2, 0.55,
                "DRY BLOCK (in_rain_window=False)\n──── post_rain group ────",
                ha="center", va="center", fontsize=10)
    ax_top.text((POST_RAIN_MAX_MSR + x_max) / 2, 0.55,
                "DRY BLOCK (continued)\n──── pure_dry group ────",
                ha="center", va="center", fontsize=10)

    ax_top.set_yticks([])
    ax_top.set_title("(a) Timeline — color bands relative to SegmentEnd",
                     fontsize=12, loc="left", pad=8)

    # ============== Middle subplot: ILLEGAL anchor msr=61 ==============
    ax_mid.set_xlim(x_min, x_max)
    ax_mid.set_ylim(0, 1)
    draw_bands(ax_mid)

    illegal_anchor = POST_WINDOW + 1  # msr=61
    illegal_input_start = illegal_anchor - SEQ_LEN  # = 1
    illegal_input_end = illegal_anchor - 1          # = 60
    illegal_target_end = illegal_anchor + PRED_LEN - 1  # = 75

    # input band (red dashed because it spans rain/post-window)
    ax_mid.add_patch(mpatches.Rectangle(
        (illegal_input_start, 0.45), illegal_input_end - illegal_input_start + 1, 0.25,
        facecolor=c_input, alpha=0.45,
        edgecolor=c_illegal, linewidth=2.2, linestyle="--"))
    ax_mid.text((illegal_input_start + illegal_input_end) / 2, 0.575,
                f"input ({SEQ_LEN} min)\nspans rain + post-window  ✗",
                ha="center", va="center", fontsize=10.5, fontweight="bold", color=c_illegal)

    # target band
    ax_mid.add_patch(mpatches.Rectangle(
        (illegal_anchor, 0.45), PRED_LEN, 0.25,
        facecolor=c_target, alpha=0.45,
        edgecolor=c_illegal, linewidth=2.2, linestyle="--"))
    ax_mid.text((illegal_anchor + illegal_target_end) / 2, 0.575,
                f"target\n({PRED_LEN} min)", ha="center", va="center", fontsize=10)

    # anchor marker
    ax_mid.axvline(illegal_anchor, color=c_anchor, linewidth=1.8, ymin=0.4, ymax=0.78)
    ax_mid.annotate(f"anchor\n(msr={illegal_anchor})",
                    xy=(illegal_anchor, 0.78), xytext=(illegal_anchor + 10, 0.92),
                    fontsize=10, ha="left",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # rule violation banner
    ax_mid.text(x_max - 5, 0.20,
                "VIOLATION:\ninput must lie entirely\ninside the dry block.\nHere it crosses into the\nrain segment + post-window.",
                ha="right", va="center", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=c_illegal, lw=1.5))

    ax_mid.set_yticks([])
    ax_mid.set_title(f"(b) ILLEGAL candidate window — anchor at msr={illegal_anchor}",
                     fontsize=12, loc="left", color=c_illegal)

    # ============== Bottom subplot: LEGAL anchor msr=121 ==============
    ax_bot.set_xlim(x_min, x_max)
    ax_bot.set_ylim(0, 1)
    draw_bands(ax_bot)

    legal_anchor = earliest_anchor  # = 121
    legal_input_start = legal_anchor - SEQ_LEN   # = 61
    legal_input_end = legal_anchor - 1           # = 120
    legal_target_end = legal_anchor + PRED_LEN - 1  # = 135

    ax_bot.add_patch(mpatches.Rectangle(
        (legal_input_start, 0.45), legal_input_end - legal_input_start + 1, 0.25,
        facecolor=c_input, alpha=0.55,
        edgecolor=c_legal, linewidth=2.2))
    ax_bot.text((legal_input_start + legal_input_end) / 2, 0.575,
                f"input ({SEQ_LEN} min)\nall in dry block  ✓",
                ha="center", va="center", fontsize=10.5, fontweight="bold", color=c_legal)

    ax_bot.add_patch(mpatches.Rectangle(
        (legal_anchor, 0.45), PRED_LEN, 0.25,
        facecolor=c_target, alpha=0.55,
        edgecolor=c_legal, linewidth=2.2))
    ax_bot.text((legal_anchor + legal_target_end) / 2, 0.575,
                f"target\n({PRED_LEN} min)", ha="center", va="center", fontsize=10)

    ax_bot.axvline(legal_anchor, color=c_anchor, linewidth=1.8, ymin=0.4, ymax=0.78)
    ax_bot.annotate(f"anchor\n(msr={legal_anchor})",
                    xy=(legal_anchor, 0.78), xytext=(legal_anchor + 10, 0.92),
                    fontsize=10, ha="left",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    ax_bot.text(x_max - 5, 0.20,
                f"EARLIEST LEGAL ANCHOR:\n  msr = WinEnd + 1 + seq_len\n      = {POST_WINDOW} + 1 + {SEQ_LEN}\n      = {earliest_anchor}",
                ha="right", va="center", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=c_legal, lw=1.5))

    ax_bot.set_yticks([])
    ax_bot.set_xlabel("minutes since SegmentEnd  (msr)", fontsize=11)
    ax_bot.set_title(f"(c) LEGAL candidate window — earliest legal anchor at msr={legal_anchor}",
                     fontsize=12, loc="left", color=c_legal)

    # ----- shared legend -----
    legend_handles = [
        mpatches.Patch(facecolor=c_rain, alpha=0.6, label="Rain segment"),
        mpatches.Patch(facecolor=c_postwin, alpha=0.6, label="Post-window (training buffer)"),
        mpatches.Patch(facecolor=c_dry, alpha=0.5, label="Dry block (in_rain_window=False)"),
        mpatches.Patch(facecolor=c_input, alpha=0.6, label=f"Input ({SEQ_LEN} min)"),
        mpatches.Patch(facecolor=c_target, alpha=0.6, label=f"Target ({PRED_LEN} min)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"eval_dry.py  —  sliding window placement (seq_len={SEQ_LEN}, pred_len={PRED_LEN}, post_window={POST_WINDOW})",
        fontsize=15, fontweight="bold", y=0.985,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# =========================================================================== #
#  Main                                                                        #
# =========================================================================== #

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_pipeline_figure(OUT_DIR / "eval_dry_pipeline.png")
    draw_window_timeline_figure(OUT_DIR / "eval_dry_window_timeline.png")
    print(f"Done. Output dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
