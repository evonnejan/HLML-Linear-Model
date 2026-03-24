import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def _add_nodes(ax, x, y_values, color, labels=None, radius=0.1):
    coords = []
    for idx, y in enumerate(y_values):
        ax.add_patch(plt.Circle((x, y), radius, color=color, ec="black", zorder=3))
        if labels is not None:
            ax.text(x - 0.32, y, labels[idx], ha="right", va="center", fontsize=9)
        coords.append((x, y))
    return coords


def _connect_all(ax, src_nodes, dst_nodes, style="--", color="gray", alpha=0.45, lw=0.8):
    for x1, y1 in src_nodes:
        for x2, y2 in dst_nodes:
            ax.annotate(
                "",
                xy=(x2 - 0.12, y2),
                xytext=(x1 + 0.12, y1),
                arrowprops=dict(arrowstyle="->", linestyle=style, color=color, lw=lw, alpha=alpha),
            )


def _block(ax, x, y, w, h, title, subtitle=None, edge="black", face="#f7f7f7", title_size=11):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size)
    if subtitle is not None:
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=9.5, color="dimgray")


def _arrow(ax, x1, y1, x2, y2, text=None, color="black", lw=1.5, rad=0.0, text_offset=(0, 0)):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=4,
    )
    if text is not None:
        tx = (x1 + x2) / 2 + text_offset[0]
        ty = (y1 + y2) / 2 + text_offset[1]
        ax.text(tx, ty, text, fontsize=9.5, color=color, ha="center", va="center")


def draw_dlinear_block_flow(seq_len_demo=60, pred_len_demo=15, channels_demo=1):
    fig, ax = plt.subplots(figsize=(15.5, 6.8))

    _block(
        ax,
        x=0.5,
        y=2.35,
        w=2.3,
        h=1.6,
        title="Input x",
        subtitle=f"shape: (B, {seq_len_demo}, {channels_demo})",
        edge="#1f77b4",
        face="#e8f4fb",
    )

    _block(
        ax,
        x=3.5,
        y=2.2,
        w=2.7,
        h=1.9,
        title="Series Decomposition",
        subtitle="moving_avg + residual",
        edge="#444",
        face="#f2f2f2",
    )

    _block(
        ax,
        x=7.2,
        y=3.55,
        w=3.0,
        h=1.7,
        title="Linear_Seasonal",
        subtitle=f"Ws: {pred_len_demo} × {seq_len_demo}",
        edge="#2e7d32",
        face="#eaf7ea",
    )
    _block(
        ax,
        x=7.2,
        y=0.95,
        w=3.0,
        h=1.7,
        title="Linear_Trend",
        subtitle=f"Wt: {pred_len_demo} × {seq_len_demo}",
        edge="#b26a00",
        face="#fff3e0",
    )

    _block(
        ax,
        x=11.2,
        y=2.3,
        w=1.35,
        h=1.7,
        title="+",
        subtitle="element-wise",
        edge="#555",
        face="#fafafa",
        title_size=18,
    )

    _block(
        ax,
        x=13.2,
        y=2.35,
        w=2.4,
        h=1.6,
        title="Output ŷ",
        subtitle=f"shape: (B, {pred_len_demo}, {channels_demo})",
        edge="#d62728",
        face="#fdecec",
    )

    _arrow(ax, 2.8, 3.15, 3.5, 3.15, text="x", color="#1f77b4", text_offset=(0, 0.18))

    split_x, split_y = 6.4, 3.15
    ax.plot(split_x, split_y, marker="o", markersize=5.5, color="black", zorder=5)
    _arrow(ax, 6.2, 3.15, split_x, split_y, color="black", lw=1.6)

    _arrow(
        ax,
        split_x,
        split_y,
        7.2,
        4.35,
        text=f"seasonal_init (B, {seq_len_demo}, {channels_demo})",
        color="#2e7d32",
        rad=-0.12,
        text_offset=(0.2, 0.35),
    )
    _arrow(
        ax,
        split_x,
        split_y,
        7.2,
        1.85,
        text=f"trend_init (B, {seq_len_demo}, {channels_demo})",
        color="#b26a00",
        rad=0.12,
        text_offset=(0.2, -0.35),
    )

    _arrow(ax, 10.2, 4.35, 11.2, 3.45, text="seasonal_out", color="#2e7d32", text_offset=(0.0, 0.25))
    _arrow(ax, 10.2, 1.85, 11.2, 2.85, text="trend_out", color="#b26a00", text_offset=(0.0, -0.25))
    _arrow(ax, 12.55, 3.15, 13.2, 3.15, text="ŷ = seasonal + trend", color="#444", text_offset=(0, 0.22))


    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.4, 6.2)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_dlinear_strict(seq_len_demo=6, pred_len_demo=4):
    fig, ax = plt.subplots(figsize=(16, 9))

    input_y = [9.5 - i * 1.0 for i in range(seq_len_demo)]
    input_labels = [f"x(t-{seq_len_demo-1-i})" for i in range(seq_len_demo - 1)] + ["x(t)"]
    input_nodes = _add_nodes(ax, x=0.8, y_values=input_y, color="#9bd3e8", labels=input_labels)

    ax.add_patch(patches.Rectangle((2.0, 3.6), 1.9, 4.0, fill=False, ec="black", lw=1.2))
    ax.text(2.95, 7.95, "Series", ha="center", va="bottom", fontsize=12)
    ax.text(2.95, 7.48, "Decomposition", ha="center", va="bottom", fontsize=12)
    ax.text(2.95, 6.78, "(Moving Avg)", ha="center", va="bottom", fontsize=10.5, color="dimgray")

    for _, y in input_nodes:
        ax.annotate(
            "",
            xy=(2.0, y),
            xytext=(0.95, y),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.9, alpha=0.75),
        )

    # Clear trunk flow: Input -> Decomposition -> Splitter
    decomp_out = (3.95, 5.6)
    splitter = (4.35, 5.6)
    ax.annotate(
        "",
        xy=decomp_out,
        xytext=(3.9, 5.6),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.6),
        zorder=4,
    )
    ax.plot(splitter[0], splitter[1], marker="o", markersize=5, color="black", zorder=5)
    ax.text(4.02, 5.9, "decomp output", fontsize=9.2, color="black", ha="left")

    pred_base = [8.2 - i * 1.7 for i in range(pred_len_demo)]
    seasonal_out_y = [v + 0.30 for v in pred_base]
    trend_out_y = [v - 0.30 for v in pred_base]

    seasonal_in_y = [10.0 - i * 0.75 for i in range(seq_len_demo)]
    trend_in_y = [4.35 - i * 0.75 for i in range(seq_len_demo)]

    seasonal_in_nodes = _add_nodes(ax, x=4.9, y_values=seasonal_in_y, color="#d6f5d6")
    trend_in_nodes = _add_nodes(ax, x=4.9, y_values=trend_in_y, color="#f8e0b6")

    seasonal_out_nodes = _add_nodes(ax, x=7.1, y_values=seasonal_out_y, color="#8ddf8d")
    trend_out_nodes = _add_nodes(ax, x=7.1, y_values=trend_out_y, color="#f6be74")

    ax.add_patch(patches.Rectangle((4.2, 5.55), 3.6, 4.95, fill=False, ec="#2e7d32", lw=1.3))
    ax.text(6.0, 10.72, "Seasonal branch", ha="center", fontsize=12, color="#2e7d32")
    ax.text(6.0, 10.27, "Linear_Seasonal", ha="center", fontsize=10.5)
    ax.text(6.0, 9.92, f"Ws: {pred_len_demo} × {seq_len_demo}", ha="center", fontsize=9.5, color="dimgray")

    ax.add_patch(patches.Rectangle((4.2, -0.15), 3.6, 4.95, fill=False, ec="#b26a00", lw=1.3))
    ax.text(6.0, 5.02, "Trend branch", ha="center", fontsize=12, color="#b26a00")
    ax.text(6.0, 4.57, "Linear_Trend", ha="center", fontsize=10.5)
    ax.text(6.0, 4.22, f"Wt: {pred_len_demo} × {seq_len_demo}", ha="center", fontsize=9.5, color="dimgray")

    ax.annotate(
        "",
        xy=(4.7, seasonal_in_y[0]),
        xytext=splitter,
        arrowprops=dict(arrowstyle="->", lw=1.35, color="#2e7d32", connectionstyle="arc3,rad=-0.18"),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(4.7, trend_in_y[0]),
        xytext=splitter,
        arrowprops=dict(arrowstyle="->", lw=1.35, color="#b26a00", connectionstyle="arc3,rad=0.20"),
        zorder=4,
    )
    ax.text(4.38, 6.35, "seasonal_init", fontsize=9.6, color="#2e7d32", ha="left")
    ax.text(4.38, 4.90, "trend_init", fontsize=9.6, color="#b26a00", ha="left")

    _connect_all(ax, seasonal_in_nodes, seasonal_out_nodes, style="--", color="#2e7d32", alpha=0.35, lw=0.8)
    _connect_all(ax, trend_in_nodes, trend_out_nodes, style="--", color="#b26a00", alpha=0.35, lw=0.8)

    sum_nodes = _add_nodes(ax, x=9.2, y_values=pred_base, color="#f4a7a7")
    output_labels = [f"ŷ(t+{i+1})" for i in range(pred_len_demo)]
    for idx, (x, y) in enumerate(sum_nodes):
        ax.text(x + 0.4, y, output_labels[idx], ha="left", va="center", fontsize=11)

    for i in range(pred_len_demo):
        ax.annotate(
            "",
            xy=(9.05, pred_base[i]),
            xytext=(7.25, seasonal_out_y[i]),
            arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.0, connectionstyle="arc3,rad=-0.08"),
        )
        ax.annotate(
            "",
            xy=(9.05, pred_base[i]),
            xytext=(7.25, trend_out_y[i]),
            arrowprops=dict(arrowstyle="->", color="#b26a00", lw=1.0, connectionstyle="arc3,rad=0.08"),
        )
        ax.text(8.78, pred_base[i] + 0.03, "+", fontsize=12, ha="center", va="center", color="dimgray")

    ax.text(9.2, 8.95, "Element-wise Sum", ha="center", fontsize=12)
    ax.text(10.25, 8.45, "Final Output\n(B, pred_len, C)", ha="left", fontsize=11)

    ax.add_patch(patches.Rectangle((0.2, 3.2), 1.2, 6.8, fill=False, ec="black", lw=1.0))
    ax.text(0.8, 10.32, "Input", ha="center", fontsize=12)
    ax.text(0.8, 9.95, "(time steps)", ha="center", fontsize=10.5)

    ax.text(6.0, -0.85, "Strict DLinear: no hidden layer; two linear branches + element-wise add", ha="center", fontsize=10.5, color="dimgray")

    ax.set_xlim(-0.3, 11.8)
    ax.set_ylim(-1.2, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DLinear architecture")
    parser.add_argument("--style", type=str, default="block", choices=["block", "neuron"], help="diagram style")
    parser.add_argument("--seq_len_demo", type=int, default=6)
    parser.add_argument("--pred_len_demo", type=int, default=4)
    parser.add_argument("--channels_demo", type=int, default=1)
    args = parser.parse_args()

    if args.style == "neuron":
        draw_dlinear_strict(seq_len_demo=args.seq_len_demo, pred_len_demo=args.pred_len_demo)
    else:
        draw_dlinear_block_flow(
            seq_len_demo=args.seq_len_demo,
            pred_len_demo=args.pred_len_demo,
            channels_demo=args.channels_demo,
        )