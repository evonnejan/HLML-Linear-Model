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


def draw_dlinearmix_block_flow(seq_len_demo=60, pred_len_demo=15, branch_in=3, exog_in=1):
    """DLinearMix: EARLY fusion — concat (input_cols + exog_cols) along channels,
    feed the mixed multivariate series through a single DLinear, target forecast.
    """
    fig, ax = plt.subplots(figsize=(16, 6.8))

    _block(ax, 0.3, 4.3, 2.6, 1.5,
           title=f"Input cols × {branch_in}",
           subtitle=f"shape: (B, {seq_len_demo}, {branch_in})",
           edge="#1f77b4", face="#e8f4fb")
    _block(ax, 0.3, 1.6, 2.6, 1.5,
           title=f"Exog cols × {exog_in}",
           subtitle=f"shape: (B, {seq_len_demo}, {exog_in})",
           edge="#b26a00", face="#fff3e0")

    total = branch_in + exog_in
    _block(ax, 3.6, 2.95, 2.6, 1.8,
           title="Early Fusion",
           subtitle=f"concat → (B, {seq_len_demo}, {total})",
           edge="#444", face="#f2f2f2")

    _block(ax, 7.0, 2.85, 3.4, 2.0,
           title="DLinear",
           subtitle="decomp + Linear_S + Linear_T",
           edge="#2e7d32", face="#eaf7ea")

    _block(ax, 11.2, 2.85, 2.8, 2.0,
           title="Output ŷ",
           subtitle=f"shape: (B, {pred_len_demo}, 1)",
           edge="#d62728", face="#fdecec")

    _arrow(ax, 2.9, 5.05, 3.6, 4.20, color="#1f77b4")
    _arrow(ax, 2.9, 2.35, 3.6, 3.55, color="#b26a00")
    _arrow(ax, 6.2, 3.85, 7.0, 3.85, text="mixed multivariate", color="#444", text_offset=(0, 0.25))
    _arrow(ax, 10.4, 3.85, 11.2, 3.85, text="ŷ", color="#444", text_offset=(0, 0.25))

    ax.text(7.0, 0.55, "DLinearMix: early concat (HL + rain/gate) → single DLinear",
            ha="center", fontsize=10.5, color="dimgray")

    ax.set_xlim(0.0, 14.5)
    ax.set_ylim(0.0, 6.5)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_dlinearmix2_block_flow(seq_len_demo=60, pred_len_demo=15, branch_in=3, exog_in=2,
                                exog_emb_dim=16, fusion_hidden_dim=32, flatten_fusion=False):
    """DLinearMix2: per-channel DLinear branches + GRU exogenous encoder + fusion MLP.

    Faithful to models/DLinearMix2.py:
      input_cols[i] -> DLinearBranch_i -> [B,P,1]
      concat across i -> branch_preds [B,P,K]
      exog_cols  -> GRU encoder -> context [B, E]
      HorizonWiseFusion(False) or FlattenFusion(True) -> ŷ [B, P, 1]
    """
    fig, ax = plt.subplots(figsize=(17, 8.6))

    # ----- Left: input channels (each gets its own DLinear branch) -----
    branch_y_top = 7.5
    branch_gap = 1.05
    branch_box_h = 0.78
    branch_xs = (0.4, 3.4)

    for i in range(branch_in):
        y_in = branch_y_top - i * branch_gap
        _block(ax, branch_xs[0], y_in, 1.7, branch_box_h,
               title=f"x_input[{i}]",
               subtitle=f"(B,{seq_len_demo},1)",
               edge="#1f77b4", face="#e8f4fb", title_size=10)

        _block(ax, branch_xs[1], y_in - 0.05, 2.6, branch_box_h + 0.1,
               title=f"DLinearBranch_{i}",
               subtitle="decomp + Lin_S + Lin_T",
               edge="#2e7d32", face="#eaf7ea", title_size=10)

        _arrow(ax, branch_xs[0] + 1.7, y_in + branch_box_h / 2,
               branch_xs[1], y_in + branch_box_h / 2,
               color="#2e7d32", lw=1.2)

    # ----- Left bottom: exogenous channels + GRU -----
    exog_y = branch_y_top - branch_in * branch_gap - 0.6
    _block(ax, branch_xs[0], exog_y - 0.4, 1.7, 1.1,
           title=f"x_exog × {exog_in}",
           subtitle=f"(B,{seq_len_demo},{exog_in})",
           edge="#b26a00", face="#fff3e0", title_size=10)

    _block(ax, branch_xs[1], exog_y - 0.4, 2.6, 1.1,
           title="GRU Encoder",
           subtitle=f"emb_dim={exog_emb_dim}",
           edge="#b26a00", face="#fff3e0", title_size=10)
    _arrow(ax, branch_xs[0] + 1.7, exog_y + 0.15,
           branch_xs[1], exog_y + 0.15, color="#b26a00", lw=1.2)

    # ----- Middle: concat point for branches -----
    concat_x = 7.0
    concat_y_top = branch_y_top + branch_box_h / 2
    concat_y_bot = branch_y_top + branch_box_h / 2 - (branch_in - 1) * branch_gap
    branch_concat_y = (concat_y_top + concat_y_bot) / 2

    _block(ax, concat_x, branch_concat_y - 0.55, 1.9, 1.1,
           title="concat",
           subtitle=f"branch_preds\n(B,{pred_len_demo},{branch_in})",
           edge="#444", face="#fafafa", title_size=11)

    for i in range(branch_in):
        y_out = branch_y_top + branch_box_h / 2 - i * branch_gap
        _arrow(ax, branch_xs[1] + 2.6, y_out, concat_x, branch_concat_y,
               color="#2e7d32", lw=1.0, rad=0.0)

    # Exog context arrow up to fusion
    context_x = concat_x + 1.0
    _arrow(ax, branch_xs[1] + 2.6, exog_y + 0.15,
           context_x, exog_y + 0.15,
           text=f"context (B,{exog_emb_dim})", color="#b26a00",
           text_offset=(0.4, 0.25), lw=1.2)

    # ----- Right: Fusion MLP -----
    fusion_x = 10.4
    fusion_y = branch_concat_y - 0.7
    fusion_title = "FlattenFusion" if flatten_fusion else "HorizonWiseFusion"
    if flatten_fusion:
        fusion_sub = f"in: P·K+E = {pred_len_demo*branch_in + exog_emb_dim}\nhidden={max(fusion_hidden_dim, 64)}"
    else:
        fusion_sub = f"per-horizon MLP\nin=K+E={branch_in + exog_emb_dim}, hid={fusion_hidden_dim}"

    _block(ax, fusion_x, fusion_y, 3.2, 2.0,
           title=fusion_title, subtitle=fusion_sub,
           edge="#6a1b9a", face="#f3e5f5", title_size=11)

    _arrow(ax, concat_x + 1.9, branch_concat_y, fusion_x, fusion_y + 1.4,
           color="#2e7d32", lw=1.3)
    _arrow(ax, context_x, exog_y + 0.15, fusion_x, fusion_y + 0.4,
           color="#b26a00", lw=1.3, rad=-0.18)

    # ----- Output -----
    _block(ax, 14.1, fusion_y + 0.2, 2.4, 1.6,
           title="Output ŷ",
           subtitle=f"(B,{pred_len_demo},1)",
           edge="#d62728", face="#fdecec")
    _arrow(ax, fusion_x + 3.2, fusion_y + 1.0, 14.1, fusion_y + 1.0,
           text="ŷ", color="#444", text_offset=(0, 0.22), lw=1.4)

    fusion_label = "flatten" if flatten_fusion else "horizon-wise"
    ax.text(8.5, 0.5,
            f"DLinearMix2: per-channel DLinear branches + GRU exog encoder + {fusion_label} fusion MLP",
            ha="center", fontsize=10.5, color="dimgray")

    ax.set_xlim(0.0, 17.0)
    ax.set_ylim(0.0, 9.0)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_for_model(model_name: str, seq_len_demo, pred_len_demo, channels_demo,
                   branch_in, exog_in, style, flatten_fusion):
    key = model_name.lower()
    if key in ("dlinear", "linear", "nlinear"):
        if style == "neuron":
            draw_dlinear_strict(seq_len_demo=seq_len_demo, pred_len_demo=pred_len_demo)
        else:
            draw_dlinear_block_flow(
                seq_len_demo=seq_len_demo,
                pred_len_demo=pred_len_demo,
                channels_demo=channels_demo,
            )
    elif key == "dlinearmix":
        draw_dlinearmix_block_flow(
            seq_len_demo=seq_len_demo,
            pred_len_demo=pred_len_demo,
            branch_in=branch_in,
            exog_in=exog_in,
        )
    elif key == "dlinearmix2":
        draw_dlinearmix2_block_flow(
            seq_len_demo=seq_len_demo,
            pred_len_demo=pred_len_demo,
            branch_in=branch_in,
            exog_in=exog_in,
            flatten_fusion=flatten_fusion,
        )
    else:
        raise ValueError(
            f"Unknown --model '{model_name}'. "
            f"Supported: DLinear, Linear, NLinear, DLinearMix, DLinearMix2."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model architecture")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name: DLinear / Linear / NLinear / DLinearMix / DLinearMix2. "
                             "If given, overrides --style for DLinearMix*.")
    parser.add_argument("--style", type=str, default="block", choices=["block", "neuron"],
                        help="Diagram style for DLinear-family. Ignored for DLinearMix*.")
    parser.add_argument("--seq_len_demo", type=int, default=60)
    parser.add_argument("--pred_len_demo", type=int, default=15)
    parser.add_argument("--channels_demo", type=int, default=1,
                        help="Channels demo, used by single-channel DLinear block diagram.")
    parser.add_argument("--branch_in", type=int, default=3,
                        help="Number of input channels for DLinearMix / DLinearMix2 diagrams.")
    parser.add_argument("--exog_in", type=int, default=2,
                        help="Number of exogenous channels for DLinearMix / DLinearMix2 diagrams.")
    parser.add_argument("--flatten_fusion", action="store_true",
                        help="DLinearMix2 only: draw with FlattenFusion instead of HorizonWiseFusion.")
    args = parser.parse_args()

    if args.model is not None:
        draw_for_model(
            model_name=args.model,
            seq_len_demo=args.seq_len_demo,
            pred_len_demo=args.pred_len_demo,
            channels_demo=args.channels_demo,
            branch_in=args.branch_in,
            exog_in=args.exog_in,
            style=args.style,
            flatten_fusion=args.flatten_fusion,
        )
    else:
        # Backwards compatible: no --model → original DLinear behavior.
        if args.style == "neuron":
            draw_dlinear_strict(seq_len_demo=args.seq_len_demo, pred_len_demo=args.pred_len_demo)
        else:
            draw_dlinear_block_flow(
                seq_len_demo=args.seq_len_demo,
                pred_len_demo=args.pred_len_demo,
                channels_demo=args.channels_demo,
            )