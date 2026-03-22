"""Generate all tutorial diagrams as PNGs — plain labels only, no math formulas."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 150
BG = "#1a1a2e"

# Colors
C_INPUT  = "#E07C4F"
C_ENC    = "#4A90D9"
C_LATENT = "#7B68EE"
C_DEC    = "#50C878"
C_PURPLE = "#6A5ACD"
C_RED    = "#FF6B6B"
C_GRAY   = "#888888"
C_ARROW  = "#cccccc"
C_TEXT   = "#aaaaaa"
C_LOSS_R = "#D4534B"
C_LOSS_K = "#D4A34B"


def rounded_box(ax, xy, w, h, text, color=C_ENC, fontsize=11, text_color="white"):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.15", facecolor=color,
                         edgecolor="white", linewidth=1.5)
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold")


def arrow(ax, start, end, color="#555555", lw=2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=color, lw=lw))


def setup_ax(ax):
    ax.set_facecolor(BG)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# 1. VAE Big Picture — only component names, no formulas
# ─────────────────────────────────────────────────────────────────────────────
def make_overview():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)
    setup_ax(ax)
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-1, 3.8)

    rounded_box(ax, (0, 0.8), 2, 1.8,    "Input\nImage",  color=C_INPUT, fontsize=13)
    rounded_box(ax, (3.2, 0.8), 2.2, 1.8, "Encoder",      color=C_ENC,   fontsize=13)
    rounded_box(ax, (6.6, 0.8), 1.8, 1.8, "Latent\nSpace", color=C_LATENT, fontsize=13)
    rounded_box(ax, (9.6, 0.8), 2.2, 1.8, "Decoder",      color=C_DEC,   fontsize=13)

    arrow(ax, (2.1, 1.7), (3.1, 1.7), color=C_ARROW, lw=2.5)
    arrow(ax, (5.5, 1.7), (6.5, 1.7), color=C_ARROW, lw=2.5)
    arrow(ax, (8.5, 1.7), (9.5, 1.7), color=C_ARROW, lw=2.5)

    ax.text(2.6, 2.3, "compress", ha="center", fontsize=10, color=C_TEXT, style="italic")
    ax.text(6.0, 2.3, "sample",   ha="center", fontsize=10, color=C_TEXT, style="italic")
    ax.text(9.0, 2.3, "reconstruct", ha="center", fontsize=10, color=C_TEXT, style="italic")

    # Generation path
    ax.annotate("", xy=(6.6, 0.8), xytext=(7.5, -0.2),
                arrowprops=dict(arrowstyle="->", color="#FFD700", lw=2, linestyle="--"))
    ax.text(7.5, -0.65, "Sample from prior\n(generation mode)", ha="center", fontsize=9,
            color="#FFD700", style="italic")

    fig.savefig(f"{DIR}/vae_overview.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. AE vs VAE — structural diagram + latent space scatter
# ─────────────────────────────────────────────────────────────────────────────
def make_ae_vs_vae():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)

    for idx, (ax, title) in enumerate(zip(axes, ["Autoencoder (AE)", "Variational Autoencoder (VAE)"])):
        setup_ax(ax)
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-1.2, 6.5)
        ax.set_title(title, fontsize=14, color="white", fontweight="bold", pad=10)

        if idx == 0:  # AE
            rounded_box(ax, (0.5, 5), 2, 0.9, "Input",   color=C_INPUT)
            rounded_box(ax, (0.5, 3.5), 2, 0.9, "Encoder", color=C_ENC)
            rounded_box(ax, (0.5, 2.0), 2, 0.9, "z (point)", color=C_LATENT)
            rounded_box(ax, (0.5, 0.5), 2, 0.9, "Decoder", color=C_DEC)
            rounded_box(ax, (0.5, -0.8), 2, 0.9, "Output",  color=C_INPUT)
            for y_top, y_bot in [(5.0, 4.5), (3.5, 3.0), (2.0, 1.5), (0.5, 0.1)]:
                arrow(ax, (1.5, y_top), (1.5, y_bot), color=C_ARROW)

            # Sparse latent space with holes
            np.random.seed(42)
            pts = np.random.randn(40, 2) * 0.8
            pts = pts[np.abs(pts).max(axis=1) < 2]
            mask = ~((pts[:, 0] > -0.3) & (pts[:, 0] < 0.5) &
                      (pts[:, 1] > -0.5) & (pts[:, 1] < 0.3))
            pts = pts[mask]
            ax.scatter(pts[:, 0] + 4.5, pts[:, 1] + 3.0, c=C_INPUT, s=30, alpha=0.8, zorder=5)
            ax.text(4.5, 1.0, "Sparse,\nholes!", ha="center", fontsize=10, color=C_INPUT, style="italic")

        else:  # VAE
            rounded_box(ax, (0.2, 5), 2, 0.9, "Input",   color=C_INPUT)
            rounded_box(ax, (0.2, 3.5), 2, 0.9, "Encoder", color=C_ENC)
            rounded_box(ax, (-0.3, 2.0), 1.2, 0.7, "mean",    color=C_PURPLE, fontsize=10)
            rounded_box(ax, (2.1, 2.0), 1.2, 0.7, "variance", color=C_PURPLE, fontsize=9)
            rounded_box(ax, (0.5, 0.8), 1.5, 0.7, "sample z", color=C_LATENT, fontsize=10)
            rounded_box(ax, (0.2, -0.5), 2, 0.9, "Decoder", color=C_DEC)

            arrow(ax, (1.2, 5.0), (1.2, 4.5), color=C_ARROW)
            arrow(ax, (0.7, 3.5), (0.3, 2.8), color=C_ARROW)
            arrow(ax, (1.7, 3.5), (2.5, 2.8), color=C_ARROW)
            arrow(ax, (0.3, 2.0), (0.9, 1.6), color=C_ARROW)
            arrow(ax, (2.5, 2.0), (1.6, 1.6), color=C_ARROW)
            arrow(ax, (1.25, 0.8), (1.2, 0.5), color=C_ARROW)

            # Smooth latent space
            np.random.seed(42)
            pts = np.random.randn(200, 2) * 0.7
            ax.scatter(pts[:, 0] + 4.5, pts[:, 1] + 3.0, c=C_DEC, s=8, alpha=0.4, zorder=5)
            ax.text(4.5, 1.0, "Smooth,\ncontinuous!", ha="center", fontsize=10, color=C_DEC, style="italic")

        ax.add_patch(plt.Circle((4.5, 3.0), 1.5, fill=False, edgecolor="#444444",
                                linewidth=1.5, linestyle="--"))
        ax.text(4.5, 4.8, "Latent Space", ha="center", fontsize=11, color="white", fontweight="bold")

    fig.savefig(f"{DIR}/ae_vs_vae.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reparameterization Trick — boxes + gradient flow arrows
# ─────────────────────────────────────────────────────────────────────────────
def make_reparam():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    titles = [("Without Reparameterization", C_RED),
              ("With Reparameterization", C_DEC)]

    for idx, (ax, (title, tc)) in enumerate(zip(axes, titles)):
        setup_ax(ax)
        ax.set_xlim(-0.5, 8)
        ax.set_ylim(-0.5, 5.8)
        ax.set_title(title, fontsize=13, color=tc, fontweight="bold", pad=10)

        if idx == 0:  # Without
            rounded_box(ax, (0.5, 4.0), 2.5, 1, "Encoder", color=C_ENC)
            rounded_box(ax, (0.5, 2.2), 2.5, 1, "mean, std", color=C_PURPLE)
            rounded_box(ax, (0.5, 0.2), 2.5, 1, "SAMPLE z",  color=C_RED)
            rounded_box(ax, (4.5, 0.2), 2.5, 1, "Decoder",   color=C_DEC)

            arrow(ax, (1.75, 4.0), (1.75, 3.3), color=C_ARROW)
            arrow(ax, (1.75, 2.2), (1.75, 1.3), color=C_ARROW)
            arrow(ax, (3.1, 0.7), (4.4, 0.7), color=C_ARROW)

            # X mark blocking gradients
            cx, cy = 1.75, 1.6
            ax.plot([cx-0.4, cx+0.4], [cy+0.2, cy-0.2], color=C_RED, lw=4)
            ax.plot([cx-0.4, cx+0.4], [cy-0.2, cy+0.2], color=C_RED, lw=4)
            ax.text(4.5, 1.8, "Gradients\nBLOCKED", fontsize=12, color=C_RED,
                    ha="center", fontweight="bold")

        else:  # With
            rounded_box(ax, (0.5, 4.0), 2.5, 1, "Encoder", color=C_ENC)
            rounded_box(ax, (-0.2, 2.2), 1.5, 0.9, "mean", color=C_PURPLE, fontsize=12)
            rounded_box(ax, (2.2, 2.2), 1.5, 0.9, "std",  color=C_PURPLE, fontsize=12)
            rounded_box(ax, (4.0, 4.0), 2.5, 1, "Random\nnoise", color=C_GRAY)

            rounded_box(ax, (1.0, 0.4), 3.5, 1, "Combine:\nmean + std * noise", color=C_LATENT, fontsize=10)
            rounded_box(ax, (5.5, 0.4), 2, 1, "Decoder", color=C_DEC)

            arrow(ax, (1.0, 4.0), (0.55, 3.2), color=C_ARROW)
            arrow(ax, (2.5, 4.0), (2.95, 3.2), color=C_ARROW)
            arrow(ax, (0.55, 2.2), (2.0, 1.5), color=C_DEC, lw=2.5)
            arrow(ax, (2.95, 2.2), (2.8, 1.5), color=C_DEC, lw=2.5)
            arrow(ax, (5.25, 4.0), (3.8, 1.5), color=C_GRAY)
            arrow(ax, (4.6, 0.9), (5.4, 0.9), color=C_ARROW)

            ax.text(0.0, 1.7, "Gradients\nFLOW", fontsize=10, color=C_DEC,
                    ha="center", style="italic", fontweight="bold")
            ax.text(5.25, 3.4, "no learnable\nparams", fontsize=9, color=C_GRAY,
                    ha="center", style="italic")

    fig.savefig(f"{DIR}/reparam_trick.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Architecture — layer shapes only, no formulas
# ─────────────────────────────────────────────────────────────────────────────
def make_architecture():
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    setup_ax(ax)
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.5, 6)

    y_center = 2.5

    enc_layers = [
        ("3 x 64 x 64\nInput",    0,   4.5, 1.2, C_INPUT),
        ("32 x 32 x 32", 1.8, 3.8, 1.1, C_ENC),
        ("64 x 16 x 16", 3.4, 3.2, 1.0, C_ENC),
        ("128 x 8 x 8",  5.0, 2.6, 0.9, C_ENC),
        ("256 x 4 x 4",  6.5, 2.0, 0.8, C_ENC),
    ]
    for (label, x, h, w, color) in enc_layers:
        rounded_box(ax, (x, y_center - h/2), w, h, label, color=color, fontsize=7)

    # Latent heads
    rounded_box(ax, (7.8, 3.0), 1.0, 0.8, "mean\n[128]",     color=C_PURPLE, fontsize=8)
    rounded_box(ax, (7.8, 1.2), 1.2, 0.8, "variance\n[128]",  color=C_PURPLE, fontsize=8)
    rounded_box(ax, (9.5, 1.9), 1.0, 1.2, "z\n[128]",         color=C_LATENT, fontsize=9)

    dec_layers = [
        ("256 x 4 x 4",   11.0, 2.0, 0.9, C_DEC),
        ("128 x 8 x 8",   12.2, 2.6, 0.9, C_DEC),
        ("64 x 16 x 16",  13.3, 3.2, 1.0, C_DEC),
        ("32 x 32 x 32",  14.4, 3.8, 1.0, C_DEC),
        ("3 x 64 x 64\nOutput", 15.6, 4.5, 1.2, C_INPUT),
    ]
    for (label, x, h, w, color) in dec_layers:
        rounded_box(ax, (x - 0.1, y_center - h/2), w, h, label, color=color, fontsize=7)

    # Encoder arrows
    for i in range(len(enc_layers) - 1):
        x1 = enc_layers[i][1] + enc_layers[i][3]
        x2 = enc_layers[i+1][1]
        arrow(ax, (x1 + 0.05, y_center), (x2 - 0.05, y_center), color="#666666")

    x_last = enc_layers[-1][1] + enc_layers[-1][3]
    arrow(ax, (x_last + 0.05, y_center + 0.3), (7.75, 3.4), color="#666666")
    arrow(ax, (x_last + 0.05, y_center - 0.3), (7.75, 1.6), color="#666666")
    arrow(ax, (8.85, 3.4), (9.45, 2.8), color=C_DEC)
    arrow(ax, (9.05, 1.6), (9.45, 2.2), color=C_DEC)
    arrow(ax, (10.55, y_center), (10.85, y_center), color="#666666")

    for i in range(len(dec_layers) - 1):
        x1 = dec_layers[i][1] - 0.1 + dec_layers[i][3]
        x2 = dec_layers[i+1][1] - 0.1
        arrow(ax, (x1 + 0.05, y_center), (x2 - 0.05, y_center), color="#666666")

    # Section labels
    ax.text(3.5, 5.5, "ENCODER", fontsize=14, color=C_ENC, fontweight="bold", ha="center")
    ax.text(9.0, 5.5, "LATENT",  fontsize=14, color=C_LATENT, fontweight="bold", ha="center")
    ax.text(13.5, 5.5, "DECODER", fontsize=14, color=C_DEC, fontweight="bold", ha="center")

    ax.axvline(x=7.5,  color="#444444", linestyle="--", lw=1)
    ax.axvline(x=10.8, color="#444444", linestyle="--", lw=1)

    fig.savefig(f"{DIR}/architecture.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary / Loss diagram — plain component names
# ─────────────────────────────────────────────────────────────────────────────
def make_summary():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    setup_ax(ax)
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 5)

    steps = [
        (0.0,  "Input",          C_INPUT),
        (2.5,  "Encoder",        C_ENC),
        (5.0,  "Latent\nSample", C_LATENT),
        (7.5,  "Decoder",        C_DEC),
        (10.0, "Output",         C_INPUT),
    ]
    for (x, label, color) in steps:
        rounded_box(ax, (x, 2.8), 1.8, 1.5, label, color=color, fontsize=11)
    for i in range(len(steps) - 1):
        arrow(ax, (steps[i][0] + 1.9, 3.55), (steps[i+1][0] - 0.1, 3.55), color=C_ARROW, lw=2.5)

    # Loss boxes
    rounded_box(ax, (3.5, 0.2), 3.5, 1.2, "Reconstruction\nLoss", color=C_LOSS_R, fontsize=11)
    arrow(ax, (0.9, 2.8), (4.5, 1.5), color=C_LOSS_R, lw=1.5)
    arrow(ax, (10.9, 2.8), (6.2, 1.5), color=C_LOSS_R, lw=1.5)

    rounded_box(ax, (8.0, 0.2), 3.5, 1.2, "KL Divergence\nLoss", color=C_LOSS_K, fontsize=11)
    arrow(ax, (5.9, 2.8), (9.0, 1.5), color=C_LOSS_K, lw=1.5)

    ax.text(6.0, -0.3, "Total Loss = Reconstruction + KL", ha="center", fontsize=13,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#333355", edgecolor="#666688"))

    fig.savefig(f"{DIR}/summary.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()


if __name__ == "__main__":
    make_overview()
    make_ae_vs_vae()
    make_reparam()
    make_architecture()
    make_summary()
    print("All diagrams generated!")
