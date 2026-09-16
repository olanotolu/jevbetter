#!/usr/bin/env python3
"""Draw the jevbetter architecture in the tubeworks style: rainbow tubes on
dark background, one stream per tensor, boxes for ops, equations at right."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

BG = "#0b0d12"
INK = "#14171f"
RAINBOW = ["#ff5d8f", "#ff9e4d", "#ffd54d", "#7bd88f", "#5db9ff", "#b388ff"]
GRAY = "#9aa3b2"
NEW_EDGE = "#7bd88f"


def catmull_rom(pts, samples=26):
    pts = np.asarray(pts, float)
    P = [pts[0], *pts, pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5 * ((2 * p1) + (-p0 + p2) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            )
    out.append(pts[-1])
    return np.array(out)


def tube(ax, waypoints, colors=RAINBOW, gap=0.85, lw=5.0, zorder=2):
    curve = catmull_rom(waypoints)
    tang = np.gradient(curve, axis=0)
    n = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1) / n
    m = len(colors)
    for k, c in enumerate(colors):
        pts = curve + normal * ((k - (m - 1) / 2) * gap)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=lw, alpha=0.95,
                solid_capstyle="round", zorder=zorder)


def block(ax, x, y, w, h, label, sub=None, new=False, fs=9):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.12",
        fc=INK, ec=NEW_EDGE if new else "white", lw=1.7 if new else 1.2, zorder=3))
    dy = 0.7 if sub else 0
    ax.text(x, y + dy, label, ha="center", va="center", color="white",
            fontsize=fs, family="monospace", weight="bold", zorder=4)
    if sub:
        ax.text(x, y - 1.1, sub, ha="center", va="center", color=GRAY,
                fontsize=7, family="monospace", zorder=4)


def junction(ax, x, y, sym, tag=None):
    ax.add_patch(Circle((x, y), 1.7, fc=INK, ec="white", lw=1.4, zorder=3))
    ax.text(x, y, sym, ha="center", va="center", color="white",
            fontsize=12, zorder=4)
    if tag:
        ax.text(x, y - 3.6, tag, ha="center", va="center", color=GRAY,
                fontsize=7, family="monospace", zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(11.35, 6.37), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(24, 96)
    ax.axis("off")

    ax.text(50, 91.5,
            "A jevbetter option scorer: N options + context in, prob(N) out, one pass",
            ha="center", va="center", color="white", fontsize=17, family="serif")

    # ---- options stream (y = 70) ----
    tube(ax, [(2, 70), (8, 70)])                       # in -> ngram
    tube(ax, [(11.5, 70), (13, 70)])                   # ngram -> rival
    tube(ax, [(20, 70), (21.75, 70)])                 # rival -> Q
    tube(ax, [(26.25, 70), (29.3, 70)])               # Q -> J1
    tube(ax, [(32.7, 70), (34.5, 70)])                # J1 -> softmax
    tube(ax, [(39.5, 70), (43, 70), (44, 65), (44.6, 59), (46, 56.2)])  # A down to J2
    block(ax, 8, 70, 7, 4.6, "n-gram", "hash embed", new=True)
    block(ax, 16.5, 70, 7, 4.6, "O↔O", "rival attn", new=True)
    block(ax, 24, 70, 4.5, 4.6, "Q")
    junction(ax, 31, 70, "⊗", "QKᵀ")
    block(ax, 37, 70, 5, 4.6, "softmax", "over i")

    # ---- context stream (y = 54) ----
    tube(ax, [(2, 54), (4.5, 54)])
    tube(ax, [(11.5, 54), (11.75, 54)])
    tube(ax, [(16.25, 54), (18, 54)])
    tube(ax, [(24, 54), (28, 54), (30, 57), (30, 59.2)])   # up toward K
    tube(ax, [(30, 62.8), (30, 66), (30, 68.3)])           # K -> J1
    tube(ax, [(24, 54), (33.75, 54)])                      # -> V
    tube(ax, [(38.25, 54), (44.3, 54)])                    # V -> J2
    block(ax, 8, 54, 7, 4.6, "n-gram", "hash embed", new=True)
    block(ax, 14, 54, 4.5, 4.6, "+pos")
    block(ax, 21, 54, 6, 4.6, "2-layer", "transformer", new=True)
    block(ax, 30, 61, 4, 3.6, "K", fs=8)
    block(ax, 36, 54, 4.5, 4.6, "V")

    junction(ax, 46, 54, "⊗", "A·V")

    # ---- rival residual arc O' -> J3 (skip connection over the top) ----
    tube(ax, [(16.5, 72.6), (16.5, 77), (24, 80), (40, 80), (52, 79),
              (57, 74), (57.5, 66), (56, 60), (54, 56.9), (52, 56.2)],
         lw=4.2, gap=0.7)
    junction(ax, 52, 54, "⊙", "O′⊙C")

    # ---- C -> head ----
    tube(ax, [(47.7, 54), (49.6, 54)])
    tube(ax, [(54.4, 54), (67.3, 54)])
    block(ax, 58.5, 54, 8.5, 4.6, "gated MLP", "z ⊙ σ(Wz)", new=True, fs=8.5)
    block(ax, 65, 54, 6, 4.6, "softmax", "s / T", new=True, fs=8.5)

    ax.text(1.2, 74.6, "options: N texts", color=GRAY, fontsize=8,
            family="monospace", ha="left", va="center")
    ax.text(1.2, 58.6, "context: 1 text", color=GRAY, fontsize=8,
            family="monospace", ha="left", va="center")
    ax.text(69.5, 54, "p", color="white", fontsize=12, family="monospace",
            weight="bold", ha="left", va="center")
    ax.text(33.5, 65.2, "4 heads", color=GRAY, fontsize=7, family="monospace",
            ha="left", va="center")

    # ---- equations (right) ----
    eqs = [
        "O : n d      options (mean-pooled n-grams)",
        "H : i d      context tokens (2L transformer)",
        "",
        "O′ = rival_attn(O)             n d → n d",
        "Q = O′ @ Wq                    n d → n d",
        "K = H @ Wk    V = H @ Wv       i d → i d",
        "A = softmax(Q Kᵀ / √d, axis='i')   (4 heads)",
        "C = A @ V                      n i,i d → n d",
        "z = O′ ⊙ C                     rival-aware",
        "s = MLP(z ⊙ σ(W_g z))          n d → n",
        "s = s / T                      temperature",
        "p = softmax(s, axis='n')",
    ]
    ax.text(74, 78, "\n".join(eqs), color="white", fontsize=9.5,
            family="monospace", ha="left", va="top", linespacing=1.45)

    ax.text(50, 27,
            "green-ringed blocks are new vs jevlike  ·  drawn with matplotlib",
            ha="center", va="center", color=GRAY, fontsize=8.5, family="serif",
            style="italic")

    fig.tight_layout(pad=0.4)
    fig.savefig("docs/architecture.png", facecolor=BG, dpi=100)
    print("wrote docs/architecture.png")


if __name__ == "__main__":
    main()
