#!/usr/bin/env python3
"""Head-to-head scoreboard: jevbetter vs jevlike on the benchmark numbers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#0b0d12"
GRAY = "#9aa3b2"
JL = "#8a93a6"      # jevlike gray
JB = "#7bd88f"      # jevbetter green

# (metric label, jevlike text, jevlike goodness 0..1, jevbetter text, jevbetter goodness 0..1,
#  winner: 'jl' | 'jb', note)
ROWS = [
    ("TOP-1 ACCURACY", "87.3%", 0.873, "91.6%", 0.916, "jb", "+4.3 pp"),
    ("TOP-3 ACCURACY", "99.5%", 0.995, "99.9%", 0.999, "jb", ""),
    ("MRR", "--", 0.0, "0.955", 0.955, "jb", "jevlike n/a"),
    ("CALIBRATION ERROR  (lower is better)", "0.0367", 1 - 0.0367 / 0.05,
     "0.0182", 1 - 0.0182 / 0.05, "jb", "2x lower"),
    ("THROUGHPUT  (menus/sec, log scale)", "4,608", 1.0, "40", 0.437,
     "jl", "the encoder cost"),
]

fig, ax = plt.subplots(figsize=(11.35, 6.37), dpi=100)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 92, "jevbetter  vs  jevlike", ha="center", va="center",
        color="white", fontsize=21, family="serif")
ax.text(50, 86.5, "head-to-head  ·  800 held-out hard menus  ·  identical data, matched budgets",
        ha="center", va="center", color=GRAY, fontsize=9.5, family="monospace")

# scoreboard
ax.text(44, 78, "jevbetter", ha="right", va="center", color=JB, fontsize=13,
        family="monospace", weight="bold")
ax.text(50, 77, "4 - 1", ha="center", va="center", color="white", fontsize=30,
        family="monospace", weight="bold")
ax.text(56, 78, "jevlike", ha="left", va="center", color=JL, fontsize=13,
        family="monospace", weight="bold")

ax.text(31, 70.5, "jevlike", ha="center", va="center", color=JL, fontsize=10,
        family="monospace")
ax.text(69, 70.5, "jevbetter", ha="center", va="center", color=JB, fontsize=10,
        family="monospace", weight="bold")

ys = [62, 51, 40, 29, 18]
for y, (label, jl_t, jl_g, jb_t, jb_g, winner, note) in zip(ys, ROWS):
    tag = f"{label}  ·  {note}" if note else label
    ax.text(50, y + 4.6, tag, ha="center", va="center", color=GRAY,
            fontsize=8.5, family="monospace")
    # diverging bars from center
    ax.barh(y, -jl_g * 36, left=50, height=4.2, color=JL, zorder=2)
    ax.barh(y, jb_g * 36, left=50, height=4.2, color=JB, zorder=2)
    ax.text(11.5, y, jl_t, ha="right", va="center",
            color="white" if winner == "jl" else JL, fontsize=12,
            family="monospace", weight="bold" if winner == "jl" else "normal")
    ax.text(88.5, y, jb_t, ha="left", va="center",
            color="white" if winner == "jb" else JB, fontsize=12,
            family="monospace", weight="bold" if winner == "jb" else "normal")

ax.text(50, 6.5,
        "jevbetter-benchmark  ·  same 8 epochs / batch / seed, CPU  ·  "
        "github.com/olanotolu/jevbetter",
        ha="center", va="center", color=GRAY, fontsize=8.5, family="serif",
        style="italic")

fig.tight_layout(pad=0.4)
fig.savefig("docs/scoreboard.png", facecolor=BG, dpi=100)
print("wrote docs/scoreboard.png")
