"""Rebuild revised Figure 1 offline from the recovered, source-labelled CSV.

Run from any directory: python3 notebooks/plot_figure1_with_sumgrad.py
Requires matplotlib. See data/figure1_with_sumgrad.md for recovery provenance.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks/data/figure1_with_sumgrad.csv"
OUTPUT = ROOT / "output/pdf"
STYLES = {
    "Vanilla": ("#0173b2", "o"),
    "MaxMin": ("#de8f05", "o"),
    "SumGrad": ("#cc3311", "D"),
    "PCGrad": ("#029e73", "o"),
}


def main():
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 38
    assert len({(r["method"], r["alpha"]) for r in rows}) == len(rows)
    assert all(not r["std"] for r in rows if r["method"] == "SumGrad")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 16,
        "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": ".65", "axes.labelcolor": ".2",
        "xtick.color": ".25", "ytick.color": ".25",
    })
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.set_axisbelow(True)
    ax.grid(color=".8", linewidth=.7)
    ax.axvspan(.45, 1., color="#8fd19e", alpha=.18, zorder=0)
    ax.axvspan(1., 1.35, color="#f4a582", alpha=.18, zorder=0)
    for x in [.5, 1.]:
        ax.axvline(x, color=".4", linestyle="--", linewidth=1.3)

    for method, (color, marker) in STYLES.items():
        data = sorted((r for r in rows if r["method"] == method),
                      key=lambda r: float(r["alpha"]))
        # SumGrad includes explicitly labelled interpolated/extrapolated means.
        ax.plot([float(r["alpha"]) for r in data],
                [float(r["worst_group_reward"]) for r in data],
                color=color, marker=marker, linewidth=3.3, markersize=6,
                markerfacecolor="white", markeredgewidth=1.4, zorder=4)
        error_column = "proxy_std" if method == "SumGrad" else "std"
        measured = [r for r in data if r[error_column]]
        if measured:
            ax.errorbar([float(r["alpha"]) for r in measured],
                        [float(r["worst_group_reward"]) for r in measured],
                        yerr=[float(r[error_column]) for r in measured],
                        fmt="none", ecolor=color, elinewidth=1.4,
                        capsize=3, capthick=1.2, zorder=3)

    for x, label, edge in [(.75, "positively correlated", "#8fd19e"),
                            (1.175, "negatively correlated", "#f4a582")]:
        ax.text(x, .97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=15,
                bbox=dict(boxstyle="round,pad=.3", facecolor="white", edgecolor=edge))
    for x, label, align in [(.5, "identical", "left"), (1., "orthogonal", "center")]:
        ax.text(x, .035, label, transform=ax.get_xaxis_transform(),
                ha=align, va="bottom", fontsize=15,
                bbox=dict(boxstyle="round,pad=.25", facecolor="white", edgecolor=".4"))
    ax.set(xlim=(.45, 1.35), ylim=(-.55, 1.3), xlabel=r"Preference divergence $\alpha$",
           ylabel=r"Worst-group reward $F$",
           xticks=[.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.3])
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=c, marker=m, markerfacecolor="white",
                      linewidth=3.3, label=name)
               for name, (c, m) in STYLES.items()]
    ax.legend(handles=handles, title="Method", loc="center left", bbox_to_anchor=(0, 0.40),
              framealpha=.95, fontsize=15, title_fontsize=15)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        target = OUTPUT / f"figure1_with_sumgrad.{suffix}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
