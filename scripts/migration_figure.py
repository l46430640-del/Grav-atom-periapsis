"""Plot orbital decay and the cumulative horizon count on the same clock."""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
import numpy as np


DATA = Path(__file__).resolve().parents[1] / "data"
CASES = [(3, "#235589", "-", "o", "Libration"), (60, "#ac4728", (0, (4, 2)), "s", "Circulation")]


def histories():
    grid = json.loads((DATA / "formed_periapsis.json").read_text())["grid"]
    params = grid["parameters"]
    # s = epsilon_c t/t_B; the mass at this stage sets t_B, not the initial hole mass.
    mass_seconds = grid["formation"]["mass"] * 1.e4 * 4.925490947e-6
    years_per_s = mass_seconds / (params["alpha"]**3 * params["cloud"] * 365.25 * 86400.)
    with np.load(DATA / "selection_traces.npz") as traces:
        for offset, color, style, marker, motion in CASES:
            row = dict(zip(traces["columns"], traces[f"migration-{offset}-0.0001-C-zero"].T))
            label = f"{motion}  ({90 + offset}\N{DEGREE SIGN})"
            yield row["s"]*years_per_s, row["a"], row["horizon_fraction"], color, style, marker, label


def draw(destination):
    style = {"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
             "font.size": 9.6, "mathtext.fontset": "stix", "axes.linewidth": .65,
             "lines.solid_capstyle": "round", "axes.labelsize": 10.2,
             "xtick.direction": "in", "ytick.direction": "in"}
    with mpl.rc_context(style):
        fig, (orbit, horizon) = plt.subplots(1, 2, figsize=(7.0, 2.18), sharex=True)
        fig.subplots_adjust(left=.087, right=.977, bottom=.23, top=.84, wspace=.40)
        # Match derivatives at the join; an exactly empty mode remains visible at zero.
        horizon.set_yscale("symlog", linthresh=1.e-5, base=10, linscale=.9/np.log(10))
        for years, a, count, color, ls, marker, label in histories():
            for ax, y in ((orbit, a), (horizon, count)):
                ax.plot(years, y, color=color, ls=ls, lw=1.36, label=label, zorder=3)
                ax.plot(years[-1], y[-1], marker, ms=4.2, mec=color, mfc="white",
                        mew=1., zorder=4)
        for ax, heading in ((orbit, "(a)  Orbital decay"), (horizon, "(b)  Horizon absorption")):
            ax.set_title(heading, fontsize=10.6, loc="left", pad=6, fontweight="bold")
            ax.set_xlim(-.6, 30.8)
            ax.set_xticks([0, 10, 20, 30])
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.tick_params(which="both", top=True, right=True, width=.55, pad=3)
            ax.tick_params(which="major", length=3.4)
            ax.tick_params(which="minor", length=1.9)
            ax.set_xlabel(r"$t$ (yr)", labelpad=2)
        orbit.set(ylim=(3.087, 3.266), yticks=[3.10, 3.15, 3.20, 3.25], ylabel=r"$a/a_B$")
        orbit.yaxis.set_minor_locator(MultipleLocator(.025))
        orbit.axhline(3.10, color="#aab0b4", lw=.58, ls=(0, (2, 2)), zorder=1)
        orbit.legend(loc="upper right", frameon=False, fontsize=8.9, borderaxespad=.28,
                     handlelength=2.1, labelspacing=.60, handletextpad=.5)
        horizon.set_ylim(-1.4e-6, .0021)
        horizon.set_yticks([0, 1.e-5, 1.e-4, 1.e-3],
                           ["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"])
        horizon.set_ylabel(r"$F_H$", labelpad=5)
        horizon.yaxis.set_minor_locator(NullLocator())
        fig.savefig(destination, metadata={"Title": "Orbital decay and particles absorbed by the horizon",
                                           "CreationDate": None, "ModDate": None})
        plt.close(fig)


if __name__ == "__main__":
    draw(Path(__file__).with_name("migration.pdf"))
