#!/usr/bin/env python3
"""Generate the nu=2/5 and nu=3/5 weak-coupling phase diagrams.

Both use N=2, p=1. At nu=3/5, assume g_I=1, no Gaussian IF mixing,
common fractional velocities, u_I/u_f -> 0, and no direct fractional bath loss.
Arrows show the relevant CAR or tunneling contribution in the gapped regions.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

COLORS = ["#0082c8", "#e6194b", "#b5dcec", "#f5c0cd", "#c5dfb4"]
LABELS = [
    "Gapped CAR",
    "Gapped tunneling",
    "Gapless CAR",
    "Gapless tunneling",
    "Gapless IF",
]


def scaling_dimensions(g_charge, g_neutral):
    """Dimensions for the upstream N=2, p=1 Jain block and normal IF."""
    gc, gn = np.asarray(g_charge), np.asarray(g_neutral)
    delta_a = (5 / gc + 1 / gn) / 2
    delta_t = (5 * gc + gn) / 2
    delta_if = 0.5 + (delta_a + delta_t) / 4
    return delta_a, delta_t, delta_if


def phase_indices(g_charge, g_neutral, filling):
    delta_a, delta_t, delta_if = scaling_dimensions(g_charge, g_neutral)
    dimensions = (
        [delta_a, delta_t, delta_if] if filling == "3over5" else [delta_a, delta_t]
    )
    phases = np.argmin(np.stack(dimensions), axis=0) + 2
    phases = np.where(delta_a < 2, 0, phases)
    return np.where(delta_t < 2, 1, phases)


def render_phase_diagram(filling, output_dir):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    values = np.linspace(0.005, 5.0, 700)
    gc, gn = np.meshgrid(values, values)
    delta_a, delta_t, delta_if = scaling_dimensions(gc, gn)
    phases = phase_indices(gc, gn, filling)
    colors = COLORS if filling == "3over5" else COLORS[:4]
    figure, axis = plt.subplots(figsize=(3.4, 2.3))
    axis.pcolormesh(
        gc,
        gn,
        phases,
        cmap=ListedColormap(colors),
        shading="nearest",
        vmin=-0.5,
        vmax=len(colors) - 0.5,
        rasterized=True,
    )
    for dimension in [delta_a, delta_t]:
        axis.contour(
            gc,
            gn,
            dimension,
            levels=[2],
            colors="black",
            linewidths=0.8,
            linestyles="--",
        )
    boundaries = (
        [delta_a - delta_if, delta_t - delta_if]
        if filling == "3over5"
        else [delta_a - delta_t]
    )
    for difference in boundaries:
        axis.contour(
            gc,
            gn,
            difference,
            levels=[0],
            colors="black",
            linewidths=0.8,
            linestyles=":",
        )

    dg_c, dg_n = np.full_like(gc, np.nan), np.full_like(gn, np.nan)
    dg_c[phases == 0], dg_n[phases == 0] = 5.0, 1.0
    dg_c[phases == 1] = -5 * gc[phases == 1] ** 2
    dg_n[phases == 1] = -(gn[phases == 1] ** 2)
    magnitude = np.hypot(dg_c, dg_n)
    axis.streamplot(
        values,
        values,
        dg_c / magnitude,
        dg_n / magnitude,
        color="#00000080",
        linewidth=0.5,
        arrowsize=0.8,
        density=0.6,
        broken_streamlines=False,
        zorder=3,
    )
    axis.plot(
        1,
        1,
        "o",
        color="black",
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=4.5,
        zorder=5,
    )
    axis.legend(
        handles=[Patch(color=c, label=s) for c, s in zip(colors, LABELS)],
        loc="upper right",
        framealpha=0.96,
        fontsize=10,
        handlelength=1.2,
        borderpad=0.4,
        labelspacing=0.3,
    )
    axis.set_xlabel(r"$g_{\mathrm{C}}$")
    axis.set_ylabel(r"$g_{\mathrm{N}}$", labelpad=0)
    axis.set_xlim(0, 5)
    figure.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.20)
    # Keep the existing plot rectangle and use the same physical unit scale
    # on both axes by reducing the displayed neutral-parameter range.
    box = axis.get_position()
    width = figure.get_figwidth() * box.width
    height = figure.get_figheight() * box.height
    axis.set_ylim(0, 5 * height / width)
    axis.set_aspect("equal", adjustable="box")
    name = "pd_cn" if filling == "2over5" else "pd_3over5"
    figure.savefig(
        output_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300
    )
    figure.savefig(
        output_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=300
    )
    plt.close(figure)
    print(f"Updated {output_dir / name}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "paper_figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filling in ("2over5", "3over5"):
        render_phase_diagram(filling, args.output_dir)


if __name__ == "__main__":
    main()
