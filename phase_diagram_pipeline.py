#!/usr/bin/env python3
"""Generate Gaussian weak-coupling charge/transport diagrams.

Both use N=2, p=1. At nu=2/5, thermal segments require comparing coherent
rates T^(2*d-3) with the fractional bath rate T^(2*d_e-1).
At nu=3/5, assume g_I=1, a fixed coherence length, no Gaussian IF mixing,
common fractional velocities, u_I/u_f -> 0, and no direct fractional bath loss.
Streamlines show the field from strictly relevant elementary and neutral
operators, at equal elementary amplitudes and a stated neutral/elementary
amplitude ratio. They are not integrated RG trajectories. Colors classify
Gaussian electrical regimes before any neutral coupling reaches strong coupling.
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
    "Relevant CAR",
    "Relevant tunneling",
    "Leading CAR",
    "Leading tunneling",
]


def scaling_dimensions(g_charge, g_neutral):
    """Dimensions for the N=2, p=1 Jain block and normal IF at g_I=1."""
    gc, gn = np.asarray(g_charge), np.asarray(g_neutral)
    delta_a = (5 / gc + 1 / gn) / 2
    delta_t = (5 * gc + gn) / 2
    delta_if = 0.5 + (delta_a + delta_t) / 4
    return delta_a, delta_t, delta_if


def rate_exponents(g_charge, g_neutral, filling):
    """Return temperature powers for CAR, tunneling, and loss/IF rates."""
    delta_a, delta_t, delta_if = scaling_dimensions(g_charge, g_neutral)
    delta_e = (delta_a + delta_t) / 4
    if filling == "2over5":
        return 2 * delta_a - 3, 2 * delta_t - 3, 2 * delta_e - 1
    if filling == "3over5":
        return 2 * delta_a - 2, 2 * delta_t - 2, 2 * delta_if - 2
    raise ValueError(f"Unknown filling: {filling}")


def phase_indices(g_charge, g_neutral, filling):
    delta_a, delta_t, _ = scaling_dimensions(g_charge, g_neutral)
    powers = rate_exponents(g_charge, g_neutral, filling)
    phases = np.argmin(np.stack(powers), axis=0) + 2
    phases = np.where(delta_a < 2, 0, phases)
    return np.where(delta_t < 2, 1, phases)


def gaussian_parameter_flow(
    g_charge, g_neutral, *, y_a=0, y_t=0, y_neutral_a=0,
    y_neutral_t=0, alpha_f=0, y_if=0, velocity_ratio=0,
):
    """Instantaneous quadratic beta functions for N=2, p=1 at g_I=1.

    ``y_if`` is the counter-propagating IF amplitude. Its coefficient is
    (1-v)/(1+v)^2, with v=u_I/u_f; omit it for a purely fractional edge.
    Amplitudes may be arrays or complex numbers. The fractional velocities
    are equal at the shell where these derivatives are evaluated.
    """
    gc, gn = np.asarray(g_charge), np.asarray(g_neutral)
    a, t = np.abs(y_a) ** 2, np.abs(y_t) ** 2
    na, nt = np.abs(y_neutral_a) ** 2, np.abs(y_neutral_t) ** 2
    relaxation = alpha_f + np.abs(y_if) ** 2 * (
        (1 - velocity_ratio) / (1 + velocity_ratio) ** 2
    )
    return (
        5 * (a - t * gc**2 + relaxation * (1 - gc**2)),
        a - t * gn**2 + relaxation * (1 - gn**2) + 2 * (na - nt * gn**2),
    )


def relevant_parameter_flow(g_charge, g_neutral, neutral_ratio=1.0):
    """Schematic relevant contributions, divided by a common weak |y|^2.

    At each point, retain only operators of dimension <2. The elementary
    amplitude is |y| and the neutral amplitude is neutral_ratio*|y|.
    Irrelevant terms, fusion, and amplitude evolution are not plotted.
    At g_N=1 the two equally weighted marginal neutral terms cancel.
    Both paper fillings use this field: fractional dissipation is irrelevant
    and the retained IF dimension is >=2 throughout the Gaussian plane.
    """
    if not np.isfinite(neutral_ratio) or neutral_ratio < 0:
        raise ValueError("The neutral amplitude ratio must be finite and nonnegative")
    gc, gn = np.asarray(g_charge), np.asarray(g_neutral)
    delta_a, delta_t, _ = scaling_dimensions(gc, gn)
    return gaussian_parameter_flow(
        gc, gn,
        y_a=np.asarray(delta_a < 2, dtype=float),
        y_t=np.asarray(delta_t < 2, dtype=float),
        y_neutral_a=neutral_ratio * (gn > 1),
        y_neutral_t=neutral_ratio * (gn < 1),
    )


def render_phase_diagram(filling, output_dir, neutral_ratio=1.0):
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
    delta_a, delta_t, _ = scaling_dimensions(gc, gn)
    phases = phase_indices(gc, gn, filling)
    colors = COLORS
    labels = LABELS + ["Leading IF" if filling == "3over5" else "Leading dissipation"]
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
    powers = rate_exponents(gc, gn, filling)
    for first, second, other in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        # A tie between subleading rates is not a region boundary.
        hidden = (powers[other] < np.minimum(powers[first], powers[second])) | (
            (delta_a < 2) | (delta_t < 2)
        )
        difference = np.ma.masked_where(hidden, powers[first] - powers[second])
        if difference.count() == 0 or difference.min() > 0 or difference.max() < 0:
            continue
        axis.contour(
            gc,
            gn,
            difference,
            levels=[0],
            colors="black",
            linewidths=0.8,
            linestyles=":",
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
        handles=[Patch(color=c, label=s) for c, s in zip(colors, labels)],
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
    # Streamlines of the same schematic field, with no amplitude evolution.
    # Normalization changes the speed along a curve, not its direction.
    dg_c, dg_n = relevant_parameter_flow(gc, gn, neutral_ratio)
    magnitude = np.hypot(dg_c, dg_n)
    # Stop at switches in the retained operator set. Interpolation across
    # these discontinuities would invent flow along the threshold curves.
    branches = (delta_a < 2).astype(int) + 2 * (delta_t < 2) + 4 * (gn > 1)
    switches = np.zeros_like(branches, dtype=bool)
    horizontal = branches[:, 1:] != branches[:, :-1]
    vertical = branches[1:, :] != branches[:-1, :]
    switches[:, 1:] |= horizontal
    switches[:, :-1] |= horizontal
    switches[1:, :] |= vertical
    switches[:-1, :] |= vertical
    speed = np.ma.masked_where((magnitude == 0) | switches, magnitude)
    axis.streamplot(
        values, values, dg_c / speed, dg_n / speed,
        color="#00000080", linewidth=0.5, arrowsize=0.8,
        density=0.6, broken_streamlines=False, zorder=3,
    )
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
    parser.add_argument(
        "--neutral-ratio", type=float, default=1.0,
        help="Relevant neutral/elementary amplitude ratio in the arrows (default: 1)",
    )
    args = parser.parse_args()
    if not np.isfinite(args.neutral_ratio) or args.neutral_ratio < 0:
        parser.error("--neutral-ratio must be finite and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filling in ("2over5", "3over5"):
        render_phase_diagram(filling, args.output_dir, args.neutral_ratio)


if __name__ == "__main__":
    main()
