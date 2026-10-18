#!/usr/bin/env python3
"""Plot the conductance and appendix response figures from generated data."""

import argparse
import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FILLINGS = {
    "2over5": (
        "2/5",
        [("1p", "1p"), ("1p5", "1p5"), ("2p", "2p"), ("3p", "3p")],
    ),
    "2over3": ("2/3", [("1p", None), ("1p5", None), ("2p", None), ("3p", None)]),
    "4over3": ("4/3", [("1p", None), ("1p5", None), ("2p", None), ("3p", None)]),
}

COLORS = [
    (0 / 255, 130 / 255, 200 / 255),
    (245 / 255, 130 / 255, 48 / 255),
    (128 / 255, 0 / 255, 0 / 255),
    (0 / 255, 0 / 255, 128 / 255),
]


def decode_number(token: str) -> str:
    """Convert a filename token such as 1p45 to its display form."""
    token = token[:-1] if token.endswith("p") else token
    return token.replace("p", ".")


def curve_path(
    data_dir: Path, filling: str, scenario: str, g_charge: str, g_neutral: str | None
) -> Path:
    parameter_suffix = f"_gC_{g_charge}"
    if g_neutral is not None:
        parameter_suffix += f"_gN_{g_neutral}"
    return data_dir / f"conductance_nu_{filling}_{scenario}{parameter_suffix}.csv"


def read_curve(path: Path) -> tuple[list[float], list[float]]:
    """Read temperature in kelvin and physical conductance in units of G_Q."""
    temperatures: list[float] = []
    conductances: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        expected = {"temperature_K", "mathcal_G", "G_over_GQ"}
        if rows.fieldnames is None or set(rows.fieldnames) != expected:
            raise ValueError(f"Unexpected columns in {path}: {rows.fieldnames}")
        for row in rows:
            temperature = float(row["temperature_K"])
            conductance = float(row["G_over_GQ"])
            if not (math.isfinite(temperature) and math.isfinite(conductance)):
                raise ValueError(f"Non-finite value in {path}")
            temperatures.append(temperature)
            conductances.append(conductance)
    if len(temperatures) < 2:
        raise ValueError(f"Too few adaptive points in {path}")
    if any(right <= left for left, right in zip(temperatures, temperatures[1:])):
        raise ValueError(f"Temperature mesh is not strictly increasing in {path}")
    return temperatures, conductances


def parameter_label(g_charge: str, g_neutral: str | None) -> str:
    charge = decode_number(g_charge)
    if g_neutral is None:
        return rf"$g_{{\mathrm{{C}}}}={charge}$"
    neutral = decode_number(g_neutral)
    if charge == neutral:
        return rf"$g_{{\mathrm{{C}}}}=g_{{\mathrm{{N}}}}={charge}$"
    return (
        rf"$(g_{{\mathrm{{C}}}},g_{{\mathrm{{N}}}})"
        rf"=({charge},{neutral})$"
    )


def shade_low_temperature(axis, crossover_temperature: float) -> None:
    """Fill the panel's stated low-temperature reference interval."""
    axis.axvspan(
        0.0,
        crossover_temperature,
        facecolor="#c6dfc3",
        edgecolor="none",
        linewidth=0,
        zorder=0,
    )


def render_filling(
    filling: str, crossover_temperature: float, data_dir: Path, output_dir: Path
) -> None:
    filling_fraction, parameter_values = FILLINGS[filling]
    nu = float(Fraction(filling_fraction))
    figure, axis = plt.subplots(figsize=(3.4, 2.55), dpi=600)
    shade_low_temperature(axis, crossover_temperature)

    all_conductances: list[float] = []
    scenario_styles = {
        "sc_stronger": "-",
        "tunneling_stronger": "--",
    }
    plot_order = ("tunneling_stronger", "sc_stronger")
    for color, (g_charge, g_neutral) in zip(COLORS, parameter_values):
        for scenario in plot_order:
            path = curve_path(data_dir, filling, scenario, g_charge, g_neutral)
            temperatures, conductances = read_curve(path)
            all_conductances.extend(conductances)
            axis.plot(
                temperatures,
                conductances,
                color=color,
                linestyle=scenario_styles[scenario],
            )

    if filling == "2over3":
        axis.set_ylim(-0.035 * nu, 0.06 * nu)
        axis.set_yticks([-0.02, 0.0, 0.02, 0.04])
    elif filling == "4over3":
        axis.set_ylim(-0.28 * nu, 0.28 * nu)
    else:
        data_min = min(all_conductances)
        data_max = max(all_conductances)
        span = max(data_max - data_min, 1.0e-3)
        padding = 0.06 * span
        axis.set_ylim(data_min - padding, data_max + padding)
    axis.set_xlim(0.0, 10.0)
    axis.set_xlabel(r"$T[\mathrm{K}]$")
    axis.set_ylabel(r"$G/G_{\mathrm Q}$", labelpad=0)
    axis.grid(True)

    if filling == "4over3":
        axis.axhline(1 / 3, color="black", linewidth=1.0, linestyle="-.")
        axis.axhline(-1 / 3, color="black", linewidth=1.0, linestyle="-.")

    parameter_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle="-",
            label=parameter_label(g_charge, g_neutral),
        )
        for color, (g_charge, g_neutral) in zip(COLORS, parameter_values)
    ]
    legend_options = dict(
        framealpha=1.0,
        handlelength=1.2,
        labelspacing=0.2,
        borderpad=0.3,
        columnspacing=0.7,
    )
    scenario_handles = [
        Line2D(
            [], [], color="black", linestyle="-", label=r"$y_{\mathrm A}>y_{\mathrm T}$"
        ),
        Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            label=r"$y_{\mathrm T}>y_{\mathrm A}$",
        ),
    ]
    if filling == "2over3":
        parameter_legend = axis.legend(
            handles=parameter_handles,
            loc="lower left",
            bbox_to_anchor=(0.008, 0.008),
            fontsize=7.5,
            **legend_options,
        )
        axis.add_artist(parameter_legend)
        axis.legend(
            handles=scenario_handles,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.008),
            fontsize=7.5,
            **legend_options,
        )
    elif filling == "4over3":
        parameter_legend = axis.legend(
            handles=parameter_handles,
            loc="upper left",
            bbox_to_anchor=(0.10, 0.90),
            **legend_options,
        )
        axis.add_artist(parameter_legend)
        axis.legend(
            handles=scenario_handles,
            loc="upper left",
            bbox_to_anchor=(0.46, 0.90),
            fontsize=7.5,
            **legend_options,
        )
    else:
        handles = parameter_handles + scenario_handles
        handles = [handles[i] for i in (0, 2, 4, 1, 3, 5)]
        axis.legend(
            handles=handles, loc="upper right", ncol=2, fontsize=7, **legend_options
        )

    if filling == "2over3":
        inset = axis.inset_axes([0.12, 0.76, 0.34, 0.18])
        shade_low_temperature(inset, crossover_temperature)
        for color, (g_charge, g_neutral) in zip(COLORS, parameter_values):
            for scenario in plot_order:
                temperatures, conductances = read_curve(
                    curve_path(data_dir, filling, scenario, g_charge, g_neutral)
                )
                inset.plot(
                    temperatures,
                    conductances,
                    color=color,
                    linestyle=scenario_styles[scenario],
                )
        inset.tick_params(labelsize=6.5, pad=1.5, length=2)
        inset.grid(True)
        inset.set_xlim(0.0, 3.0)
        inset.set_xticks([0, 1, 2, 3])
        inset.set_yticks([0.0, 0.01], labels=[r"$0$", r"$0.01$"])
        # Resolve the two small non-quantized integer plateaus: CAR-dominated
        # integer scattering is negative and tunneling-dominated scattering
        # is positive.
        inset.set_ylim(-6e-3 * nu, 27e-3 * nu)
        # The parent grid continues outside the inset frame. Give the small
        # tick labels a white background so no parent grid line crosses text.
        for label in inset.get_xticklabels() + inset.get_yticklabels():
            label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.3))

    if filling == "2over5":
        # The g=2 return to transmission occurs below a millikelvin and is
        # compressed against the vertical axis in the main linear-scale plot.
        # Leave room below the inset for its tick labels, above the main curves.
        inset = axis.inset_axes([0.38, 0.62, 0.52, 0.16])
        for scenario in plot_order:
            temperatures, conductances = read_curve(
                curve_path(data_dir, filling, scenario, "2p", "2p")
            )
            inset.semilogx(
                temperatures,
                conductances,
                color=COLORS[2],
                linestyle=scenario_styles[scenario],
            )
        inset.axhline(nu, color="black", linewidth=0.6, linestyle=":")
        inset.set_xlim(1e-7, 1e-2)
        inset.set_ylim(-0.45, 0.48)
        inset.set_xticks([1e-6, 1e-4, 1e-2])
        inset.set_yticks([-nu, nu], labels=[r"$-2/5$", r"$+2/5$"])
        inset.tick_params(labelsize=6.5, pad=1.5, length=2)
        inset.minorticks_off()
        inset.grid(True, which="major", alpha=0.3, linewidth=0.5)
        inset.text(
            0.96,
            0.72,
            r"$g=2$",
            transform=inset.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        for label in inset.get_xticklabels() + inset.get_yticklabels():
            label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.3))

    pdf_name = f"Conductance_{filling}.pdf"
    png_name = f"Conductance_{filling}.png"
    # Fit the labels closely while preserving the figure's printed aspect ratio.
    figure.tight_layout(pad=0.1)
    figure.savefig(output_dir / pdf_name)
    figure.savefig(output_dir / png_name, dpi=600)
    plt.close(figure)


def render_response(metadata, data_dir, output_dir):
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), layout="constrained")
    with (data_dir / "response_crossover.csv").open(newline="") as handle:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    for ax, (d, spin), letter in zip(axes, metadata["responseCases"], "abc"):
        case = [row for row in rows if row["d"] == d and row["spin"] == spin]
        x = np.array([row["x"] for row in case])
        values = np.array([[row["exact"], row["short"], row["long"]] for row in case])
        ax.loglog(x, values[:, 0], color="black", lw=1.7, label="Exact integral")
        ax.loglog(
            x, values[:, 1], color="#0072B2", ls="--", lw=1.3, label="Short segment"
        )
        ax.loglog(
            x, values[:, 2], color="#D55E00", ls=":", lw=1.7, label="Long segment"
        )
        ax.set_title(rf"({letter}) $d_r={d:g},\ s_r={spin:g}$")
        ax.set_xlabel(r"$\ell_{\mathrm{seg}}/\ell_T$")
        ax.set_xlim(0.01, 100)
        ax.grid(alpha=0.2, which="major")
    axes[0].set_ylabel(r"$A_r(T)$", labelpad=0)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.savefig(output_dir / "Response_crossover.pdf")
    fig.savefig(output_dir / "Response_crossover.png", dpi=220)
    plt.close(fig)


def render_conductances(metadata, data_dir, output_dir):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "lines.linewidth": 1.0,
        }
    )
    for filling in FILLINGS:
        threshold = metadata["shadingThresholdsK"][filling]
        render_filling(filling, threshold, data_dir, output_dir)
        print(f"{filling}: shading T < {threshold:.6g} K.")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=script_dir / "generated_data_finite"
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "paper_figures")
    args = parser.parse_args()
    metadata_path = args.data_dir / "run_metadata.json"
    if not metadata_path.exists():
        parser.error(
            "Missing data. Run conductance_pipeline.py first (with the same --data-dir)."
        )
    metadata = json.loads(metadata_path.read_text())
    if "shadingThresholdsK" not in metadata or "responseCases" not in metadata:
        parser.error(
            "Data predates the plotting split. Run conductance_pipeline.py to regenerate it."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_conductances(metadata, args.data_dir, args.output_dir)
    render_response(metadata, args.data_dir, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
