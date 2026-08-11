#!/usr/bin/env python3
"""Draw a conventional lines plan for the analytical Dufour 39 surrogate.

The drawing contains only the submerged, upright canoe body represented by
``dufour_39_approx_hull``.  It does not add topsides, keel, rudder, appendages,
or the production yacht's open transom.  Consequently this is an analytical
lines plan for the Michell-model geometry, not a builder's construction plan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np

from wave_resistance import HullGeometryWarning, dufour_39_approx_hull


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=161, help="longitudinal points")
    parser.add_argument("--nz", type=int, default=65, help="vertical points")
    parser.add_argument(
        "--svg",
        type=Path,
        default=SCRIPT_DIR / "dufour_39_lines_plan.svg",
        help="vector output path",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=SCRIPT_DIR / "dufour_39_lines_plan.png",
        help="raster output path",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    return parser.parse_args()


def interpolate_stations(hull, positions: np.ndarray) -> np.ndarray:
    """Return half-breadths at specified x positions on the offset grid."""

    return np.vstack(
        [
            np.interp(positions, hull.x, hull.half_breadths[:, level])
            for level in range(hull.z.size)
        ]
    ).T


def interpolate_waterline(hull, depth: float) -> np.ndarray:
    """Return the half-breadth curve at one global depth below the waterline."""

    return np.asarray(
        [np.interp(depth, hull.z, section) for section in hull.half_breadths]
    )


def buttock_profile(hull, half_breadth: float) -> np.ndarray:
    """Return z(x) where a section intersects a constant half-breadth plane."""

    tolerance = 64.0 * np.finfo(float).eps * max(1.0, half_breadth)
    depths = np.full_like(hull.x, np.nan)
    for station, section in enumerate(hull.half_breadths):
        if section[0] + tolerance < half_breadth:
            continue
        below = np.flatnonzero(section <= half_breadth + tolerance)
        below = below[below > 0]
        if below.size == 0:
            continue
        upper = int(below[0] - 1)
        lower = upper + 1
        y_upper = float(section[upper])
        y_lower = float(section[lower])
        if abs(y_upper - y_lower) <= tolerance:
            depths[station] = hull.z[lower]
        else:
            fraction = (y_upper - half_breadth) / (y_upper - y_lower)
            depths[station] = hull.z[upper] + fraction * (
                hull.z[lower] - hull.z[upper]
            )
    return depths


def analytical_rocker(hull) -> np.ndarray:
    """Return the smooth local centreline closure used by the surrogate."""

    half_beam = float(np.max(hull.half_breadths[:, 0]))
    longitudinal = hull.half_breadths[:, 0] / half_beam
    rocker = hull.draft_m * longitudinal**0.45
    rocker[[0, -1]] = 0.0
    return rocker


def style_axis(axis) -> None:
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_color("#4f5b63")
        spine.set_linewidth(0.7)
    axis.tick_params(colors="#37434b", labelsize=7, length=3, width=0.6)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HullGeometryWarning)
        hull = dufour_39_approx_hull(nx=args.nx, nz=args.nz)

    try:
        if not args.show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MultipleLocator
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required; install with `python -m pip install -e '.[plot]'`"
        ) from exc

    length = float(hull.metadata.length_ref_m)
    draft = float(hull.draft_m)
    waterline_half_beam = float(np.max(hull.half_breadths[:, 0]))
    station_numbers = np.arange(21)
    station_x = length * station_numbers / 20.0
    station_sections = interpolate_stations(hull, station_x)
    # Eight equal depth intervals: L.A. 0 is the design waterline and L.A. 8
    # coincides with the baseline, which is drawn separately.
    waterline_depths = draft * np.arange(8, dtype=float) / 8.0
    buttock_offsets = waterline_half_beam * np.asarray((0.2, 0.4, 0.6, 0.8))
    analytical_keel_z = analytical_rocker(hull)

    navy = "#17374c"
    blue = "#24759a"
    blue_light = "#8db6ca"
    rust = "#a44b32"
    green = "#3f7662"
    grid_colour = "#c7ced2"
    baseline_colour = "#66737b"
    annotation_colour = "#37434b"

    figure = plt.figure(figsize=(16.54, 11.69), constrained_layout=False)
    figure.patch.set_facecolor("white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.975,
        bottom=0.08,
        top=0.895,
        width_ratios=(4.65, 1.70),
        height_ratios=(1.0, 1.0),
        wspace=0.09,
        hspace=0.17,
    )
    profile = figure.add_subplot(grid[0, 0])
    plan = figure.add_subplot(grid[1, 0], sharex=profile)
    body = figure.add_subplot(grid[0, 1])
    notes = figure.add_subplot(grid[1, 1])

    for axis in (profile, plan, body):
        style_axis(axis)

    # Common station ordinates make the longitudinal views register exactly.
    for axis in (profile, plan):
        for x_value in station_x:
            axis.axvline(x_value, color=grid_colour, lw=0.48, zorder=0)

    # Piano longitudinale: waterline/baseline grid, immersed outline, buttocks.
    profile.axhline(0.0, color=rust, lw=1.25, zorder=4)
    profile.axhline(draft, color=baseline_colour, lw=0.9, ls=(0, (5, 3)), zorder=1)
    for number, depth in enumerate(waterline_depths):
        breadth = interpolate_waterline(hull, float(depth))
        valid = breadth > 1.0e-12
        if np.any(valid):
            x_valid = hull.x[valid]
            profile.plot(
                [x_valid[0], x_valid[-1]],
                [depth, depth],
                color=blue_light if number else rust,
                lw=0.65 if number else 1.25,
                zorder=1,
            )
    profile.plot(hull.x, analytical_keel_z, color=navy, lw=1.7, zorder=5)
    for offset in buttock_offsets:
        profile.plot(
            hull.x,
            buttock_profile(hull, float(offset)),
            color=rust,
            lw=0.85,
            alpha=0.88,
            zorder=3,
        )
    profile.set_xlim(0.0, length)
    profile.set_ylim(draft * 1.08, -0.045)
    profile.set_aspect(4.0, adjustable="box", anchor="C")
    profile.set_ylabel("z sotto il galleggiamento (m)", fontsize=8)
    profile.set_title(
        "PIANO LONGITUDINALE — PROFILO IMMERSO",
        loc="left",
        fontsize=10,
        fontweight="bold",
        color=navy,
        pad=9,
    )
    profile.tick_params(axis="x", labelbottom=False)
    profile.yaxis.set_major_locator(MultipleLocator(0.1))
    profile.text(
        0.015,
        0.08,
        "POPPA",
        transform=profile.transAxes,
        fontsize=7,
        color=annotation_colour,
        ha="left",
    )
    profile.text(
        0.985,
        0.08,
        "PRUA",
        transform=profile.transAxes,
        fontsize=7,
        color=annotation_colour,
        ha="right",
    )
    profile.text(
        length * 0.995,
        0.008,
        "L.G.  z=0",
        fontsize=6.5,
        color=rust,
        ha="right",
        va="bottom",
    )
    profile.text(
        length * 0.995,
        draft - 0.008,
        "LINEA DI BASE",
        fontsize=6.5,
        color=baseline_colour,
        ha="right",
        va="bottom",
    )
    station_axis = profile.secondary_xaxis("top")
    station_axis.set_xticks(station_x)
    station_axis.set_xticklabels([str(value) for value in station_numbers], fontsize=6.5)
    station_axis.tick_params(length=2.5, width=0.55, pad=2, colors=annotation_colour)
    station_axis.set_xlabel("NUMERO STAZIONE", fontsize=7, labelpad=4, color=annotation_colour)

    # Piano orizzontale: exact waterline curves of the offset tensor.
    plan.axhline(0.0, color=navy, lw=1.15, zorder=4)
    for offset in buttock_offsets:
        plan.axhline(offset, color=rust, lw=0.45, ls=(0, (4, 3)), alpha=0.55, zorder=0)
    for number, depth in enumerate(waterline_depths):
        breadth = interpolate_waterline(hull, float(depth))
        valid = np.flatnonzero(breadth > 1.0e-12)
        if valid.size:
            first = max(0, int(valid[0]) - 1)
            last = min(hull.x.size - 1, int(valid[-1]) + 1)
        else:
            first, last = 0, 0
        plan.plot(
            hull.x[first : last + 1],
            breadth[first : last + 1],
            color=navy if number == 0 else blue,
            lw=1.75 if number == 0 else 0.95,
            zorder=4 if number == 0 else 3,
        )
        if valid.size:
            label_index = int(valid[max(0, int(0.80 * (valid.size - 1)))])
            plan.annotate(
                "L.A. {}".format(number),
                (hull.x[label_index], breadth[label_index]),
                xytext=(2, 1),
                textcoords="offset points",
                fontsize=5.8,
                color=blue if number else navy,
                ha="left",
                va="bottom",
            )
    plan.set_ylim(waterline_half_beam * 1.09, -0.055)
    plan.set_aspect("equal", adjustable="box", anchor="C")
    plan.set_xlabel("x da poppa verso prua (m)", fontsize=8)
    plan.set_ylabel("semilarghezza y (m)", fontsize=8)
    plan.set_title(
        "PIANO ORIZZONTALE — SEMILARGHEZZE",
        loc="left",
        fontsize=10,
        fontweight="bold",
        color=navy,
        pad=9,
    )
    plan.xaxis.set_major_locator(MultipleLocator(1.0))
    plan.yaxis.set_major_locator(MultipleLocator(0.25))

    # Piano trasversale: after sections left, forward sections right.
    body.axvline(0.0, color=navy, lw=1.05, zorder=2)
    body.axhline(0.0, color=rust, lw=1.15, zorder=2)
    body.axhline(draft, color=baseline_colour, lw=0.8, ls=(0, (5, 3)), zorder=0)
    for depth in waterline_depths[1:]:
        body.axhline(depth, color=blue_light, lw=0.45, zorder=0)
    for offset in buttock_offsets:
        body.axvline(-offset, color=rust, lw=0.4, ls=(0, (4, 3)), alpha=0.48, zorder=0)
        body.axvline(offset, color=rust, lw=0.4, ls=(0, (4, 3)), alpha=0.48, zorder=0)
    for number in range(1, 20):
        sides = (
            (-1.0, 1.0)
            if number == 10
            else (-1.0 if number < 10 else 1.0,)
        )
        section = station_sections[number]
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(section)))
        positive = np.flatnonzero(section > tolerance)
        if positive.size == 0:
            continue
        # Stop at the first local centreplane closure.  Plotting the remaining
        # zero offsets would falsely suggest a centreplane bottom plate.
        closure = min(int(positive[-1]) + 1, hull.z.size - 1)
        for side in sides:
            body.plot(
                side * section[: closure + 1],
                hull.z[: closure + 1],
                color=green if number == 10 else navy,
                lw=1.3 if number == 10 else 0.95,
                alpha=0.95 if number == 10 else 0.90,
                zorder=4 if number == 10 else 3,
            )
            endpoint = side * float(section[0])
            if number % 4 == 0:
                body.annotate(
                    str(number),
                    (endpoint, 0.0),
                    xytext=(0, -4),
                    textcoords="offset points",
                    fontsize=5.8,
                    color=navy,
                    ha="center",
                    va="bottom",
                    clip_on=False,
                )
    body.text(
        -0.96 * waterline_half_beam,
        draft * 0.98,
        "SEZIONI POPPIERE",
        fontsize=6.4,
        color=annotation_colour,
        ha="left",
        va="bottom",
    )
    body.text(
        0.96 * waterline_half_beam,
        draft * 0.98,
        "SEZIONI PRODIERE",
        fontsize=6.4,
        color=annotation_colour,
        ha="right",
        va="bottom",
    )
    body.set_xlim(-waterline_half_beam * 1.10, waterline_half_beam * 1.10)
    body.set_ylim(draft * 1.08, -0.045)
    body.set_aspect(4.0, adjustable="box", anchor="C")
    body.set_xlabel("y (m): poppa ← C.L. → prua", fontsize=8)
    body.set_ylabel("z sotto il galleggiamento (m)", fontsize=8)
    body.set_title(
        "PIANO TRASVERSALE — SEZIONI",
        loc="left",
        fontsize=10,
        fontweight="bold",
        color=navy,
        pad=9,
    )
    body.xaxis.set_major_locator(MultipleLocator(0.5))
    body.yaxis.set_major_locator(MultipleLocator(0.1))

    # Compact title block and drawing key; no additional geometry is implied.
    notes.set_axis_off()
    notes.set_xlim(0.0, 1.0)
    notes.set_ylim(0.0, 1.0)
    notes.add_patch(
        plt.Rectangle(
            (0.0, 0.08),
            1.0,
            0.84,
            facecolor="none",
            edgecolor=baseline_colour,
            linewidth=0.8,
        )
    )
    notes.plot([0.0, 1.0], [0.67, 0.67], color=baseline_colour, lw=0.65)
    notes.plot([0.0, 1.0], [0.40, 0.40], color=baseline_colour, lw=0.65)
    notes.plot([0.50, 0.50], [0.08, 0.40], color=baseline_colour, lw=0.65)
    notes.text(
        0.04,
        0.87,
        "DUFOUR 39 (2026)",
        fontsize=13,
        fontweight="bold",
        color=navy,
        va="top",
    )
    notes.text(
        0.04,
        0.79,
        "PIANO DI COSTRUZIONE DELLA CARENA ANALITICA",
        fontsize=7.5,
        fontweight="bold",
        color=annotation_colour,
        va="top",
    )
    notes.text(
        0.04,
        0.71,
        "Non sono piani del cantiere",
        fontsize=7,
        color=rust,
        va="top",
    )
    notes.text(
        0.04,
        0.63,
        "LWL       10.500 m\n"
        "BWL ass.   3.700 m\n"
        "T canoa    {:.3f} m\n"
        "Volume     {:.3f} m³\n"
        "S bagnata  {:.2f} m²".format(draft, hull.displaced_volume, hull.wetted_area_m2),
        fontsize=7.2,
        family="DejaVu Sans Mono",
        color=annotation_colour,
        linespacing=1.35,
        va="top",
    )
    legend_handles = [
        Line2D([0], [0], color=navy, lw=1.5, label="contorno / sezioni"),
        Line2D([0], [0], color=blue, lw=1.0, label="linee d’acqua (L.A.)"),
        Line2D([0], [0], color=rust, lw=0.9, label="longitudinali (buttocks)"),
        Line2D(
            [0],
            [0],
            color=green,
            lw=1.3,
            label="sezione 10",
        ),
        Line2D(
            [0],
            [0],
            color=baseline_colour,
            lw=0.9,
            ls=(0, (5, 3)),
            label="linea di base",
        ),
    ]
    notes.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.38),
        frameon=False,
        fontsize=6.2,
        handlelength=2.4,
        borderaxespad=0.0,
        labelspacing=0.55,
    )
    notes.text(
        0.53,
        0.365,
        "Coordinate in metri; z positivo in basso.\n"
        "Profilo/sezioni: scala verticale ×4.\n"
        "Sezioni lisce senza spigoli.\n"
        "Solo carena immersa; opera morta,\n"
        "chiglia e timone esclusi. AP e FP chiuse\n"
        "sul piano di simmetria; specchio aperto\n"
        "reale non rappresentato.",
        fontsize=5.8,
        color=annotation_colour,
        linespacing=1.20,
        va="top",
    )

    figure.suptitle(
        "DUFOUR 39 — PIANI DI COSTRUZIONE DEL SURROGATO ANALITICO",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=navy,
    )
    figure.text(
        0.055,
        0.932,
        "Carena nuda alla linea di galleggiamento di progetto · geometria per il modello di Michell",
        ha="left",
        fontsize=9,
        color=annotation_colour,
    )
    figure.text(
        0.975,
        0.032,
        "Geometria analitica approssimata — non usare per la costruzione",
        ha="right",
        fontsize=7,
        color=rust,
    )

    for path in (args.svg, args.png):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.svg, format="svg", bbox_inches="tight", facecolor="white")
    figure.savefig(args.png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print("Wrote {}".format(args.svg))
    print("Wrote {}".format(args.png))
    print(
        "21 stations; {} waterlines; {} buttocks; LWL={:.3f} m; "
        "BWL={:.3f} m; canoe draft={:.3f} m; volume={:.3f} m3".format(
            waterline_depths.size,
            buttock_offsets.size,
            length,
            2.0 * waterline_half_beam,
            draft,
            hull.displaced_volume,
        )
    )
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
