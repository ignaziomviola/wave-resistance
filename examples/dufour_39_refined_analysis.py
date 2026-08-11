#!/usr/bin/env python3
"""Plot the refined Dufour 39 surrogate and compute Michell resistance.

The calculation is a deep-water, fixed-attitude thin-ship screening result for
the smooth analytical bare canoe body.  It is not a prediction for the complete
yacht: keel, rudder, appendages, viscosity, free attitude and the real transom
are excluded.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np

from wave_resistance import (
    HullGeometryWarning,
    MichellSolver,
    SolverSettings,
    dufour_39_approx_hull,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FROUDE_NUMBERS = np.linspace(0.10, 0.45, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "dufour_39_refined_analysis.png",
        help="combined geometry and resistance plot",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=SCRIPT_DIR / "dufour_39_refined_resistance.csv",
        help="resistance table",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="reuse an existing results CSV without rerunning the solver",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def solve_curve(hull, froude_numbers: np.ndarray) -> list:
    settings = SolverSettings(
        relative_tolerance=1.0e-5,
        amplitude_crosscheck=False,
    )
    solver = MichellSolver(settings)
    cases = []
    for index, froude in enumerate(froude_numbers, start=1):
        result = solver.solve(hull, [float(froude)])
        cases.append(result)
        print(
            "[{}/{}] Fn={:.3f}: Rw={:.3f} N, Cw={:.7f}, converged={}".format(
                index,
                froude_numbers.size,
                froude,
                result.wave_resistance_N[0],
                result.c_wave_resistance[0],
                bool(result.converged[0]),
            ),
            flush=True,
        )
    return cases


def write_results(path: Path, cases: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "fn",
                "speed_m_s",
                "speed_kn",
                "wave_resistance_N",
                "c_wave_resistance",
                "quadrature_error",
                "tail_fraction",
                "cancellation_ratio",
                "converged",
                "validity_flags",
            )
        )
        for case in cases:
            writer.writerow(
                (
                    "{:.6g}".format(case.fn[0]),
                    "{:.12g}".format(case.speed_m_s[0]),
                    "{:.12g}".format(1.94384449244 * case.speed_m_s[0]),
                    "{:.12g}".format(case.wave_resistance_N[0]),
                    "{:.12g}".format(case.c_wave_resistance[0]),
                    "{:.12g}".format(case.quadrature_error[0]),
                    "{:.12g}".format(case.tail_fraction[0]),
                    "{:.12g}".format(case.cancellation_ratio[0]),
                    str(bool(case.converged[0])),
                    ";".join(case.validity_flags[0]),
                )
            )


def read_results(path: Path) -> list:
    cases = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            cases.append(
                SimpleNamespace(
                    fn=np.asarray([float(row["fn"])]),
                    wave_resistance_N=np.asarray([float(row["wave_resistance_N"])]),
                    c_wave_resistance=np.asarray([float(row["c_wave_resistance"])]),
                )
            )
    if not cases:
        raise ValueError("results CSV is empty")
    return cases


def make_plot(hull, cases: list, output: Path, show: bool) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fn = np.asarray([case.fn[0] for case in cases])
    resistance = np.asarray([case.wave_resistance_N[0] for case in cases])
    coefficient = np.asarray([case.c_wave_resistance[0] for case in cases])

    figure = plt.figure(figsize=(13.2, 9.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    perspective = figure.add_subplot(grid[0, 0], projection="3d")
    body = figure.add_subplot(grid[0, 1])
    force_axis = figure.add_subplot(grid[1, 0])
    coefficient_axis = figure.add_subplot(grid[1, 1])

    x_surface, z_surface = np.meshgrid(hull.x, hull.z, indexing="ij")
    stride_x = max(1, hull.x.size // 50)
    stride_z = max(1, hull.z.size // 36)
    for sign, colour in ((-1.0, "#285f79"), (1.0, "#4a91ad")):
        perspective.plot_surface(
            x_surface,
            sign * hull.half_breadths,
            -z_surface,
            rstride=stride_x,
            cstride=stride_z,
            color=colour,
            edgecolor="#173b4c",
            linewidth=0.12,
            alpha=0.82,
            antialiased=True,
        )
    perspective.plot(
        hull.x, hull.half_breadths[:, 0], 0.0, color="#dc8b2c", lw=1.8
    )
    perspective.plot(
        hull.x, -hull.half_breadths[:, 0], 0.0, color="#dc8b2c", lw=1.8
    )
    perspective.set_box_aspect((4.8, 2.1, 1.0))
    perspective.view_init(elev=21.0, azim=-62.0)
    perspective.set_xlabel("x aft → forward (m)")
    perspective.set_ylabel("y (m)")
    perspective.set_zticks([])
    perspective.set_title("Refined smooth canoe body · 201 × 129 offsets")

    station_fractions = np.asarray((0.10, 0.20, 0.30, 0.43, 0.55, 0.70, 0.85, 0.95))
    for fraction in station_fractions:
        station = int(np.argmin(np.abs(hull.x / hull.metadata.length_ref_m - fraction)))
        section = hull.half_breadths[station]
        positive = np.flatnonzero(section > 1.0e-12)
        closure = min(int(positive[-1]) + 1, hull.z.size - 1)
        body.plot(section[: closure + 1], -hull.z[: closure + 1], lw=1.15)
        body.plot(-section[: closure + 1], -hull.z[: closure + 1], lw=1.15)
    body.axhline(0.0, color="#dc8b2c", lw=1.2)
    body.set_aspect("equal", adjustable="box")
    body.set_xlabel("y (m)")
    body.set_ylabel("z below WL (m)")
    body.set_title("Smooth transverse sections")
    body.grid(alpha=0.20)

    force_axis.plot(fn, resistance, "o-", color="#246f91", lw=1.7, ms=4)
    force_axis.fill_between(fn, 0.0, resistance, color="#246f91", alpha=0.10)
    force_axis.set_xlabel("Froude number")
    force_axis.set_ylabel("Michell wave resistance (N)")
    force_axis.set_title("Radiated-wave resistance")
    force_axis.grid(alpha=0.22)

    coefficient_axis.plot(
        fn, 1.0e3 * coefficient, "o-", color="#bd622f", lw=1.7, ms=4
    )
    coefficient_axis.set_xlabel("Froude number")
    coefficient_axis.set_ylabel(r"$10^3 C_{R_w}$")
    coefficient_axis.set_title("Wave-resistance coefficient")
    coefficient_axis.grid(alpha=0.22)

    figure.suptitle(
        "Dufour 39-inspired refined surrogate — geometry and Michell solution",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.495,
        "Screening calculation only: broad hull outside strict thin-ship validity; "
        "keel, rudder, viscosity, free attitude and real transom excluded.",
        ha="center",
        fontsize=8,
        color="#8a3f2c",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    print("Wrote {}".format(output), flush=True)
    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HullGeometryWarning)
        hull = dufour_39_approx_hull()
    if args.plot_only:
        cases = read_results(args.results_csv)
    else:
        cases = solve_curve(hull, DEFAULT_FROUDE_NUMBERS)
        write_results(args.results_csv, cases)
    make_plot(hull, cases, args.output, args.show)
    if not args.plot_only:
        print("Wrote {}".format(args.results_csv), flush=True)


if __name__ == "__main__":
    main()
