"""Generate reproducible data and figures for the journal-style test case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from wave_resistance import (
    LinearFreeSurfaceSolver,
    MeshSettings,
    SolverSettings,
    WaterProperties,
    image_inspired_yacht,
    kochin_amplitude,
    kochin_wave_coefficient,
    kochin_wave_pattern,
    wigley_hull,
    wigley_michell_coefficient,
)

PRIMARY_FROUDE = np.round(np.arange(0.15, 0.4501, 0.025), 3)
PATTERN_FROUDE = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
GRID_LEVELS = (
    ("coarse", 31, 13),
    ("medium", 41, 15),
    ("fine", 51, 19),
)
MODEL_LENGTH_M = 10.0
WATER = WaterProperties(density_kg_m3=1025.0, gravity_m_s2=9.80665)


def _geometry(station_count: int, vertical_count: int):
    hull = image_inspired_yacht(
        station_count=81,
        vertical_count=41,
        length_ref_m=MODEL_LENGTH_M,
    )
    solver = LinearFreeSurfaceSolver(
        MeshSettings(
            hull_stations=station_count,
            hull_vertical_points=vertical_count,
            free_surface_x_points=13,
            free_surface_y_points=5,
        ),
        SolverSettings(
            wave_cut_points=64,
            kochin_integration_points=6001,
            kochin_t_max=60.0,
        ),
        WATER,
    )
    return hull, solver._prepare(hull)


def _resistance_curve(station_count: int, vertical_count: int):
    hull, geometry = _geometry(station_count, vertical_count)
    strengths = -geometry.normals[:, 0]
    coefficient = []
    tail = []
    for froude in PRIMARY_FROUDE:
        value, tail_value = kochin_wave_coefficient(
            geometry.collocation,
            geometry.areas,
            strengths,
            float(froude),
            geometry.wetted_area_over_l2,
            integration_points=6001,
            t_max=60.0,
        )
        coefficient.append(value)
        tail.append(tail_value)
    coefficient_array = np.asarray(coefficient)
    speed = PRIMARY_FROUDE * np.sqrt(WATER.gravity_m_s2 * MODEL_LENGTH_M)
    resistance = (
        0.5
        * WATER.density_kg_m3
        * speed**2
        * geometry.wetted_area_over_l2
        * MODEL_LENGTH_M**2
        * coefficient_array
    )
    return hull, geometry, coefficient_array, np.asarray(tail), speed, resistance


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _configure_plotting() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _plot_geometry(hull, geometry, output: Path) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _configure_plotting()
    figure = plt.figure(figsize=(7.05, 6.35))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.12, 1.0))
    surface_axis = figure.add_subplot(grid[0, :], projection="3d")
    profile_axis = figure.add_subplot(grid[1, 0])
    body_axis = figure.add_subplot(grid[1, 1])

    vertices = geometry.mesh.vertices / MODEL_LENGTH_M
    starboard = vertices[geometry.mesh.faces]
    port = starboard.copy()
    port[:, :, 1] *= -1.0
    polygons = np.concatenate((starboard, port), axis=0)
    collection = Poly3DCollection(
        polygons,
        facecolor="#d8e6ef",
        edgecolor="#5d7281",
        linewidth=0.20,
        alpha=0.94,
    )
    collection.set_rasterized(True)
    surface_axis.add_collection3d(collection)
    surface_axis.set_xlim(-0.52, 0.52)
    surface_axis.set_ylim(-0.18, 0.18)
    surface_axis.set_zlim(-0.075, 0.015)
    surface_axis.set_xlabel(r"$x/L$")
    surface_axis.set_ylabel(r"$y/L$")
    surface_axis.set_zlabel(r"$z/L$")
    surface_axis.view_init(elev=21, azim=-64)
    surface_axis.set_proj_type("ortho")
    surface_axis.set_box_aspect((1.0, 0.36, 0.18), zoom=1.55)
    surface_axis.grid(False)
    surface_axis.text2D(0.01, 0.96, r"(a)", transform=surface_axis.transAxes)

    x = hull.x_m / MODEL_LENGTH_M
    keel = np.min(hull.z_m / MODEL_LENGTH_M, axis=1)
    for fraction in np.linspace(0.12, 0.88, 7):
        z_line = np.asarray(
            [
                np.interp(
                    fraction,
                    np.linspace(0.0, 1.0, hull.z_m.shape[1]),
                    row / MODEL_LENGTH_M,
                )
                for row in hull.z_m
            ]
        )
        profile_axis.plot(x, z_line, color="#7693a7", linewidth=0.65)
    profile_axis.plot(x, keel, color="#173f5f", linewidth=1.6)
    profile_axis.axhline(0.0, color="0.25", linewidth=0.7)
    profile_axis.set_xlim(-0.5, 0.5)
    profile_axis.set_ylim(-0.065, 0.005)
    profile_axis.set_xlabel(r"$x/L$")
    profile_axis.set_ylabel(r"$z/L$")
    profile_axis.text(0.02, 0.92, r"(b)", transform=profile_axis.transAxes)

    station_indices = np.linspace(1, hull.x_m.size - 2, 21, dtype=int)
    for index in station_indices:
        section_y = hull.half_breadth_m[index] / MODEL_LENGTH_M
        section_z = hull.z_m[index] / MODEL_LENGTH_M
        sign = -1.0 if hull.x_m[index] < 0.0 else 1.0
        body_axis.plot(sign * section_y, section_z, color="#315d78", linewidth=0.75)
    body_axis.axvline(0.0, color="0.25", linewidth=0.7)
    body_axis.axhline(0.0, color="0.25", linewidth=0.7)
    body_axis.set_xlim(-0.15, 0.15)
    body_axis.set_ylim(-0.065, 0.005)
    body_axis.set_aspect("equal", adjustable="box")
    body_axis.set_xlabel(r"$y/L$")
    body_axis.set_ylabel(r"$z/L$")
    body_axis.text(0.02, 0.92, r"(c)", transform=body_axis.transAxes)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.99, hspace=0.20, wspace=0.27)
    figure.savefig(output)
    plt.close(figure)


def _plot_resistance(
    curves: Dict[str, np.ndarray],
    speed: np.ndarray,
    resistance: np.ndarray,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_plotting()
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.82))
    styles = {
        "coarse": ("#a1adb4", "--", 1.0),
        "medium": ("#5c7f92", "-.", 1.1),
        "fine": ("#153f5b", "-", 1.7),
    }
    for label, values in curves.items():
        color, linestyle, width = styles[label]
        axes[0].plot(
            PRIMARY_FROUDE,
            1.0e3 * values,
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=label,
        )
    axes[0].set_xlim(0.15, 0.45)
    axes[0].set_ylim(0.0, 42.0)
    axes[0].set_xticks([0.15, 0.25, 0.35, 0.45])
    axes[0].set_yticks([0, 10, 20, 30, 40])
    axes[0].set_xlabel(r"$Fn$")
    axes[0].set_ylabel(r"$10^3 C_\mathrm{W}$")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(alpha=0.20)
    axes[0].text(0.02, 0.94, r"(a)", transform=axes[0].transAxes, va="top")

    axes[1].plot(speed, resistance, color="#153f5b", linewidth=1.7)
    axes[1].scatter(speed, resistance, color="#153f5b", s=13, zorder=3)
    axes[1].set_xlim(1.4, 4.6)
    axes[1].set_ylim(0.0, 18_000.0)
    axes[1].set_xticks([1.5, 2.5, 3.5, 4.5])
    axes[1].set_yticks([0, 5_000, 10_000, 15_000])
    axes[1].set_xlabel(r"$\hat u_\infty$ [m s$^{-1}$]")
    axes[1].set_ylabel(r"$\hat R_\mathrm{W}$ [N]")
    axes[1].grid(alpha=0.20)
    axes[1].text(0.02, 0.94, r"(b)", transform=axes[1].transAxes, va="top")
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.97, wspace=0.31)
    figure.savefig(output)
    plt.close(figure)


def _plot_numerical_diagnostics(
    curves: Dict[str, np.ndarray], tails: Dict[str, np.ndarray], output: Path
) -> None:
    import matplotlib.pyplot as plt

    _configure_plotting()
    relative = 100.0 * np.abs(curves["fine"] - curves["medium"]) / np.maximum(
        curves["fine"], np.finfo(float).tiny
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))
    axes[0].plot(PRIMARY_FROUDE, relative, "o-", color="#174d6b", markersize=3.5)
    axes[0].axhline(2.0, color="0.45", linestyle="--", linewidth=0.8)
    axes[0].set_xlim(0.15, 0.45)
    axes[0].set_ylim(0.0, 4.0)
    axes[0].set_xticks([0.15, 0.25, 0.35, 0.45])
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].set_xlabel(r"$Fn$")
    axes[0].set_ylabel(r"$100|C_\mathrm{W,f}-C_\mathrm{W,m}|/C_\mathrm{W,f}$")
    axes[0].grid(alpha=0.20)
    axes[0].text(0.03, 0.93, r"(a)", transform=axes[0].transAxes, va="top")

    styles = {
        "coarse": ("0.68", "--"),
        "medium": ("0.38", "-."),
        "fine": ("0.08", "-"),
    }
    for label, values in tails.items():
        color, linestyle = styles[label]
        axes[1].semilogy(
            PRIMARY_FROUDE,
            values,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[1].axhline(1.0e-3, color="0.45", linestyle=":", linewidth=0.8)
    axes[1].set_xlim(0.15, 0.45)
    axes[1].set_ylim(1.0e-8, 2.0e-3)
    axes[1].set_xticks([0.15, 0.25, 0.35, 0.45])
    axes[1].set_xlabel(r"$Fn$")
    axes[1].set_ylabel(r"$I_\mathrm{tail}/I$")
    axes[1].grid(alpha=0.20, which="both")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].text(0.03, 0.93, r"(b)", transform=axes[1].transAxes, va="top")
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.20, top=0.97, wspace=0.34)
    figure.savefig(output)
    plt.close(figure)


def _plot_wave_patterns(geometry, output: Path) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    import matplotlib.pyplot as plt

    _configure_plotting()
    x = np.linspace(0.45, 3.0, 340)
    y = np.linspace(-1.0, 1.0, 241)
    strengths = -geometry.normals[:, 0]
    patterns: Dict[float, np.ndarray] = {}
    figure, axes = plt.subplots(3, 2, figsize=(7.05, 7.15), sharex=True, sharey=True)
    levels = np.linspace(-1.0, 1.0, 41)
    image = None
    for index, (axis, froude) in enumerate(zip(axes.flat, PATTERN_FROUDE)):
        pattern = kochin_wave_pattern(
            geometry.collocation,
            geometry.areas,
            strengths,
            float(froude),
            x,
            y,
            integration_points=1601,
            t_max=12.0,
        )
        patterns[float(froude)] = pattern
        axis.set_rasterization_zorder(1)
        image = axis.contourf(
            x,
            y,
            pattern,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
            zorder=0,
        )
        waterline_x = geometry.hull.x_m / MODEL_LENGTH_M
        waterline_y = geometry.hull.waterline_half_breadth_m / MODEL_LENGTH_M
        axis.fill_between(waterline_x, -waterline_y, waterline_y, color="0.18", zorder=4)
        axis.plot(waterline_x, waterline_y, color="white", linewidth=0.45, zorder=5)
        axis.plot(waterline_x, -waterline_y, color="white", linewidth=0.45, zorder=5)
        axis.set_xlim(-0.55, 3.0)
        axis.set_ylim(-1.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.text(0.03, 0.91, f"({chr(97 + index)}) $Fn={froude:.2f}$", transform=axis.transAxes)
        axis.set_xticks([-0.5, 0.5, 1.5, 2.5])
        axis.set_yticks([-1.0, 0.0, 1.0])
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$y/L$")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$x/L$")
    figure.subplots_adjust(left=0.08, right=0.93, bottom=0.07, top=0.99, hspace=0.08, wspace=0.08)
    colorbar_axis = figure.add_axes([0.945, 0.16, 0.016, 0.70])
    figure.colorbar(image, cax=colorbar_axis, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], label=r"$\zeta/|\zeta|_\mathrm{max}$")
    figure.savefig(output)
    plt.close(figure)
    return x, y, patterns


def _plot_spectra(geometry, output: Path) -> None:
    import matplotlib.pyplot as plt

    _configure_plotting()
    strengths = -geometry.normals[:, 0]
    figure, axis = plt.subplots(figsize=(7.05, 3.05))
    greys = np.linspace(0.76, 0.08, PATTERN_FROUDE.size)
    line_styles = ["--", "-.", ":", "--", "-.", "-"]
    for froude, grey, line_style in zip(PATTERN_FROUDE, greys, line_styles):
        t, lam, amplitude = kochin_amplitude(
            geometry.collocation,
            geometry.areas,
            strengths,
            float(froude),
            integration_points=3001,
            t_max=12.0,
        )
        density = lam * np.abs(amplitude) ** 2
        density /= max(float(np.max(density)), np.finfo(float).tiny)
        axis.plot(
            t,
            density,
            color=str(grey),
            linestyle=line_style,
            label=rf"$Fn={froude:.2f}$",
        )
    axis.set_xlim(0.0, 4.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks([0, 1, 2, 3, 4])
    axis.set_yticks([0, 0.5, 1.0])
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$\lambda |a|^2/(\lambda |a|^2)_\mathrm{max}$")
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, ncol=3, loc="upper right")
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.20, top=0.97)
    figure.savefig(output)
    plt.close(figure)


def _plot_verification(output: Path) -> Dict[str, float]:
    import matplotlib.pyplot as plt

    _configure_plotting()
    froude = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
    hull = wigley_hull(
        station_count=81,
        vertical_count=41,
        length_ref_m=1.0,
        beam_to_length=0.05,
        draft_to_length=0.0625,
    )
    solver = LinearFreeSurfaceSolver(
        MeshSettings(
            hull_stations=41,
            hull_vertical_points=15,
            free_surface_x_points=13,
            free_surface_y_points=5,
        ),
        SolverSettings(wave_cut_points=64, kochin_integration_points=6001),
    )
    geometry = solver._prepare(hull)
    calculated = []
    reference = []
    tails = []
    for value in froude:
        coefficient, tail = kochin_wave_coefficient(
            geometry.collocation,
            geometry.areas,
            -geometry.normals[:, 0],
            float(value),
            geometry.wetted_area_over_l2,
            integration_points=6001,
            t_max=60.0,
        )
        calculated.append(coefficient)
        tails.append(tail)
        reference.append(
            wigley_michell_coefficient(
                float(value),
                hull.hydrostatics.wetted_area_m2,
                beam_over_length=0.05,
                integration_points=20_001,
            )
        )
    calculated_array = np.asarray(calculated)
    reference_array = np.asarray(reference)
    relative = np.abs(calculated_array - reference_array) / reference_array

    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))
    axes[0].plot(froude, 1.0e3 * reference_array, color="0.15", linewidth=1.5, label="Michell")
    axes[0].plot(froude, 1.0e3 * calculated_array, "o--", color="#2b6f8e", markersize=4, label="panel Kochin")
    axes[0].set_xlabel(r"$Fn$")
    axes[0].set_ylabel(r"$10^3 C_\mathrm{W}$")
    axes[0].set_xticks([0.20, 0.30, 0.40])
    axes[0].grid(alpha=0.20)
    axes[0].legend(frameon=False)
    axes[0].text(0.03, 0.93, r"(a)", transform=axes[0].transAxes, va="top")
    axes[1].plot(froude, 100.0 * relative, "s-", color="#2b6f8e", markersize=4)
    axes[1].axhline(6.0, color="0.4", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel(r"$Fn$")
    axes[1].set_ylabel(r"$100|C_\mathrm{W}-C_\mathrm{W,M}|/C_\mathrm{W,M}$")
    axes[1].set_xticks([0.20, 0.30, 0.40])
    axes[1].set_ylim(0.0, 7.0)
    axes[1].set_yticks([0, 2, 4, 6])
    axes[1].grid(alpha=0.20)
    axes[1].text(0.03, 0.93, r"(b)", transform=axes[1].transAxes, va="top")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.97, wspace=0.33)
    figure.savefig(output)
    plt.close(figure)
    return {
        "maximum_absolute_relative_error": float(np.max(relative)),
        "maximum_kochin_tail_fraction": float(np.max(tails)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "article" / "data",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output_dir.resolve()
    figures = output.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    curves: Dict[str, np.ndarray] = {}
    tails: Dict[str, np.ndarray] = {}
    panels: Dict[str, int] = {}
    finest = None
    for label, station_count, vertical_count in GRID_LEVELS:
        finest = _resistance_curve(station_count, vertical_count)
        _, geometry, coefficient, tail, speed, resistance = finest
        curves[label] = coefficient
        tails[label] = tail
        panels[label] = int(geometry.collocation.shape[0])
    assert finest is not None
    hull, geometry, fine_curve, fine_tail, speed, resistance = finest

    _write_csv(
        output / "resistance_curve.csv",
        (
            "fn",
            "speed_m_s",
            "wave_resistance_N",
            "c_wave_coarse",
            "c_wave_medium",
            "c_wave_fine",
            "fine_kochin_tail_fraction",
        ),
        zip(
            PRIMARY_FROUDE,
            speed,
            resistance,
            curves["coarse"],
            curves["medium"],
            curves["fine"],
            fine_tail,
        ),
    )

    _plot_geometry(hull, geometry, figures / "hull_geometry.pdf")
    _plot_resistance(curves, speed, resistance, figures / "resistance_convergence.pdf")
    _plot_numerical_diagnostics(curves, tails, figures / "numerical_diagnostics.pdf")
    x, y, patterns = _plot_wave_patterns(geometry, figures / "wave_patterns.pdf")
    _plot_spectra(geometry, figures / "kochin_spectra.pdf")
    verification = _plot_verification(figures / "wigley_verification.pdf")

    arrays = {"x_over_l": x, "y_over_l": y}
    for froude, pattern in patterns.items():
        arrays[f"pattern_fn_{froude:.2f}".replace(".", "p")] = pattern
    np.savez_compressed(output / "wave_patterns.npz", **arrays)

    hydro = hull.hydrostatics
    relative_fine_medium = np.abs(curves["fine"] - curves["medium"]) / np.maximum(
        curves["fine"], np.finfo(float).tiny
    )
    peak_index = int(np.argmax(fine_curve))
    summary = {
        "case": {
            "hull": hull.name,
            "length_m": MODEL_LENGTH_M,
            "beam_m": hydro.maximum_waterline_beam_m,
            "draft_m": hydro.maximum_canoe_draft_m,
            "displacement_volume_m3": hydro.displacement_volume_m3,
            "displacement_mass_kg": WATER.density_kg_m3 * hydro.displacement_volume_m3,
            "wetted_area_m2": hydro.wetted_area_m2,
            "waterplane_area_m2": hydro.waterplane_area_m2,
            "longitudinal_center_of_buoyancy_from_midships_m": hydro.longitudinal_center_of_buoyancy_m,
            "beam_over_length": hydro.maximum_waterline_beam_m / MODEL_LENGTH_M,
            "draft_over_length": hydro.maximum_canoe_draft_m / MODEL_LENGTH_M,
            "volume_over_length_cubed": hydro.displacement_volume_m3 / MODEL_LENGTH_M**3,
            "water_density_kg_m3": WATER.density_kg_m3,
            "gravity_m_s2": WATER.gravity_m_s2,
        },
        "grids": {
            label: {
                "stations": station_count,
                "vertical_points": vertical_count,
                "half_hull_panels": panels[label],
                "maximum_kochin_tail_fraction": float(np.max(tails[label])),
            }
            for label, station_count, vertical_count in GRID_LEVELS
        },
        "convergence": {
            "maximum_fine_medium_relative_change_all_froude": float(np.max(relative_fine_medium)),
            "maximum_fine_medium_relative_change_fn_ge_0p20": float(
                np.max(relative_fine_medium[PRIMARY_FROUDE >= 0.20])
            ),
            "mean_fine_medium_relative_change_fn_ge_0p20": float(
                np.mean(relative_fine_medium[PRIMARY_FROUDE >= 0.20])
            ),
        },
        "principal_result": {
            "peak_froude_number": float(PRIMARY_FROUDE[peak_index]),
            "peak_c_wave": float(fine_curve[peak_index]),
            "peak_speed_m_s": float(speed[peak_index]),
            "peak_wave_resistance_N": float(resistance[peak_index]),
        },
        "verification": verification,
        "software": {
            "numpy_version": np.__version__,
            "python_script": "examples/journal_test_case.py",
            "wave_pattern_note": (
                "Kochin far-field phase reconstruction, independently normalized at each Froude number; "
                "not a dimensional free-surface elevation."
            ),
        },
    }
    (output / "test_case_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
