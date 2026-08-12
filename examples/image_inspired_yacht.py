"""Reproduce the generic image-inspired hull and its wave-resistance study."""

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
    image_inspired_yacht,
    kochin_wave_coefficient,
)


def _radiation_curve(
    hull,
    froude_numbers: np.ndarray,
    station_count: int,
    vertical_count: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    mesh = MeshSettings(
        hull_stations=station_count,
        hull_vertical_points=vertical_count,
        free_surface_x_points=13,
        free_surface_y_points=5,
    )
    settings = SolverSettings(
        wave_cut_points=64,
        kochin_integration_points=6001,
        kochin_t_max=60.0,
    )
    solver = LinearFreeSurfaceSolver(mesh, settings)
    geometry = solver._prepare(hull)
    strengths = -geometry.normals[:, 0]
    coefficients = []
    tails = []
    for froude in froude_numbers:
        coefficient, tail = kochin_wave_coefficient(
            geometry.collocation,
            geometry.areas,
            strengths,
            float(froude),
            geometry.wetted_area_over_l2,
            integration_points=settings.kochin_integration_points,
            t_max=settings.kochin_t_max,
        )
        coefficients.append(coefficient)
        tails.append(tail)
    return np.asarray(coefficients), np.asarray(tails), geometry.collocation.shape[0]


def _load_targets(path: Path) -> Dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        name: np.asarray([float(row[name]) for row in rows])
        for name in rows[0]
    }


def _plot_lines_and_curve(
    hull,
    targets: Dict[str, np.ndarray],
    froude: np.ndarray,
    curves: Dict[str, np.ndarray],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12.0, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    profile = figure.add_subplot(grid[0, 0])
    plan = figure.add_subplot(grid[0, 1])
    body = figure.add_subplot(grid[1, 0])
    resistance = figure.add_subplot(grid[1, 1])

    xi = -hull.x_m / hull.length_ref_m
    order = np.argsort(xi)
    xi_ordered = xi[order]
    keel = np.min(hull.z_m / hull.length_ref_m, axis=1)[order]
    waterline = hull.waterline_half_breadth_m[order] / hull.length_ref_m

    profile.plot(xi_ordered, keel, color="#12355b", linewidth=2.2, label="fitted keel")
    profile.scatter(
        targets["source_view_x_over_l"],
        -targets["keel_depth_over_l"],
        color="#d1495b",
        s=24,
        zorder=3,
        label="traced target",
    )
    for fraction in np.linspace(0.15, 0.85, 6):
        z_line = np.asarray(
            [np.interp(fraction, np.linspace(0.0, 1.0, hull.z_m.shape[1]), row) for row in hull.z_m]
        )[order]
        profile.plot(xi_ordered, z_line, color="#7a9cc6", linewidth=0.7)
    profile.axhline(0.0, color="0.25", linewidth=0.8)
    profile.set_title("Profile: source view (stern left, bow right)")
    profile.set_xlabel(r"source-view $x/L$")
    profile.set_ylabel(r"$z/L$")
    profile.legend(frameon=False, fontsize=8)

    plan.plot(xi_ordered, waterline, color="#12355b", linewidth=2.2)
    plan.plot(xi_ordered, -waterline, color="#12355b", linewidth=2.2)
    plan.scatter(
        targets["source_view_x_over_l"],
        targets["waterline_half_breadth_over_l"],
        color="#d1495b",
        s=22,
        zorder=3,
    )
    plan.scatter(
        targets["source_view_x_over_l"],
        -targets["waterline_half_breadth_over_l"],
        color="#d1495b",
        s=22,
        zorder=3,
    )
    for fraction in np.linspace(0.15, 0.85, 6):
        breadth = np.asarray(
            [np.interp(fraction, np.linspace(0.0, 1.0, hull.z_m.shape[1]), row) for row in hull.half_breadth_m]
        )[order]
        plan.plot(xi_ordered, breadth, color="#7a9cc6", linewidth=0.7)
        plan.plot(xi_ordered, -breadth, color="#7a9cc6", linewidth=0.7)
    plan.set_aspect("equal", adjustable="box")
    plan.set_title("Half-breadths and waterlines")
    plan.set_xlabel(r"source-view $x/L$")
    plan.set_ylabel(r"$y/L$")

    station_indices = np.linspace(2, hull.x_m.size - 3, 17, dtype=int)
    for index in station_indices:
        section_y = hull.half_breadth_m[index] / hull.length_ref_m
        section_z = hull.z_m[index] / hull.length_ref_m
        if hull.x_m[index] < 0.0:
            body.plot(section_y, section_z, color="#245a7a", linewidth=0.9)
        else:
            body.plot(-section_y, section_z, color="#8b4b6b", linewidth=0.9)
    body.axvline(0.0, color="0.25", linewidth=0.8)
    body.axhline(0.0, color="0.25", linewidth=0.8)
    body.set_aspect("equal", adjustable="box")
    body.set_title("Body plan: aft left, forward right")
    body.set_xlabel(r"$y/L$")
    body.set_ylabel(r"$z/L$")

    styles = {
        "coarse 31x13": ("#8aa1b1", "--", 1.2),
        "medium 41x15": ("#e28f41", "-.", 1.4),
        "fine 51x19": ("#12355b", "-", 2.2),
    }
    for label, values in curves.items():
        color, linestyle, width = styles[label]
        resistance.plot(froude, 1.0e3 * values, label=label, color=color, linestyle=linestyle, linewidth=width)
    resistance.set_title("Kochin wave-resistance convergence")
    resistance.set_xlabel(r"$Fn=U/\sqrt{gL}$")
    resistance.set_ylabel(r"$10^3 C_W$")
    resistance.grid(alpha=0.25)
    resistance.legend(frameon=False, fontsize=8)

    figure.suptitle("Image-inspired generic displacement yacht", fontsize=15)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_wave_field(result, froude_number: float, output: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    fields = result.wave_fields[float(froude_number)]
    x = fields["x"]
    y = fields["y"]
    elevation = fields["elevation_over_l"]
    reflected_x = np.concatenate((x, x))
    reflected_y = np.concatenate((y, -y))
    reflected_elevation = np.concatenate((elevation, elevation))
    triangulation = mtri.Triangulation(reflected_x, reflected_y)
    figure, axis = plt.subplots(figsize=(11.0, 4.5), constrained_layout=True)
    contour = axis.tricontourf(
        triangulation,
        1.0e3 * reflected_elevation,
        levels=31,
        cmap="RdBu_r",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x/L$ (flow direction)")
    axis.set_ylabel(r"$y/L$")
    axis.set_title(rf"Linear free-surface elevation at $Fn={froude_number:.2f}$")
    figure.colorbar(contour, ax=axis, label=r"$10^3\zeta/L$")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args(argv)
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    hull = image_inspired_yacht(station_count=81, vertical_count=41)
    hull.to_csv(
        output / "image_inspired_offsets.csv",
        output / "image_inspired_metadata.json",
    )
    targets = _load_targets(Path(__file__).with_name("image_inspired_reference_targets.csv"))
    froude = np.round(np.arange(0.15, 0.4501, 0.025), 3)
    curves = {}
    convergence = {}
    for label, station_count, vertical_count in (
        ("coarse 31x13", 31, 13),
        ("medium 41x15", 41, 15),
        ("fine 51x19", 51, 19),
    ):
        coefficient, tail, panel_count = _radiation_curve(
            hull, froude, station_count, vertical_count
        )
        curves[label] = coefficient
        convergence[label] = {
            "hull_panels_per_half": panel_count,
            "maximum_kochin_tail_fraction": float(np.max(tail)),
            "c_wave": coefficient.tolist(),
        }

    selected_fn = 0.35
    solver = LinearFreeSurfaceSolver(
        MeshSettings(hull_stations=41, hull_vertical_points=15),
        SolverSettings(kochin_integration_points=6001, wave_cut_points=256),
    )
    field_result = solver.solve(hull, [selected_fn], retain_wave_fields=True)
    field_result.wave_fields_to_npz(output / "image_inspired_wave_field.npz")
    field_result.diagnostics_to_json(output / "image_inspired_field_diagnostics.json")

    fine = curves["fine 51x19"]
    with (output / "image_inspired_resistance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["fn", "c_wave_kochin"])
        writer.writerows((f"{fn:.3f}", value) for fn, value in zip(froude, fine))
    convergence["froude_number"] = froude.tolist()
    relative_change = np.abs(fine - curves["medium 41x15"]) / np.maximum(fine, 1.0e-14)
    convergence["fine_vs_medium_max_relative_change_all_fn"] = float(
        np.max(relative_change)
    )
    convergence["fine_vs_medium_max_relative_change_fn_ge_0p20"] = float(
        np.max(relative_change[froude >= 0.20 - 1.0e-12])
    )
    source_x = targets["source_view_x_over_l"]
    hull_x = np.sort(-hull.x_m / hull.length_ref_m)
    hull_order = np.argsort(-hull.x_m / hull.length_ref_m)
    fitted_beam = np.interp(
        source_x,
        hull_x,
        hull.waterline_half_breadth_m[hull_order] / hull.length_ref_m,
    )
    fitted_draft = np.interp(
        source_x,
        hull_x,
        hull.local_draft_m[hull_order] / hull.length_ref_m,
    )
    convergence["geometry_fit"] = {
        "maximum_waterline_error_as_fraction_of_half_beam": float(
            np.max(np.abs(fitted_beam - targets["waterline_half_breadth_over_l"])) / 0.14
        ),
        "maximum_profile_error_as_fraction_of_draft": float(
            np.max(np.abs(fitted_draft - targets["keel_depth_over_l"])) / 0.06
        ),
    }
    (output / "image_inspired_convergence.json").write_text(
        json.dumps(convergence, indent=2), encoding="utf-8"
    )
    _plot_lines_and_curve(
        hull,
        targets,
        froude,
        curves,
        output / "image_inspired_geometry_and_resistance.png",
    )
    _plot_wave_field(
        field_result,
        selected_fn,
        output / "image_inspired_wave_field.png",
    )
    print(f"Wrote image-inspired example to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
