"""Verify the three-dimensional Kochin calculation in the Wigley thin limit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from wave_resistance import (
    LinearFreeSurfaceSolver,
    MeshSettings,
    SolverSettings,
    kochin_wave_coefficient,
    wigley_hull,
    wigley_michell_coefficient,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args(argv)
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    hull = wigley_hull(
        station_count=161,
        vertical_count=65,
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
        SolverSettings(
            wave_cut_points=64,
            kochin_integration_points=6001,
            kochin_t_max=60.0,
        ),
    )
    geometry = solver._prepare(hull)
    strengths = -geometry.normals[:, 0]
    froude = np.round(np.arange(0.20, 0.4001, 0.025), 3)
    calculated = []
    reference = []
    tails = []
    for fn in froude:
        coefficient, tail = kochin_wave_coefficient(
            geometry.collocation,
            geometry.areas,
            strengths,
            float(fn),
            geometry.wetted_area_over_l2,
            integration_points=6001,
            t_max=60.0,
        )
        calculated.append(coefficient)
        tails.append(tail)
        reference.append(
            wigley_michell_coefficient(
                float(fn),
                hull.hydrostatics.wetted_area_m2,
                beam_over_length=0.05,
                draft_over_length=0.0625,
                integration_points=20001,
                t_max=60.0,
            )
        )
    calculated_array = np.asarray(calculated)
    reference_array = np.asarray(reference)
    relative_error = (calculated_array - reference_array) / reference_array

    with (output / "wigley_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "fn",
                "c_wave_3d_kochin",
                "c_wave_michell_reference",
                "relative_error",
                "kochin_tail_fraction",
            ]
        )
        writer.writerows(zip(froude, calculated_array, reference_array, relative_error, tails))
    verification = {
        "hull": "Wigley thin-limit benchmark",
        "beam_over_length": 0.05,
        "draft_over_length": 0.0625,
        "half_hull_grid_points": [41, 15],
        "half_hull_panel_count": int(geometry.collocation.shape[0]),
        "maximum_absolute_relative_error": float(np.max(np.abs(relative_error))),
        "maximum_kochin_tail_fraction": float(np.max(tails)),
    }
    (output / "wigley_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    axis.plot(froude, 1.0e3 * reference_array, "o-", label="Michell analytic reference")
    axis.plot(froude, 1.0e3 * calculated_array, "s--", label="3-D panel Kochin")
    axis.set_xlabel(r"$Fn=U/\sqrt{gL}$")
    axis.set_ylabel(r"$10^3 C_W$")
    axis.set_title("Wigley slender-limit verification")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output / "wigley_curve.png", dpi=220)
    plt.close(figure)
    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
