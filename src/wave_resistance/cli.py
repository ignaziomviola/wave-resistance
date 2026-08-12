"""Command-line interface for reproducible resistance sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .geometry import HullOffsets, image_inspired_yacht, wigley_hull
from .models import MeshSettings, SolverSettings
from .solver import LinearFreeSurfaceSolver


def _froude_values(text: str) -> np.ndarray:
    parts = text.split(":")
    try:
        if len(parts) == 1:
            values = np.asarray([float(parts[0])])
        elif len(parts) == 3:
            start, stop, step = (float(part) for part in parts)
            values = np.arange(start, stop + 0.5 * step, step)
        else:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use FN or START:STOP:STEP") from exc
    if values.size == 0 or np.any(values <= 0.0):
        raise argparse.ArgumentTypeError("Froude numbers must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wave-resistance",
        description="Linear 3-D wave-resistance research solver",
    )
    parser.add_argument("--hull", choices=("image-inspired", "wigley", "csv"), default="image-inspired")
    parser.add_argument("--offsets", type=Path, help="canonical offset CSV for --hull csv")
    parser.add_argument("--metadata", type=Path, help="metadata JSON accompanying --offsets")
    parser.add_argument("--fn", type=_froude_values, default=_froude_values("0.20:0.45:0.05"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/run"))
    parser.add_argument("--hull-stations", type=int, default=41)
    parser.add_argument("--hull-vertical", type=int, default=15)
    parser.add_argument("--free-surface-x", type=int, default=25)
    parser.add_argument("--free-surface-y", type=int, default=10)
    parser.add_argument("--retain-wave-fields", action="store_true")
    return parser


def _load_hull(arguments: argparse.Namespace) -> HullOffsets:
    if arguments.hull == "image-inspired":
        return image_inspired_yacht()
    if arguments.hull == "wigley":
        return wigley_hull()
    if arguments.offsets is None:
        raise ValueError("--offsets is required for --hull csv")
    return HullOffsets.from_csv(arguments.offsets, arguments.metadata)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        hull = _load_hull(arguments)
        mesh = MeshSettings(
            hull_stations=arguments.hull_stations,
            hull_vertical_points=arguments.hull_vertical,
            free_surface_x_points=arguments.free_surface_x,
            free_surface_y_points=arguments.free_surface_y,
        )
        solver = LinearFreeSurfaceSolver(mesh_settings=mesh, solver_settings=SolverSettings())
        result = solver.solve(
            hull,
            arguments.fn,
            retain_wave_fields=arguments.retain_wave_fields,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "resistance.csv")
    result.diagnostics_to_json(output / "diagnostics.json")
    if arguments.retain_wave_fields:
        result.wave_fields_to_npz(output / "wave_fields.npz")
    hull.to_csv(output / "hull_offsets.csv", output / "hull_metadata.json")
    print(f"Wrote {len(result.froude_number)} cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
