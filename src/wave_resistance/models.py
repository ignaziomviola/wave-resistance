"""Configuration and result types for the free-surface panel solver."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WaterProperties:
    """Physical properties used when dimensional results are requested."""

    density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if not math.isfinite(self.density_kg_m3) or self.density_kg_m3 <= 0.0:
            raise ValueError("density_kg_m3 must be finite and positive")
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be finite and positive")


@dataclass(frozen=True)
class MeshSettings:
    """Discretisation settings for the hull and half free-surface domain."""

    hull_stations: int = 41
    hull_vertical_points: int = 15
    free_surface_x_points: int = 25
    free_surface_y_points: int = 10
    upstream_lengths: float = 1.0
    downstream_lengths: float = 2.5
    lateral_lengths: float = 1.25
    waterline_gap_cells: float = 0.75
    hull_source_offset: float = 0.35
    free_surface_source_offset: float = 0.60

    def __post_init__(self) -> None:
        if self.hull_stations < 7 or self.hull_vertical_points < 5:
            raise ValueError("hull mesh requires at least 7 x 5 points")
        if self.free_surface_x_points < 13 or self.free_surface_y_points < 5:
            raise ValueError("free-surface mesh requires at least 13 x 5 points")
        if min(self.upstream_lengths, self.downstream_lengths, self.lateral_lengths) <= 0:
            raise ValueError("free-surface extents must be positive")
        if self.waterline_gap_cells < 0.0:
            raise ValueError("waterline_gap_cells cannot be negative")
        if self.hull_source_offset <= 0.0 or self.free_surface_source_offset <= 0.0:
            raise ValueError("source offsets must be positive")


@dataclass(frozen=True)
class SolverSettings:
    """Numerical settings and declared applicability envelope."""

    fn_min: float = 0.15
    fn_max: float = 0.45
    linear_residual_tolerance: float = 1.0e-8
    condition_warning: float = 1.0e12
    direct_unknown_limit: int = 1400
    gmres_relative_tolerance: float = 1.0e-9
    gmres_max_iterations: int = 800
    kernel_chunk_size: int = 256
    sponge_strength: float = 0.04
    sponge_start_fraction: float = 0.72
    reject_outside_envelope: bool = True
    include_wave_pattern_diagnostic: bool = True
    wave_cut_points: int = 512
    kochin_integration_points: int = 6001
    kochin_t_max: float = 60.0
    kochin_tail_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        if not (0.0 < self.fn_min < self.fn_max):
            raise ValueError("invalid Froude-number envelope")
        if self.linear_residual_tolerance <= 0.0:
            raise ValueError("linear_residual_tolerance must be positive")
        if self.condition_warning <= 1.0:
            raise ValueError("condition_warning must exceed one")
        if self.direct_unknown_limit < 1 or self.gmres_max_iterations < 1:
            raise ValueError("invalid linear-solver limits")
        if self.gmres_relative_tolerance <= 0.0 or self.kernel_chunk_size < 1:
            raise ValueError("invalid GMRES or kernel settings")
        if self.sponge_strength < 0.0:
            raise ValueError("sponge_strength cannot be negative")
        if not (0.0 < self.sponge_start_fraction < 1.0):
            raise ValueError("sponge_start_fraction must lie in (0, 1)")
        if self.wave_cut_points < 64:
            raise ValueError("wave_cut_points must be at least 64")
        if self.kochin_integration_points < 1001 or self.kochin_t_max <= 5.0:
            raise ValueError("invalid Kochin quadrature settings")
        if self.kochin_tail_tolerance <= 0.0:
            raise ValueError("kochin_tail_tolerance must be positive")


@dataclass(frozen=True)
class CaseDiagnostics:
    """Numerical evidence attached to one operating point."""

    unknown_count: int
    matrix_condition: float
    relative_residual: float
    solver_backend: str
    solver_iterations: int
    double_body_drag_coefficient: float
    kochin_wave_coefficient: float
    kochin_tail_fraction: float
    near_field_pressure_coefficient: float
    wave_cut_energy_coefficient: float
    pressure_kochin_difference: float
    minimum_points_per_transverse_wavelength: float
    flags: Tuple[str, ...] = ()


@dataclass
class WaveResistanceResult:
    """Resistance curve, convergence evidence, and optional wave fields."""

    froude_number: np.ndarray
    speed_m_s: np.ndarray
    wave_resistance_N: np.ndarray
    wave_resistance_coefficient: np.ndarray
    near_field_pressure_coefficient: np.ndarray
    wave_cut_energy_coefficient: np.ndarray
    diagnostics: Tuple[CaseDiagnostics, ...]
    length_ref_m: float
    wetted_area_m2: float
    water: WaterProperties
    wave_fields: Optional[Dict[float, Dict[str, np.ndarray]]] = None

    def __post_init__(self) -> None:
        arrays = (
            self.froude_number,
            self.speed_m_s,
            self.wave_resistance_N,
            self.wave_resistance_coefficient,
            self.near_field_pressure_coefficient,
            self.wave_cut_energy_coefficient,
        )
        size = len(np.asarray(self.froude_number))
        if size == 0 or any(len(np.asarray(item)) != size for item in arrays):
            raise ValueError("all result arrays must be non-empty and equally sized")
        if len(self.diagnostics) != size:
            raise ValueError("diagnostics must contain one record per operating point")

    @property
    def converged(self) -> np.ndarray:
        """Return a Boolean convergence mask derived from diagnostic flags."""

        return np.asarray(
            [
                diagnostic.relative_residual <= 1.0e-8
                and diagnostic.kochin_tail_fraction <= 1.0e-3
                for diagnostic in self.diagnostics
            ],
            dtype=bool,
        )

    def to_csv(self, path: Path) -> None:
        """Write the scalar curve and numerical diagnostics to CSV."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                [
                    "fn",
                    "speed_m_s",
                    "wave_resistance_N",
                    "c_wave_pressure",
                    "c_wave_kochin",
                    "c_wave_cut_energy",
                    "unknown_count",
                    "matrix_condition",
                    "relative_residual",
                    "solver_backend",
                    "solver_iterations",
                    "double_body_drag_coefficient",
                    "kochin_tail_fraction",
                    "pressure_kochin_difference",
                    "minimum_points_per_transverse_wavelength",
                    "flags",
                ]
            )
            for index, diagnostic in enumerate(self.diagnostics):
                writer.writerow(
                    [
                        float(self.froude_number[index]),
                        float(self.speed_m_s[index]),
                        float(self.wave_resistance_N[index]),
                        float(self.near_field_pressure_coefficient[index]),
                        float(self.wave_resistance_coefficient[index]),
                        float(self.wave_cut_energy_coefficient[index]),
                        diagnostic.unknown_count,
                        diagnostic.matrix_condition,
                        diagnostic.relative_residual,
                        diagnostic.solver_backend,
                        diagnostic.solver_iterations,
                        diagnostic.double_body_drag_coefficient,
                        diagnostic.kochin_tail_fraction,
                        diagnostic.pressure_kochin_difference,
                        diagnostic.minimum_points_per_transverse_wavelength,
                        ";".join(diagnostic.flags),
                    ]
                )

    def diagnostics_to_json(self, path: Path) -> None:
        """Write machine-readable solver and applicability evidence."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "length_ref_m": self.length_ref_m,
            "wetted_area_m2": self.wetted_area_m2,
            "water": asdict(self.water),
            "cases": [asdict(item) for item in self.diagnostics],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def wave_fields_to_npz(self, path: Path) -> None:
        """Write all retained wave cuts and free-surface fields."""

        if self.wave_fields is None:
            raise ValueError("wave fields were not retained for this result")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for froude, fields in self.wave_fields.items():
            prefix = "fn_{:.6f}".format(froude).replace(".", "p")
            for name, values in fields.items():
                arrays[f"{prefix}_{name}"] = np.asarray(values)
        np.savez_compressed(target, **arrays)
