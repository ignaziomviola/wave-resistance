"""Public configuration and result types for the wave-resistance solver."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WaterProperties:
    """Water density and gravitational acceleration in SI units."""

    rho_kg_m3: float = 1025.0
    g_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if not math.isfinite(self.rho_kg_m3) or self.rho_kg_m3 <= 0.0:
            raise ValueError("rho_kg_m3 must be finite and positive")
        if not math.isfinite(self.g_m_s2) or self.g_m_s2 <= 0.0:
            raise ValueError("g_m_s2 must be finite and positive")


@dataclass(frozen=True)
class SolverSettings:
    """Numerical settings and the declared validity envelope."""

    relative_tolerance: float = 1.0e-6
    absolute_tolerance: float = 1.0e-12
    min_panels_per_cycle: int = 8
    coarse_order: int = 16
    fine_order: int = 32
    max_panel_refinements: int = 8
    tail_consecutive_cycles: int = 4
    lambda_cap: float = 1.0e4
    max_phase_cycles: int = 4096
    fn_min: float = 0.10
    fn_max: float = 0.45
    allow_outside_envelope: bool = False
    amplitude_crosscheck: bool = True
    amplitude_crosscheck_rtol: float = 1.0e-10
    amplitude_crosscheck_atol: float = 1.0e-12
    spectrum_points: int = 256
    batch_chunk_size: int = 512

    def __post_init__(self) -> None:
        if self.relative_tolerance <= 0.0 or self.absolute_tolerance <= 0.0:
            raise ValueError("quadrature tolerances must be positive")
        if self.min_panels_per_cycle < 8:
            raise ValueError("min_panels_per_cycle must be at least 8")
        if self.coarse_order < 2 or self.fine_order <= self.coarse_order:
            raise ValueError("fine_order must exceed coarse_order >= 2")
        if self.max_panel_refinements < 0:
            raise ValueError("max_panel_refinements cannot be negative")
        if self.tail_consecutive_cycles < 2:
            raise ValueError("tail_consecutive_cycles must be at least 2")
        if self.lambda_cap <= 1.0:
            raise ValueError("lambda_cap must exceed 1")
        if self.max_phase_cycles < self.tail_consecutive_cycles:
            raise ValueError("max_phase_cycles is too small")
        if not (0.0 < self.fn_min < self.fn_max):
            raise ValueError("invalid Froude-number envelope")
        if self.spectrum_points < 16 or self.batch_chunk_size < 1:
            raise ValueError("invalid spectrum or batch size")


@dataclass(frozen=True)
class SpectralDensity:
    """Diagnostic Michell-integral density as a function of lambda."""

    lambda_values: np.ndarray
    d_integral_d_lambda: np.ndarray


@dataclass
class WaveResistanceResult:
    """Wave-resistance curve and numerical diagnostics."""

    fn: np.ndarray
    speed_m_s: np.ndarray
    wave_resistance_N: np.ndarray
    c_wave_resistance: np.ndarray
    integral: np.ndarray
    quadrature_error: np.ndarray
    tail_fraction: np.ndarray
    cancellation_ratio: np.ndarray
    converged: np.ndarray
    validity_flags: Tuple[Tuple[str, ...], ...]
    spectral_density: Optional[Dict[float, SpectralDensity]] = None

    def __post_init__(self) -> None:
        arrays = (
            self.fn,
            self.speed_m_s,
            self.wave_resistance_N,
            self.c_wave_resistance,
            self.integral,
            self.quadrature_error,
            self.tail_fraction,
            self.cancellation_ratio,
            self.converged,
        )
        n = len(np.asarray(self.fn))
        if any(len(np.asarray(a)) != n for a in arrays):
            raise ValueError("all result arrays must have the same length")
        if len(self.validity_flags) != n:
            raise ValueError("validity_flags must contain one entry per Froude number")

    def to_csv(self, path: Path) -> None:
        """Write scalar curve outputs and diagnostics to CSV."""

        target = Path(path)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "fn",
                    "speed_m_s",
                    "wave_resistance_N",
                    "c_wave_resistance",
                    "integral",
                    "quadrature_error",
                    "tail_fraction",
                    "cancellation_ratio",
                    "converged",
                    "validity_flags",
                ]
            )
            for i in range(len(self.fn)):
                writer.writerow(
                    [
                        float(self.fn[i]),
                        float(self.speed_m_s[i]),
                        float(self.wave_resistance_N[i]),
                        float(self.c_wave_resistance[i]),
                        float(self.integral[i]),
                        float(self.quadrature_error[i]),
                        float(self.tail_fraction[i]),
                        float(self.cancellation_ratio[i]),
                        bool(self.converged[i]),
                        ";".join(self.validity_flags[i]),
                    ]
                )


@dataclass
class BatchWaveResistanceResult:
    """Results for hulls evaluated on a common tensor grid."""

    results: Tuple[WaveResistanceResult, ...]
    operator_extensions: int = 0
