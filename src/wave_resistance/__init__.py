"""Michell thin-ship wave-resistance model."""

from .geometry import (
    dufour_39_approx_hull,
    GeometryDiagnostics,
    HullGeometryWarning,
    HullMetadata,
    OffsetHull,
    sailing_yacht_hull,
    wigley_hull,
)
from .models import (
    BatchWaveResistanceResult,
    SolverSettings,
    SpectralDensity,
    WaterProperties,
    WaveResistanceResult,
)
from .solver import MichellOperator, MichellSolver
from .validation import (
    HullAttitude,
    IncompatibleValidationData,
    ResistanceQuantity,
    ValidationMetrics,
    ValidationReport,
    ValidationSeries,
    compare_series,
    score_values,
)

__all__ = [
    "BatchWaveResistanceResult",
    "dufour_39_approx_hull",
    "GeometryDiagnostics",
    "HullAttitude",
    "HullGeometryWarning",
    "HullMetadata",
    "IncompatibleValidationData",
    "MichellOperator",
    "MichellSolver",
    "OffsetHull",
    "ResistanceQuantity",
    "sailing_yacht_hull",
    "SolverSettings",
    "SpectralDensity",
    "ValidationMetrics",
    "ValidationReport",
    "ValidationSeries",
    "WaterProperties",
    "WaveResistanceResult",
    "compare_series",
    "score_values",
    "wigley_hull",
]

__version__ = "0.1.0"
