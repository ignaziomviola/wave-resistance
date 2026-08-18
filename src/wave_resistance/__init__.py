"""Michell thin-ship wave-resistance model."""

from .geometry import (
    GeometryDiagnostics,
    HullGeometryWarning,
    HullMetadata,
    OffsetHull,
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
from .potential_flow import (
    BEMSettings,
    ConvergenceSettings,
    DoubleBodyPotentialFlowSolver,
    FreeSurfaceSettings,
    GeometrySettings,
    LinearPotentialFlowSolver,
    NonlinearSettings,
    PhysicsSettings,
    PotentialFlowResult,
)
from .nonlinear import NonlinearPotentialFlowSolver
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
    "BEMSettings",
    "ConvergenceSettings",
    "DoubleBodyPotentialFlowSolver",
    "GeometryDiagnostics",
    "GeometrySettings",
    "HullAttitude",
    "HullGeometryWarning",
    "HullMetadata",
    "IncompatibleValidationData",
    "MichellOperator",
    "MichellSolver",
    "FreeSurfaceSettings",
    "LinearPotentialFlowSolver",
    "NonlinearPotentialFlowSolver",
    "NonlinearSettings",
    "OffsetHull",
    "PhysicsSettings",
    "PotentialFlowResult",
    "ResistanceQuantity",
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
