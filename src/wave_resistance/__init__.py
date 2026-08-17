"""Wave resistance of symmetric displacement monohulls."""

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
from .nonlinear import NonlinearPotentialFlowSolver
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
    "BEMSettings",
    "BatchWaveResistanceResult",
    "ConvergenceSettings",
    "DoubleBodyPotentialFlowSolver",
    "FreeSurfaceSettings",
    "GeometryDiagnostics",
    "GeometrySettings",
    "HullAttitude",
    "HullGeometryWarning",
    "HullMetadata",
    "IncompatibleValidationData",
    "LinearPotentialFlowSolver",
    "MichellOperator",
    "MichellSolver",
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

__version__ = "0.2.0"
