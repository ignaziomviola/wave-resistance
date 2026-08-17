# Wave resistance of displacement monohulls

`wave-resistance` contains three deliberately separate research methods:

- `LinearFreeSurfaceSolver`, the existing fast three-dimensional linear screening method;
- `DoubleBodyPotentialFlowSolver` and `LinearPotentialFlowSolver`, dense exact-body Rankine-panel reference solvers;
- `NonlinearPotentialFlowSolver`, a restricted fixed-attitude nonlinear solver whose released mode keeps the design waterline fixed.

The legacy methods and their public API are unchanged. No method is silently substituted for another.

## Installation and checks

```bash
python -m pip install -e '.[dev]'
python -m pytest
python examples/nonlinear_wigley.py
```

NumPy and SciPy are core dependencies. Matplotlib is optional and is used only
by examples. The dense BEM implementation is intended for reproducible
verification-scale calculations, not production meshes.

## Nonlinear quick start

```python
from wave_resistance import (
    BEMSettings,
    GeometrySettings,
    NonlinearPotentialFlowSolver,
    wigley_hull,
)

hull = wigley_hull(length_m=1.0, nx=7, nz=5)
result = NonlinearPotentialFlowSolver(
    hull,
    geometry=GeometrySettings(free_surface_nx=13, free_surface_ny=9),
    bem=BEMSettings(quadrature_order=8),
).solve()

print(result.status)
print(result.failure_reasons)
result.to_json("summary.json")
result.to_npz("fields.npz")
```

The nonlinear workflow starts from the target-speed linear solution, moves the
interior graph nodes, and advances a linear-to-exact homotopy. Its
`fixed_waterline_nonlinear` mode is explicitly an approximate waterline
treatment; it is not an exact moving-waterline method.

## Scope

The exact-body solvers support steady, inviscid, incompressible, irrotational,
deep-water flow around smooth, symmetric displacement monohulls at fixed
sinkage and trim. They report wave-making resistance only. Viscosity, total
resistance, finite depth, appendages, propulsion, dynamic attitude, transoms,
multihulls, asymmetric hulls, CAD import, surface tension, separation,
ventilation, spray, overturning and breaking waves are excluded.

`algebraic_converged`, `free_surface_converged`,
`force_balance_converged`, `mesh_converged`, and `domain_converged` are
independent. `accepted` is true only when every required gate passes. An
invalid mesh, non-graph surface, excessive slope, incompatible system,
continuation failure, or failed pressure/far-field balance produces an explicit
failure reason and an unaccepted result.

See [the nonlinear solver contract](docs/NONLINEAR_SOLVER_DESIGN.md),
[methodology](docs/METHODOLOGY.md), and [validation notes](docs/VALIDATION.md).

## Existing linear workflow

```python
import numpy as np
from wave_resistance import LinearFreeSurfaceSolver, image_inspired_yacht

result = LinearFreeSurfaceSolver().solve(
    image_inspired_yacht(),
    np.arange(0.20, 0.451, 0.05),
    retain_wave_fields=True,
)
print(result.wave_resistance_coefficient)
```

Coordinates in the existing public `HullOffsets` API are body-fixed: `x`
increases from bow to stern with the incident stream, `y` is starboard, and
`z` is upward. The new solver retains this convention to preserve API
compatibility; the design document gives the sign transformation from the
requested aft-to-forward/downward convention.

This is research software. Inspect the convergence fields and perform mesh and
domain studies before interpreting a computed force.
