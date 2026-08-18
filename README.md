# Wave resistance

`wave-resistance` contains two deliberately separate model families.  The
legacy `MichellSolver` is the fast thin-ship screening method.  The new dense
Rankine-panel reference implementation works on the finite-breadth hull and
provides double-body, exact-body linear-free-surface, and restricted nonlinear
fixed-waterline solves.  Michell's method has not been replaced or changed.

The package predicts the **wave-making component only**. It is not a total
resistance or powering model: skin friction, viscous pressure drag,
appendages, air drag, roughness and propulsion are outside its scope.

With every coordinate divided by the same reference length, the implemented
amplitude and resistance are

\[
a(\lambda)=\iint Y_X e^{-\lambda^2 Z/Fn^2}
e^{-i\lambda X/Fn^2}\,dX\,dZ,
\qquad
I=\int_0^\infty\sqrt{1+t^2}\,
|a(\sqrt{1+t^2})|^2\,dt,
\]

\[
R_W=\frac{4\rho gL^3}{\pi Fn^2}I,
\qquad
C_{R_w}=\frac{8I}{\pi Fn^4(S/L^2)}.
\]

The cell integrals are analytic for the bilinear offset interpolant; only the
outer improper integral is evaluated numerically.

Theory references: J. H. Michell, [*The wave-resistance of a
ship*](https://doi.org/10.1080/14786449808621111), and Dambrine, Pierre and
Rousseaux, [*A theoretical and numerical determination of optimal ship forms
based on Michell's wave resistance*](https://www.numdam.org/item/10.1051/cocv/2014067.pdf).

## Installation

Python 3.9 or later and NumPy are required.

```bash
python -m pip install -e .
```

Install plotting and test dependencies when needed:

```bash
python -m pip install -e '.[plot,test]'
```

The transparent dense reference path uses NumPy; plotting remains optional.

## Exact-body Rankine-BEM quick start

```python
from wave_resistance import (
    DoubleBodyPotentialFlowSolver,
    LinearPotentialFlowSolver,
    NonlinearPotentialFlowSolver,
    wigley_hull,
)

hull = wigley_hull(length_m=1.0, nx=41, nz=17)
double_body = DoubleBodyPotentialFlowSolver().solve(hull, 0.30)
linear = LinearPotentialFlowSolver().solve(hull, 0.30)
nonlinear = NonlinearPotentialFlowSolver().solve(hull, 0.30)

print(nonlinear.accepted, nonlinear.failure_reasons)
nonlinear.to_json("summary.json")
nonlinear.to_npz("fields.npz")
```

The dense implementation is intended for verification and modest meshes.
Read [`docs/NONLINEAR_SOLVER_DESIGN.md`](docs/NONLINEAR_SOLVER_DESIGN.md)
before interpreting a result.  It fixes coordinates, signs, the indirect
single-layer formulation, free-surface equations, radiation treatment, force
definitions, and acceptance rules.

The three public solvers are:

- `DoubleBodyPotentialFlowSolver`: the finite-breadth hull mirrored in the
  calm plane, useful for kernel, flux, pressure, and d'Alembert checks;
- `LinearPotentialFlowSolver`: coupled perturbation potential and elevation on
  an explicitly meshed, damped finite free surface;
- `NonlinearPotentialFlowSolver`: exact kinematic and Bernoulli residuals on a
  moving graph, using target-speed linear initialisation and homotopy.  Release
  1 is explicitly `fixed_waterline_nonlinear`; it is not an exact moving-
  waterline method.

Every `PotentialFlowResult` separates `algebraic_converged`,
`free_surface_converged`, `force_balance_converged`, `mesh_converged`, and
`domain_converged`.  `accepted` is their conjunction.  Pressure and outer
momentum-flux resistance must agree within the mixed tolerance (2% by default,
with an absolute near-zero floor).  An unaccepted result has NaN resistance
coefficients; fatal nonlinear or geometry failure also suppresses forces.
Inspect `failure_reasons` and all residual histories.

The present dense discretisation passes its double-body reference gate.  The
checked-in coarse linear/nonlinear Wigley smoke case does **not** yet meet the
2% pressure/momentum-flux balance, and is therefore correctly unaccepted.
This remaining numerical limitation is visible rather than hidden; finer
mesh/domain convergence must be demonstrated before treating this code as a
validated nonlinear resistance predictor.

Run the reproducible workflow with:

```bash
python examples/nonlinear_wigley.py --quick --output nonlinear_wigley_output
```

It exports compact JSON summaries, compressed NPZ fields, convergence CSV,
and (when Matplotlib is installed) wave-elevation and convergence plots.

Supported physics is steady, inviscid, incompressible, irrotational flow in
deep calm water; fixed attitude; a smooth symmetric displacement monohull from
`OffsetHull`; a single-valued free-surface graph; and wave-making resistance
only.  Unsupported—and rejected rather than extrapolated—are viscous or total
resistance, appendages and propulsion, finite depth, free sinkage/trim,
transoms, separation, spray, breaking/overturning waves, ventilation, surface
tension, multihulls/asymmetry, and CAD/STL/OBJ import.

## Quick start

```python
import numpy as np

from wave_resistance import MichellSolver, wigley_hull

hull = wigley_hull(length_m=4.0, nx=161, nz=65)
fn = np.linspace(0.15, 0.45, 61)
result = MichellSolver().solve(hull, fn)

print(result.c_wave_resistance)
print(result.wave_resistance_N)
print(result.converged)
print(result.validity_flags)
result.to_csv("wigley_results.csv")
```

`WaveResistanceResult` contains, point by point:

- `fn`, `speed_m_s`, `wave_resistance_N` and `c_wave_resistance`;
- the nondimensional Michell `integral`;
- `quadrature_error`, `tail_fraction` and `cancellation_ratio` diagnostics;
- `converged` and human-readable `validity_flags`;
- optional `spectral_density` when `include_spectrum=True`.

The coefficient is

\[
C_W=\frac{R_W}{\tfrac12\rho U^2 S},
\qquad Fn=\frac{U}{\sqrt{gL}},
\]

where `L` is the hull metadata reference length and `S` its static wetted
surface area. SI units are used throughout.

## Hull geometry

`OffsetHull` represents half-breadths `y(x,z)` on a tensor grid. Coordinates
use `x=0` at the aft end, `x=L` at the forward end, and positive `z` downward
from the undisturbed waterplane. Geometry constructors validate ordering,
coverage, finite values and metadata before a solve.

The built-in canonical hull is

```python
from wave_resistance import wigley_hull

hull = wigley_hull(
    length_m=4.0,
    beam_m=0.4,       # optional; default L/10
    draft_m=0.25,     # optional; default B/1.6
    nx=161,
    nz=65,
)
print(hull.diagnostics)
```

Its half-breadth is the parabolic Wigley form

\[
y=\frac{B}{2}\left[1-\left(\frac{2x}{L}-1\right)^2\right]
  \left[1-\left(\frac{z}{T}\right)^2\right].
\]

`OffsetHull.from_csv`, `OffsetHull.from_json` and
`OffsetHull.from_irregular` support user-supplied offsets; see their
docstrings for the schema and interpolation rules.

The canonical long-form CSV is:

```text
x_m,z_m,half_breadth_m
0.0,0.0,0.0
...
```

It is accompanied by a JSON file containing:

```json
{
  "schema_version": "1.0",
  "name": "Example hull",
  "length_ref_m": 4.0,
  "length_ref_kind": "LWL",
  "wetted_area_m2": 2.38
}
```

Load both with `OffsetHull.from_csv("offsets.csv", "metadata.json")`.

## Physical and numerical limits

Michell theory assumes a slender hull, small disturbance, steady forward
motion, inviscid irrotational flow, a linear free surface and effectively
deep water. The present hull is fixed at its input waterline and attitude.
The model does not resolve dynamic sinkage and trim, transom separation,
spray, breaking waves or finite-depth effects.

The default declared envelope is `0.10 <= Fn <= 0.45`. Results outside the
envelope are rejected unless explicitly enabled in `SolverSettings`; enabled
out-of-envelope results remain flagged. A converged quadrature is not evidence
that the physical assumptions are valid, so inspect both `converged` and
`validity_flags`.

## Validation without semantic shortcuts

Towing-tank publications report several distinct resistance quantities.
`wave_resistance.validation` keeps them separate:

| `ResistanceQuantity` | Meaning | Symbol |
|---|---|---|
| `TOTAL` | measured total resistance | `C_T` |
| `RESIDUARY` | `C_T - (1+k) C_F` | `C_R` |
| `WAVE_MAKING` | wave-making force | `C_W` |
| `WAVE_PATTERN` | far-field wave-analysis result | `C_WP` |

It also records the experimental attitude:

| `HullAttitude` | ITTC code | Meaning |
|---|---|---|
| `FIXED` | `FX` | sinkage and trim fixed |
| `FREE_SINKAGE` | `FS` | free to sink only |
| `FREE_SINKAGE_TRIM` | `FR` | free to sink and trim |

These categories are not silently converted. In particular, `C_R` is not
assumed to equal `C_W`, and a fixed Michell prediction cannot be scored
against a free-running experiment. `C_WP` is also not used as an automatic
proxy for `C_W`. This prevents a numerically precise but physically invalid
score.

Load a header-based experimental CSV and compare it with a result:

```text
fn,value,quantity,uncertainty,attitude,source
0.20,0.00070,wave_making,0.00002,fixed,Tank A
0.30,0.00120,wave_making,0.00003,fixed,Tank A
```

```python
from wave_resistance.validation import (
    HullAttitude,
    ResistanceQuantity,
    ValidationSeries,
)

observed = ValidationSeries.from_csv(
    "wigley_cw.csv"
)

predicted = ValidationSeries(
    result.fn,
    result.c_wave_resistance,
    quantity=ResistanceQuantity.WAVE_MAKING,
    attitude=HullAttitude.FIXED,
    label="Michell model",
)

report = observed.compare(predicted)
print(report.to_text())
print(report.metrics.as_dict())
```

Predictions are linearly interpolated to the measured Froude numbers but are
never extrapolated. Curve metrics use trapezoidal Froude-number weights:

\[
E_2=\left[\frac{\sum_i w_i(C_i^{model}-C_i^{exp})^2}
{\sum_i w_i(C_i^{exp})^2}\right]^{1/2}.
\]

The report also gives signed bias and mean absolute error in drag counts
(`1 count = 10^-4` in coefficient), maximum absolute error and its Froude
number, and location errors for the dominant interior hump and hollow. A
relative pointwise percentage error is intentionally omitted because it is
ill-conditioned near wave-resistance hollows.

## Plotting example

The same workflow is available interactively in
[`examples/wigley_curve.ipynb`](examples/wigley_curve.ipynb).
The repository also includes the verified 71-point
[`wigley_results.csv`](examples/wigley_results.csv), its
[`verification summary`](examples/wigley_verification.json), and the resulting
[`curve`](examples/wigley_curve.png).

```bash
python examples/wigley_curve.py --output wigley_curve.png \
    --result-csv wigley_results.csv
```

To overlay licensed or locally held experimental data:

```bash
python examples/wigley_curve.py --validation wigley_cw.csv \
    --validation-column C_W --output wigley_validation.png
```

The script prints the validation report and writes the plot. Matplotlib is
imported only by the example and is not a core dependency.

## Benchmark data and provenance

No experimental data are redistributed with this repository. Widely used
legacy data are publicly readable but generally do not state an open-data
licence; free access is not permission to repackage them. Record the facility,
model length, reference-length definition, wetted area, Reynolds number,
temperature, turbulence stimulation, attitude, extraction method and source
for every imported series.

Useful primary sources include:

- [17th ITTC Resistance Committee report](https://ittc.info/media/2212/report-of-resistance-committee.pdf): multi-laboratory Wigley and Series 60 total, component and wave-pattern data.
- [ITTC benchmark list](https://www.ittc.info/media/11250/list-of-benchmarks-2.pdf): Series 60, KCS, DTMB 5415 and other canonical hulls.
- [ITTC resistance-test procedure](https://ittc.info/media/11780/75-02-02-01.pdf): coefficient definitions and test semantics.
- [ITTC uncertainty guide](https://www.ittc.info/media/9601/75-02-02-02.pdf): resistance-test uncertainty sources and propagation.
- [ITTC wave-pattern procedure](https://www.ittc.info/media/11790/75-02-02-04.pdf): wave-profile measurement and far-field analysis.

If plotted legacy values are digitised, retain that fact in `metadata` and
include digitisation uncertainty. Prefer fixed-attitude data for direct
comparison with this solver. Keep complete hull families out of calibration
when assessing cross-hull predictive performance.

## Tests

```bash
python -m pytest
```

The test suite covers analytic/reference kernels, geometry ingestion,
configuration/result contracts and validation metrics. Numerical refinement
should be judged together with the reported quadrature and tail diagnostics,
not solely by agreement with an experimental curve.
