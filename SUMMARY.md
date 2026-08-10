# Project summary

## Purpose and status

This repository provides a reusable, parameter-free Python implementation of
Michell's thin-ship theory for the steady deep-water wave-making resistance of
slender, pointed displacement monohulls. Version `0.1.0` is implemented,
packaged, tested and numerically verified. Experimental or CFD validation for a
specific vessel remains future work and will not be used to calibrate the model.

The model predicts radiated wave resistance only. It is not a total-resistance
or powering method.

## Mathematical model

Coordinates are nondimensionalised by one reference length `L`:
`X=x/L`, `Y=y/L`, and `Z=z/L`, where `Y` is half-breadth and `Z` is positive
downward from the static waterline. The implemented Michell amplitude is

\[
a(\lambda)=\iint Y_X e^{-\lambda^2 Z/Fn^2}
e^{-i\lambda X/Fn^2}\,dX\,dZ.
\]

With `lambda = sqrt(1+t^2)`, the singularity-free resistance integral is

\[
I=\int_0^\infty \sqrt{1+t^2}\,|a(\sqrt{1+t^2})|^2dt,
\quad
R_W=\frac{4\rho gL^3}{\pi Fn^2}I,
\quad
C_W=\frac{8I}{\pi Fn^4(S/L^2)}.
\]

Half-breadths are represented by a bilinear interpolant. Longitudinal and
vertical exponential moments are integrated analytically; only the outer
improper integral is evaluated numerically. Direct `Y_X` and
integration-by-parts amplitude forms provide an internal cross-check.

## Inputs

Hull offsets use a long-form CSV:

```text
x_m,z_m,half_breadth_m
```

`x` increases from aft to forward, `z=0` is the static waterline, `z` is
positive downward, and half-breadth is nonnegative. A companion JSON file must
contain:

```json
{
  "schema_version": "1.0",
  "name": "Vessel name",
  "length_ref_m": 4.0,
  "length_ref_kind": "LWL",
  "wetted_area_m2": 2.38
}
```

The geometry must be symmetric, continuous, single-valued, closed at bow,
stern and the deepest point of every section, and include the waterline.
Immersed transoms and ambiguous flat bottoms are rejected rather than altered.

## Public API

- `OffsetHull.from_csv(...)` loads and validates offsets and metadata.
- `WaterProperties(...)` defines density and gravitational acceleration.
- `SolverSettings(...)` controls tolerances and the declared operating domain.
- `MichellSolver.solve(hull, froude_numbers)` solves one hull.
- `MichellOperator.solve_batch(half_breadth_batch)` reuses a grid and
  quadrature operator for design sweeps.
- `ValidationSeries.from_csv(...)` loads like-for-like validation data.

Each result reports Froude number, speed, resistance, coefficient, quadrature
error, tail fraction, cancellation ratio, convergence state, validity flags,
and optionally the spectral density.

## Numerical safeguards

- Stable analytic bilinear-cell moments use normalized `sinc`, `expm1`, and
  small-argument series.
- Complex cell amplitudes are compensated before the global modulus is taken.
- The transformed outer integral is partitioned in wave number with at least
  eight panels per fastest bow-stern interference cycle.
- Local error is estimated by comparing 16- and 32-point Gauss-Legendre rules.
- Tail convergence requires four consecutive cycle decays and independent
  doubled-cutoff and halved-panel verification.
- Geometry-independent cell and quadrature data are cached for batch studies.

Default tolerances are `rtol=1e-6` and nondimensional `atol=1e-12`. Failure to
certify the tail before `lambda=1e4` or 4096 cycles returns
`converged=False` rather than silently accepting the result.

## Verification evidence

The test suite currently contains 58 passing tests. It covers analytic cell
moments, direct/offset agreement, malformed geometry, transom and flat-bottom
rejection, positivity, zero hull, translation, reversal and scale invariance,
oscillatory alias detection, refinement and validation metrics.

The included Wigley benchmark uses `L/B=10`, `B/T=1.6`, 81 stations, 33 depth
levels, and 71 Froude numbers from 0.10 to 0.45. All 71 points converged. Against
the analytic Wigley Michell amplitude:

- normalized L2 integral error: `4.9725e-4` (0.0497%);
- maximum relative integral error: `1.0841e-3` (0.1084%);
- maximum reported tail fraction: `9.5005e-4`.

The detailed results, verification metadata, plot and notebook are in
`examples/`.

## Installation and execution

```bash
python -m pip install -e .
python -m pip install -e '.[plot,test]'
python -m pytest
python examples/wigley_curve.py --output wigley_curve.png \
  --result-csv wigley_results.csv
```

The computational core requires Python 3.9+ and NumPy only. Matplotlib is an
optional plotting dependency; pytest and mpmath are development dependencies.

## Applicability and exclusions

The default accepted range is `0.10 <= Fn <= 0.45`. The solver assumes calm,
effectively infinite-depth water; steady speed; fixed design draft and trim;
inviscid, irrotational, linearised free-surface flow; and a slender hull. It
warns when `B/L > 0.2` or slope diagnostics challenge thin-ship assumptions.

Excluded physics are viscosity, wave breaking, appendages, finite-depth
effects, multivalued bulb geometry, immersed-transom separation, dynamic
sinkage and trim, total resistance and empirical calibration. A numerically
converged result is not proof that these physical assumptions are satisfied.

## Validation policy and next data needed

Only fixed-attitude wave-resistance or wave-pattern measurements are treated
as primary like-for-like evidence. Residual and total resistance are retained
as secondary comparisons and are never relabelled as wave resistance. Reports
provide normalized L2 error, bias and MAE in drag counts, maximum error, and
hump/hollow location errors; MAPE is omitted near resistance minima.

For a vessel-specific study, add validated offsets, the companion metadata,
the target Froude-number range, water depth, and any fixed-attitude tank or CFD
data with provenance and uncertainty. No experimental data are redistributed
in this repository.

## Repository map

- `src/wave_resistance/`: computational package;
- `tests/`: automated verification suite;
- `examples/`: Wigley script, notebook, curve and numerical results;
- `README.md`: installation, API, theory and validation guide;
- `pyproject.toml`: build metadata and dependencies.

No software licence has yet been selected. Add a `LICENSE` before authorising
third-party reuse or distribution.
