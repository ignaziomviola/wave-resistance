# Wave resistance

`wave-resistance` is a transparent low-order implementation of Michell's
thin-ship theory for the steady wave-making resistance of a displacement
vessel in deep, calm water. It accepts a tensor-product hull-offset surface,
computes a resistance curve over Froude number, and reports numerical and
validity diagnostics alongside every result.

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

## Dufour 39 analytical surrogate

The package includes a named approximation to the current 2026 Dufour 39:

```python
from wave_resistance import dufour_39_approx_hull

hull = dufour_39_approx_hull(nx=201, nz=129)
print(hull.diagnostics)
```

The published particulars used are 12.00 m LOA, 11.27 m hull length, 10.50 m
LWL, 4.10 m maximum hull beam, 1.95 m total draft and 8600 kg unloaded
displacement. The builder also describes a chined hull with generous beam and
bow volume. Source: [Dufour Yachts, Dufour 39
specifications](https://www.dufour-yachts.com/en/sailboats/dufour-39/).

No builder offsets are public, so this is explicitly an analytical surrogate,
not a reconstruction of production geometry. It assumes `BWL=3.70 m` and an
8.00 m³ bare-canoe volume, giving a derived canoe-body draft of about 0.415 m.
The refined geometry uses 129 cosine-spaced vertical levels and continuously
varying superelliptic sections: rounded amidships and progressively more
V-shaped forward, with no artificial chine or slope discontinuity. Keel,
rudder and topsides are omitted, and the real open transom is replaced by the
pointed closure required by this Michell implementation. The resulting
`BWL/LWL≈0.35` is outside thin-ship proportions; any resistance result must be
treated as a screening estimate.

Run the three-view plot and export the canonical offsets with:

```bash
python examples/dufour_39_hull.py \
    --output examples/dufour_39_hull.png \
    --offsets-csv examples/dufour_39_offsets.csv \
    --metadata-json examples/dufour_39_metadata.json
```

The generated [hull plot](examples/dufour_39_hull.png),
[offsets](examples/dufour_39_offsets.csv) and
[metadata with provenance](examples/dufour_39_metadata.json) are included.

A conventional three-view lines plan of the same analytical canoe body is
generated with:

```bash
python examples/dufour_39_lines_plan.py
```

The vector [SVG drawing](examples/dufour_39_lines_plan.svg) and high-resolution
[PNG drawing](examples/dufour_39_lines_plan.png) contain 21 stations, eight
waterlines and four buttocks. They remain model documentation, not builder or
construction drawings.

Run the refined geometry and Michell resistance sweep with:

```bash
python examples/dufour_39_refined_analysis.py
```

This evaluates 15 speeds over `Fn=0.10:0.025:0.45` using a numerical relative
tolerance of `1e-5`. The generated [geometry and resistance
plot](examples/dufour_39_refined_analysis.png) and [result
table](examples/dufour_39_refined_resistance.csv) include convergence and tail
diagnostics. All 15 cases converge numerically. The persistent breadth and
longitudinal-slope flags mean the curve remains a thin-ship screening result,
not a validated resistance prediction for the complete yacht.

The illustrated [technical report](examples/DUFOUR_39_REFINED_REPORT.md)
documents the analytical geometry, hydrostatics, numerical settings, complete
resistance table, interpretation and limitations.

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
