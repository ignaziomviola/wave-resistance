# Verification and validation status

## Terminology

This repository distinguishes:

- **verification**: whether the equations and discretization are implemented
  consistently;
- **numerical convergence**: whether results stabilize under refinement;
- **validation**: agreement with independent physical measurements or trusted
  higher-fidelity calculations.

Version 0.2 is verified and convergence-tested. It is not experimentally
validated for the generic yacht.

## Automated verification

The 19 tests cover:

- analytic Wigley volume and geometric scaling;
- closure, normals, areas, and malformed inputs;
- Rankine potential and velocity kernels;
- analytic constant-source jump term;
- free-surface waterplane exclusion;
- Kochin positivity and zero-source limit;
- dimensional force scaling at fixed Froude number;
- Froude-envelope enforcement and exports;
- coupled matrix residuals and retained wave fields.

## Wigley slender limit

An independent analytic Michell amplitude is integrated on a dense transformed
grid. For a slender Wigley hull with `B/L=0.05`, `T/L=0.0625`, and a `41 x 15`
half-hull point grid, the three-dimensional linear source/Kochin calculation
differs by less than 6% at every test point:

```text
Fn = 0.20, 0.25, 0.30, 0.35, 0.40
```

This verifies normalization, symmetry, phase, exponential depth attenuation,
and outer integration. It does not validate broad-hull physics.

## Generic-yacht grid study

The committed convergence study uses `31 x 13`, `41 x 15`, and `51 x 19` hull
point grids. The maximum `41 x 15` to `51 x 19` relative change is:

- 1.7% for `0.20<=Fn<=0.45`;
- 3.5% over the full `0.15<=Fn<=0.45` sweep.

The maximum Kochin tail fraction on the fine grid is about `5.1e-4`. The two
lowest Froude numbers should retain their explicit resolution caveat.

## Coupled-field diagnostic status

At the documented `Fn=0.35` case:

- algebraic relative residual is approximately `2e-15`;
- matrix condition estimate is approximately `1.4e3`;
- double-body drag bias is approximately `2e-5`;
- the compact free-surface grid has only about 4.1 points per fundamental
  streamwise wavelength;
- near-field pressure and finite wave-cut forces do not agree with the Kochin
  force to 5%.

The result is therefore converged with respect to the primary far-field
integral but **not independently certified by the compact near-field grid**.
The code emits flags rather than suppressing this result.

## Required future validation

Before vessel-specific use, obtain:

1. production or scanned immersed offsets at a defined loading condition;
2. static hydrostatics and wetted area from the same geometry;
3. fixed-attitude wave-pattern or wave-making resistance over the target Fn
   range, with uncertainty;
4. alternatively, a documented grid-converged free-surface CFD dataset;
5. separate dynamic-attitude data if sinkage and trim are to be introduced.

Total or residuary resistance must not be relabelled as wave-making resistance.
ITTC Procedure 7.5-02-02-04 provides the relevant wave-pattern measurement and
analysis framework.
