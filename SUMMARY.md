# Project summary

## Status

Version 0.2 is a clean rewrite for upright displacement monohulls. It contains:

- a normalized sectioned-hull data model and triangular mesher;
- hydrostatic integration and canonical CSV/JSON exchange;
- three-dimensional Rankine source influence operators;
- a positive-definite Kochin far-field wave-resistance calculation;
- an experimental coupled hull/free-surface boundary-element field solve;
- an image-inspired generic yacht and Wigley benchmark;
- a nine-page JFM-style numerical test-case article with six figures;
- 20 automated tests and reproducible example artifacts.

## Primary model

Each starboard half-hull panel has centroid `(x_j,y_j,z_j)`, area `dS_j`, normal
`n_j`, and linear source strength `sigma_j=-n_xj`. Symmetry is applied
analytically. For `lambda=sqrt(1+t^2)`, the Kochin amplitude retains the complete
transverse phase:

\[
a(t)=-\sum_j\sigma_j\Delta S_j
e^{\lambda^2z_j/Fn^2-i\lambda x_j/Fn^2}
\cos(\lambda t y_j/Fn^2).
\]

The reported coefficient is

\[
C_W=\frac{8}{\pi Fn^4(S/L^2)}
\int_0^\infty\sqrt{1+t^2}\,|a(t)|^2dt.
\]

The formulation reduces to Michell's result in the slender limit but preserves
finite breadth, exact panel orientation, and three-dimensional interference.

## Current evidence

- The full test suite passes.
- The Wigley thin limit is within 6% of the analytic Michell coefficient at
  five Froude numbers from 0.20 to 0.40.
- The generic yacht's medium-to-fine grid change is at most 1.7% for
  `Fn>=0.20`, and 3.5% over the complete `0.15-0.45` sweep.
- The maximum retained Kochin tail is approximately `5.1e-4`.
- The source image envelope fit errors are 0.3% of half-beam and 0.7% of draft.
- The article test case provides six normalized Kochin wave-pattern maps from
  `Fn=0.20` to `0.45`, as well as the exact underlying NPZ arrays.

## Interpretation

The code estimates wave-making resistance only. Its image-inspired yacht is
nondimensional and generic, not a reconstruction of a production yacht. The
current coupled near-field field solver is diagnostic: compact-grid pressure
and wave-cut forces do not yet agree with the primary Kochin force to 5% and
are flagged. Vessel-specific use requires actual immersed geometry and
like-for-like towing-tank, wave-pattern, or CFD validation.
