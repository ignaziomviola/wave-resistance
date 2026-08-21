# wave-resistance

Wave resistance of a bare sailing-yacht canoe body, computed by a Neumann–Kelvin panel
method, with the Delft Systematic Yacht Hull Series (DSYHS) hull Sysser 01 as the worked
case.

## Scope

The model is steady, deep-water, inviscid and irrotational flow, with a linearised
free-surface condition and the exact body boundary condition applied on the actual wetted
hull at a prescribed attitude. The output is the wave resistance of a bare canoe body.

The following are outside the model, and no attempt is made to report them: viscous and
form resistance; circulation, lift, leeway, appendages and induced resistance, none of
which a source-only hull representation carries; nonlinear free-surface effects; dynamic
sinkage and trim as a solved equilibrium; and finite depth.

Geometry is held as a full hull. A half model is mirrored once, on loading, and the fact
that it was mirrored is recorded; no code downstream of that assumes port and starboard
symmetry.

## Status

Milestones M0 (geometry and hydrostatics), M1 (formulation and the Kelvin Green function)
and M2 (constant-source panels and the Rankine solver) are complete and verified. M3, the
full Neumann–Kelvin solve, is in progress: the influence matrix, the solve, the far-field
resistance and an independent pressure route are implemented and verified against the
Michell oracle, and what remains is the convergence study on Sysser 01 itself.

Three results are worth stating here.

**A waterline line integral is not part of this formulation.** The Green function already
satisfies the free-surface condition at every point of z = 0, so a source-only
representation never forms the integral that produces one. The waterline term of the
Neumann–Kelvin literature belongs to the Green's-identity formulation. The project plan
asserted otherwise, and that assertion is withdrawn in `docs/formulation.md`.

**The resistance constant was wrong by a factor of four, and the mistake was instructive.**
It is fixed by the thin-ship limit, and that limit turns on which source density a thin
hull carries. The two faces of a hull `y = ±f` coalesce onto the centreplane, so each
carries half the sheet strength and σ → u·n_x. That is *not* the zeroth iterate
σ = 2u·n_x of the integral equation: for a thin body the two faces' mutual influence is
O(1), so dropping the integral operator is not the thin-ship limit. Anchoring the constant
on the zeroth iterate gave ρg²/(4πu⁴) instead of ρg²/(πu⁴). The test that was supposed to
catch this back-computed the constant from the oracle and divided by the shipped value, so
it only ever checked that the two were mutually consistent; it now compares the shipped
resistance against the oracle, which can fail.

**A spurious net source flux was costing an order of magnitude.** A closed body in a stream
emits no net source strength, and the two cases that verify well carry almost none —
1.0 × 10⁻⁴ of u·S on a thin Wigley hull, 1.3 × 10⁻⁴ on a submerged sphere. Sysser 01 carried
8 × 10⁻². That matters more than its size suggests: a spurious net source is a monopole whose
far-field amplitude does not fall off with λ the way a closed body's does, while the
resistance integrand λ²(λ²−1)^(−1/2) is largest exactly where the monopole lives. Solving
subject to ∫σ dS = 0 — a constrained least-squares problem with nothing to tune — takes the
Sysser 01 far-field resistance at Fn = 0.30 from 29.07 N to 2.92 N and the pressure value from
8.40 N to 0.92 N at 480 panels, **and improves the body-condition residual**, from 0.032 to
0.023 of u, at points the solve never sees. Where the flux is already small it is a null
operation: the sphere's resistance changes in the sixth significant figure. It is the default;
`--free-net-flux` recovers the plain solve.

With it, both routes converge on the same mesh sequence — +5.8 % and +2.9 % between 280 and
480 panels — where unconstrained they swung by a factor of 3.5 and were not even monotone.

**The earlier accuracy claim for the Green function measured the wrong quantity.** It
measured g_w; the influence matrix uses ∇g_w, whose integrand carries an extra sec²θ. At
(X, Y, |Z|) = (2, 0.8, 0.02) the production quadrature returned g_w to 1.7e-3 and its
gradient **185 per cent wrong**. That error put the first Sysser 01 solve at 18.5 N at
Fn = 0.30 — five per cent of displacement weight — and it looked convincingly like the known
waterline difficulty of the Neumann–Kelvin problem for a surface-piercing body. It was
quadrature: a narrow peak in the integrand where Im c vanishes, a grading that equalised
only one term of the phase, too few nodes per panel for the gradient, and a panel cap
reached without saying so. All four are fixed and each is a test. Worst gradient error
inside the envelope a Sysser mesh actually spans is now 7.0e-4, median 6e-11.

Consequently the mesh constraint previously stated here — panel centroids at least 7 mm
below the waterline at Fn = 0.3 — is **withdrawn**. The binding quantity is the oscillation
count of a panel *pair*, which grows as |y_i − y_j| / |z_i + z_j| rather than with depth
alone. It is exposed as `greens.oscillation_count`, and `nk.influence_matrix` refuses a mesh
outside the envelope rather than returning a plausible number from a coarsened grid.

Two independent routes to the resistance are implemented, and they agree on cases with an
analytic answer: on a thin Wigley hull at Fn = 0.30 with 278 panels the far-field amplitude
gives 0.936 of the Michell value and the pressure integral 0.940, within 0.4 per cent of each
other. They are not interchangeable on a coarse mesh, though. The far-field integral is
positive-definite in σ, so discretisation error can only inflate it and never cancel — five per
cent of noise on the density doubles it, while the pressure route moves by two. **Quote the
pressure route**, with the far-field value beside it as an upper bound.

On Sysser 01 at Fn = 0.30, static attitude, 480 panels, the pressure route gives 0.921 N
against a measured residuary resistance of 1.134 N, a ratio of 0.81 — and residuary resistance
is a proxy that also contains nonlinear and viscous-form contributions, so the wave resistance
it brackets is below 1.134 N. The far-field route gives 2.92 N, still carrying its one-sided
bias.

Throughput is 1164 microseconds per panel pair on a real Sysser 01 mesh, against 928 before
the kernel correction, so the fix cost 25 per cent. Against the 10-minute assembly budget
that caps the mesh at about 700 panels.

Panel influences integrate the kernel over the whole panel; centroid value times area is
used nowhere for the Rankine part. The sphere in an unbounded stream recovers the
analytically derived density 1.5 u n_x, at first order in panel size with a Richardson
extrapolant of 1.5008. First order is the correct expectation for a piecewise-constant
density, and the plan's second-order criterion for that test was mistaken.

The derivation, every verification and the errors found along the way are in
`docs/formulation.md`; milestones are in `docs/plan.md`.

## Installation

```
pip install -e ".[test]"
pytest -q
```

Requirements are Python 3.10 or later, NumPy and SciPy.

## Use

```
wave-resistance info data/SYSSER01_surface.igs
wave-resistance hydrostatics data/SYSSER01_surface.igs --displacement 0.0376136 --compare
wave-resistance hydrostatics data/SYSSER01_surface.igs --draught 0.127 --heel 20 --trim 0.5
wave-resistance nk data/SYSSER01_surface.igs --displacement 0.0376136 \
      --fn 0.30,0.40,0.45 --length 1.6 --girth 6 --stations 28
```

The reference condition is set either by a target displaced volume, for which the code
solves the flotation datum, or by an explicit draught. Prescribed sinkage, trim and heel
are offsets from that reference condition; they are not solved for. Trim and heel are in
degrees.

`nk` reports both routes to the resistance, the wave-kernel validity envelope, the largest
wavelength the mesh can carry, the net source flux and the body-condition residual at
points that are not collocation points. Read the diagnostics: on a hull as beamy as
Sysser 01 they are what tells you the far-field number is not yet converged. See
**Status** above and `docs/formulation.md` sections 16 to 20.

## What has been verified

Every figure below is produced by the test suite.

The four independent routes to the displaced volume, from the divergence theorem applied
with the fields $(0,y,0)$, $(x,0,0)$, $(0,0,z)$ and $\bm{r}/3$, agree to
3 parts in 10^15 on Sysser 01. With the free surface at $z=0$ the waterplane lid
contributes nothing to any of them, so no lid is constructed.

Hydrostatics of the analytic Wigley hull converge to the closed-form values at second
order, the error falling by a factor of four for each halving of the mesh: the relative
error in displaced volume is 5.0 × 10^-5, 1.3 × 10^-5 and 3.1 × 10^-6 on successively
refined meshes. Waterline length, beam and draught are recovered exactly. For a
rectangular box every hydrostatic quantity is exact to 1 part in 10^12.

On Sysser 01 the same second-order behaviour holds, with successive-difference ratios of
3.99 for displaced volume and for the longitudinal centre of buoyancy. Richardson
extrapolation of the sequence gives a displaced volume of 0.0376265 m^3, a wetted area of
0.6561380 m^2 and a waterplane area of 0.5584200 m^2.

Surface evaluation is checked against B-spline identities and, for the derivatives,
against the derivative patch constructed independently from differences of control
points, which agrees to 1 part in 10^11. The Wigley Michell amplitude is checked against
direct quadrature of its defining integrals over nine decades of argument, including the
crossovers into the Taylor series that replace the closed forms where cancellation would
otherwise dominate; the worst relative error is 1.5 × 10^-13.

Plane sections classify vertices lying on the cut plane explicitly. This matters rather
than being a nicety: with a strict inequality, a cut through the pole meridians of a
tessellated sphere came out 3.4 × 10^-3 low in area, against 6.4 × 10^-5 for the same cut
displaced off the vertices. Waterplanes and station cuts land on mesh vertices routinely.

## The Sysser 01 reference condition

The primary hydrostatics release, 4TU.ResearchData
[10.4121/21501375](https://doi.org/10.4121/21501375), has been downloaded and verified
against its published MD5, and every published value used here is transcribed from it with
provenance. Two conventions it settles: LCB and LCF are measured from the midpoint of the
**upright** waterline, and the heeled columns hold displacement constant, so heel requires
solving sinkage and trim rather than rotating at fixed sinkage.

Matching the published displaced volume of 0.0376136 m³ at zero trim recovers the published
waterplane area to 0.060 % and the published canoe-body draught to 0.081 %, neither of which
was a target of the solve.

Three quantities do not reconcile: waterline length is 0.52 % high, waterline beam 1.15 %
high and wetted area 2.10 % high. Attitude does not account for it — solving heave and trim
together to match the published volume and LCB converges to 0.5704° bow-up, matches both
targets, and increases all three discrepancies.

Reading the primary release removes the remaining innocent explanation: the published table
is not a transcription error, and it is internally consistent on its own numbers. What is
left is the supplied file's provenance. Its header path reads "Rhino modellen na inmeten
2012" — Rhino models after measuring, 2012 — and its surface is named "Gerebuild oppervlak",
reconstructed surface. The file is a 2012 re-measurement of the physical model; the table
describes the hull as the series was built and towed. Wetted area is the quantity most
sensitive to local surface fairness and shows the largest gap. This is carried as a stated
geometry-provenance bias in every comparison, not assumed away.

## The Sysser 01 measurements

The measurement release, 4TU.ResearchData
[10.4121/21501402](https://doi.org/10.4121/21501402), gives total resistance and the
dynamic sinkage and trim per speed. It does not give residuary resistance: that follows from
an ITTC-57 friction line at form factor zero, so **residuary resistance is a proxy for wave
resistance, not a measurement of it** — it also contains nonlinear and viscous-form
contributions.

Two features of the reduced data govern how any comparison must be read.

At Fn = 0.30 the measured residuary resistance is 1.13 N, **0.31 % of displacement weight**,
so that speed magnifies every error in a prediction. At Fn = 0.45 it is 15.5 N and 4.2 %.
Comparisons belong across the range, and Fn = 0.30 is the least informative point in it.

At Fn = 0.10 the subtraction gives −0.008 N. That negative value is kept as it comes out: it
says the ITTC-57 line slightly over-predicts this model's friction at Re = 5.9 × 10⁵, which
is information about the reduction, and clamping it would hide the one speed at which the
reduction visibly fails.

The model sinks 25 mm and trims 1.6° bow-up by Fn = 0.45, against a canoe-body draught of
127 mm, so comparisons above Fn = 0.35 must run at the measured attitude.
`measurements.sysser01_runs()` supplies it in the units `Attitude` takes.

## Data

`data/SYSSER01_surface.igs` is a DSYHS geometry file. The primary releases are the
[geometries](https://figshare.com/articles/dataset/Delft_Systematic_Yacht_Hull_Series_Geometries_data/21501330),
the [hydrostatics](https://data.4tu.nl/articles/dataset/Delft_Systematic_Yacht_Hull_Series_hydrostatics_data/21501375)
and the [measurements](https://data.4tu.nl/articles/dataset/Delft_Systematic_Yacht_Hull_Series_Measurement_Data/21501402),
all published by Delft University of Technology. Users redistributing this file should
satisfy themselves that the terms of those releases permit it.

Published reference values live in editable data files under
`src/wave_resistance/data/`, each recording its source and any unresolved ambiguity, so
that a figure can be corrected against a primary source without touching code.

## References

The IGES reader follows the IGES 5.3 specification. Basis functions follow the Cox–de
Boor recurrence as given by Piegl and Tiller, *The NURBS Book*. The Kelvin Green function
is implemented in M1 from Noblesse (1981), *Alternative integral representations for the
Green function of the theory of ship wave resistance*, and Newman (1987), *Evaluation of
the wave-resistance Green function*. The thin-ship functional retained as an oracle is
that of Michell (1898), *The wave-resistance of a ship*.
