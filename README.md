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

Milestone M0, the geometry and hydrostatics layer, is complete and verified. The
Neumann–Kelvin solver is not yet implemented: the integral equation, its jump term, the
waterline term and the far-field amplitudes are derived and verified in M1 before any
solver code is written, because writing those constants from memory is how they come out
quietly wrong. Milestones are listed in `docs/`.

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
```

The reference condition is set either by a target displaced volume, for which the code
solves the flotation datum, or by an explicit draught. Prescribed sinkage, trim and heel
are offsets from that reference condition; they are not solved for.

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

Matching the published displaced volume of 0.0376136 m^3 at zero trim recovers the
published canoe-body draught of 0.127040 m to within 0.058 %, and the published
waterplane area to within 0.053 %, although neither was a target of the solve.

Three quantities do not reconcile: waterline length is 0.51 % high, waterline beam 1.11 %
high and wetted area 2.10 % high. Attitude does not account for this. Solving heave and
trim together to match the published volume and longitudinal centre of buoyancy converges
to a bow-up trim of 0.5704°, and matches both targets, but increases every one of those
three discrepancies.

The published table is internally consistent on its own numbers, so it describes a real
hull. The most likely explanation lies in the provenance of the supplied file, whose
header path reads "Rhino modellen na inmeten 2012", i.e. Rhino models after measuring,
2012, and whose surface is named "Gerebuild oppervlak", i.e. reconstructed surface. The
file is a re-measurement of the physical model made in 2012, whereas the published
hydrostatics may derive from the original 1981 lines. Wetted area is the quantity most
sensitive to local surface fairness, and it shows the largest discrepancy. Milestone M4
resolves this against the primary release; until then the discrepancy is carried as a
stated bias, not assumed away.

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
