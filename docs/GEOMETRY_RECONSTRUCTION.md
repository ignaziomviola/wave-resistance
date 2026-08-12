# Image-inspired generic yacht geometry

## Source and scope

The source is a single raster lines plan supplied by the user. It shows a body
plan above a combined profile and plan view. The raster contains no readable
dimensions, station numbers, hydrostatics, or recoverable vector curves.
Consequently, it cannot uniquely determine physical offsets.

Only the immersed hull was reconstructed. Topsides, deck, sheer, keel, rudder,
and production details are deliberately excluded.

## Interpreted characteristics

The reconstruction preserves the clearly visible characteristics:

- pointed and relatively fine bow;
- fuller, more gradual stern run;
- maximum waterline breadth slightly aft of midships;
- smooth keel rocker with maximum draft close to amidships;
- rounded bilges and U-shaped middle sections;
- progressively finer and more V-shaped forward sections.

The source-view abscissa places the stern at `-0.5` and bow at `+0.5`. The
solver uses the opposite sign so that the body-frame `x` axis follows the
oncoming flow from bow to stern.

## Scale and equations

The independent nondimensional scale is

\[
L=1,\qquad B/L=0.28,\qquad T_c/L=0.06.
\]

Piecewise sine laws define waterline breadth and local draft, with different
forward and aft exponents. At each station, generalized-superellipse exponents
vary continuously along the hull to control bottom roundness and flare. The
surface is sampled with cosine spacing and closes on the centreline.

`examples/image_inspired_reference_targets.csv` records normalized envelope
points manually traced from the drawing. At 81 stations and 41 vertical points,
the fitted maximum deviations are:

- 0.3% of maximum half-beam for the waterline envelope;
- 0.7% of maximum draft for the keel/profile envelope.

Transverse sections were matched qualitatively because the raster overlays
stations, buttocks, and waterlines too densely for unique automatic tracing.
The generated body plan is therefore the reproducible definition of the generic
example.

## Hydrostatics

The committed metadata file records all derived quantities. With `L=1`, the
nominal values are approximately:

- displacement volume: `0.00709 L^3`;
- LCB: `0.0395 L` aft of midships;
- waterplane area: `0.1793 L^2`;
- wetted area: `0.2173 L^2`.

These values describe only the chosen generic surface. They are not estimates
for a named production yacht.

## Reproduction

Run:

```bash
MPLCONFIGDIR=/tmp/wave-resistance-mpl \
  PYTHONPATH=src python examples/image_inspired_yacht.py
```

The script regenerates offsets, metadata, resistance and convergence tables,
wave-field data, and the two documented figures.
