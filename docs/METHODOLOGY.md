# Numerical methodology

## 1. Problem and coordinates

The vessel advances steadily in calm, inviscid, incompressible, infinitely deep
water. In the body-fixed frame the uniform flow is `U e_x`: `x` runs from bow
to stern, `y` is starboard, and `z=0` is the undisturbed free surface with the
hull at `z<=0`. All coordinates are divided by the reference waterline length
`L`; velocity potentials are divided by `UL`.

The disturbance potential satisfies Laplace's equation. The linear hull and
free-surface boundary conditions are

\[
\nabla\phi\cdot\boldsymbol n=-n_x,
\qquad Fn^2\phi_{xx}+\phi_z=0\quad(z=0),
\qquad Fn=\frac{U}{\sqrt{gL}}.
\]

Steady outgoing waves require an upstream radiation condition.

## 2. Hull representation

`HullOffsets` stores ordered half-sections. The input surface is resampled on
common cosine-spaced longitudinal and vertical coordinates, triangulated, and
reflected analytically about `y=0`. Every active panel has positive area and a
normal pointing into the starboard fluid.

Volume, LCB, waterplane area, and wetted area are calculated from the same
offsets and panels used by the solver. No empirical displacement or wetted-area
correction is applied.

## 3. Primary Rankine/Kochin resistance

The robust resistance path uses the linearized Havelock source distribution

\[
\sigma_j=-n_{x,j}
\]

on each actual three-dimensional panel. This is the Neumann-Kelvin source
approximation: it linearizes the source strength while retaining finite
breadth, depth, panel area, and transverse phase.

Let

\[
\lambda=\sqrt{1+t^2},\quad
k_x=\frac{\lambda}{Fn^2},\quad
k_y=\frac{\lambda t}{Fn^2},\quad
k=\frac{\lambda^2}{Fn^2}.
\]

For a symmetric hull, the half-hull Kochin amplitude is

\[
a(t)=-\sum_j \sigma_j\Delta S_j
\exp(kz_j-ik_xx_j)\cos(k_yy_j).
\]

The nondimensional resistance integral and area-normalized coefficient are

\[
I=\int_0^\infty\lambda|a(t)|^2dt,
\qquad
C_W=\frac{8I}{\pi Fn^4(S/L^2)}.
\]

The integral is positive definite. A quadratic `t` grid resolves the endpoint;
the last 20% of the computed integral is reported as a conservative tail
indicator. The default upper limit is `t=60` with 6001 points.

In the limit of a slender hull, `sigma dS` becomes the longitudinal derivative
of half-breadth times `dx dz`, the transverse cosine tends to one, and this
expression reduces to Michell's integral. The Wigley test checks that limit
without sharing code with the panel geometry.

## 4. Coupled free-surface diagnostic

The near-field module is deliberately independent of the primary far-field
force:

1. Constant-source triangular hull panels use the Rankine kernel
   `1/(4 pi r)` and the analytic `-1/2` single-layer normal jump.
2. Symmetry images are evaluated explicitly.
3. A half-plane free-surface grid excludes the waterplane and all four
   upstream stencil locations.
4. Free-surface sources are displaced above `z=0`, avoiding coincident source
   singularities.
5. A four-point second-order upstream stencil approximates `phi_xx`.
6. A quadratic downstream/lateral sponge term regularizes outer radiation.
7. Rows are scaled before a dense direct solve or optional SciPy GMRES.

The module returns wave elevation from the linear dynamic condition,
near-field pressure drag relative to a double-body solution, a transverse
wave-cut energy estimate, matrix condition, residual, and resolution flags.

The current compact-grid field solver is not used to overwrite the Kochin
force. Its pressure and finite-cut coefficients are retained precisely to show
whether a future field discretization has achieved independent agreement.

## 5. Numerical diagnostics

Each case reports:

- matrix unknown count, condition estimate, backend, and iterations;
- relative algebraic residual;
- double-body drag bias;
- Kochin coefficient and tail fraction;
- near-field pressure coefficient;
- transverse-cut energy coefficient;
- pressure/Kochin relative difference;
- streamwise points per fundamental wavelength;
- human-readable flags.

A small linear residual establishes only that the discrete equations were
solved. It does not establish grid convergence or physical validity.

## 6. Applicability

The declared version-0.2 range is `0.15<=Fn<=0.45`. The lower end needs finer
longitudinal resolution because wavelength scales with `Fn^2`. The method does
not represent viscosity, turbulence, flow separation, wet transoms, dynamic
attitude, breaking waves, or appendages. It should be used for research
screening and systematic comparison, followed by higher-fidelity validation.
