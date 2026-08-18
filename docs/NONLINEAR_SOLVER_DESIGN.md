# Restricted exact-body Rankine-panel solver: mathematical contract

## Scope and coordinates

This document fixes the contract for formulation version `rankine-source-v1`.
The public coordinates follow the offset convention: (x) is aft-to-forward,
(y) is starboard, and (z) is positive down from the calm waterplane.  A
ship advancing in (+x) at speed (U) is stationary in a uniform stream
(oldsymbol U_\infty=(-U,0,0)).  The total potential is

\[
 \Phi=-Ux+\phi,\qquad \boldsymbol v=\nabla\Phi .
\]

Pressure force on the body is
(oldsymbol F=-\int_{S_H}p\boldsymbol n\,dS), where the hull normal points
from the body into the fluid.  Positive resistance opposes ship motion:
(R=-F_x=\int_{S_H}p n_x\,dS).  SI units are used.  Reported coefficients use
(C_W=R/(\rho U^2S_0/2)), with the legacy metadata wetted area (S_0); the
triangulated (and, when relevant, nonlinear) wetted area is also reported.

## Boundary integral and gauge

The reference implementation is a constant-strength, indirect Rankine-source
formulation,

\[
 \phi(P)=\int_S {\sigma(Q)\over4\pi|P-Q|}\,dS_Q .
\]

The exterior, fluid-side normal derivative supplies the usual (-\sigma/2)
jump.  The same single-layer representation is used on every boundary; source,
dipole, and direct-potential equations are not mixed.  Exact polygon edge and
solid-angle expressions are used for panel velocity.  Potential self terms use
a Duffy transformation; adjacent and near terms use adaptive triangular
quadrature.  Centroid sources are permitted only beyond a configurable
distance and can be disabled.  The potential gauge is fixed by the specified
uniform stream and decaying source representation.  Before solution, the
Neumann data are projected to zero area-weighted flux on closed double-body
meshes; the removed incompatibility and the total source flux are reported.

The double-body calculation mirrors the wetted hull in (z=0), producing a
closed body in an unbounded fluid.  The physical output is restricted to the
lower, (z\geq0), wetted half.

## Free surface and truncated domain

The free surface is a conforming triangular graph (z=\eta(x,y)).  Its inner
edge shares the offset-hull waterline.  The finite patch extends upstream,
downstream, and laterally in multiples of hull length.  Hull and free-surface
impermeability are imposed through the source equation.  The outer patch uses
zero incident disturbance upstream and smoothly increasing downstream/lateral
Rayleigh damping.  Streamwise derivatives use an upwind weighted least-squares
operator.  Domain and mesh convergence are separate acceptance conditions.

On the calm plane, the coupled linear conditions are

\[
 \phi_z+U\eta_x=0,\qquad -U\phi_x-g\eta=0,
\]

hence (phi_z=(U^2/g)\phi_{xx}).  The implementation retains (phi) and
(eta) as coupled unknowns and therefore does not differentiate a
discontinuous panel potential twice.

On the exact graph the steady conditions are

\[
 \Phi_z-\Phi_x\eta_x-\Phi_y\eta_y=0,
 \qquad {1\over2}|\nabla\Phi|^2-g\eta={1\over2}U^2.
\]

Nonlinear continuation blends the corresponding linear and exact residuals
with (0\leq\lambda\leq1), beginning from the target-speed linear solution.
The graph is moved and redistributed after every accepted update.  Release 1
offers the explicitly named `fixed_waterline_nonlinear` mode: waterline nodes
remain at the design intersection.  It is an approximation, not an
exact-moving-waterline method.

## Radiation, pressure, and force balance

The upstream disturbance is pinned to zero.  First-order upwinding and cosine
sponges occupy configurable downstream and lateral fractions.  Refining and
extending this patch must reduce the reported upstream contamination and
domain-change measures.

Bernoulli pressure on the actual wetted panels is

\[
 p={\rho\over2}\left(U^2-|\boldsymbol v|^2\right),
\]

with hydrostatic pressure excluded from wave resistance.  A second estimate
is obtained from the solved source distribution through its deep-water Kochin
spectrum; it never reuses the legacy prescribed Michell source
(-n_x).  Acceptance requires

\[
 |R_p-R_f|\leq\max(\epsilon_r\max(|R_p|,|R_f|),\epsilon_a).
\]

## Acceptance and failure

Algebraic, free-surface, force-balance, mesh, and domain convergence are
stored independently.  `accepted` is their conjunction.  Failure returns NaN
resistance and coefficient values; it never returns a plausible force.
Unsupported statuses include non-symmetric or non-displacement geometry,
transom/dry or multiply connected waterlines, finite depth, free attitude,
multihulls, excessive graph slope/curvature, graph self-intersection, invalid
topology, singular/incompatible systems, divergent continuation, and failed
independent force balance.  The geometric graph checks are conservative:
finite elevations, positive projected panel area, a slope below the configured
limit, and a single waterline loop are all mandatory.

Residual histories contain the BEM algebraic residual, hull impermeability,
free-surface kinematic and dynamic residuals, update norm, waterline error,
geometry/remapping error, and force discrepancy.  A small linear-system
residual alone has no physical acceptance meaning.
