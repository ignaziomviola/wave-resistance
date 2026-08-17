# Exact-body Rankine solver contract, formulation version 1

## Scope and coordinates

The existing `HullOffsets` API uses a right-handed body frame in which `x`
increases from bow to stern with the incident stream, `y` is starboard, and
`z` is upward. This convention predates the exact-body solver and is retained
for API compatibility. Relative to the requested aft-to-forward,
downward-positive convention, the transformation is

\[
x_d=-x,\qquad y_d=y,\qquad z_d=-z,
\]

and vectors are transformed by `diag(-1,1,-1)`. All implementation equations
below use the public upward-positive frame. The incident velocity is
\(U\boldsymbol e_x\); the positive resistance magnitude is the force on the
body in `+x`. The total potential is
\(\Phi=Ux+\phi\).

SI units are used. Reference scales are \(L\), \(U\), \(\rho U^2\), and
\(\rho U^2L^2\), with \(Fn=U/\sqrt{gL}\). The principal coefficient uses the
legacy static bare-canoe wetted area so methods remain comparable; the meshed
wetted area is also reported.

The released physical scope is steady, inviscid, incompressible,
irrotational, deep-water flow in calm water around a smooth,
port/starboard-symmetric displacement monohull at fixed attitude. Only a
single-valued free-surface graph is permitted.

## Boundary-integral formulation

One formulation is used throughout: a constant source sheet on planar
triangles with the free-space Rankine Green function,

\[
 \phi(\boldsymbol x)=\sum_j \sigma_j
 \int_{T_j}\frac{1}{4\pi|\boldsymbol x-\boldsymbol \xi|}\,dS_\xi .
\]

With normals directed from the body into the fluid, the fluid-side normal
trace contains the `-1/2` source-sheet jump. Hull rows
enforce \(\nabla\Phi\cdot\boldsymbol n=0\). The double-body system includes a
zero-integrated-source compatibility constraint. Potential has an arbitrary
additive gauge; reported panel potential is the single-layer evaluation from
the flux-constrained source solution.

Panel integrals are not centroid sources. Duffy quadrature split about the
orthogonal projection treats self and near-singular interactions; the same
integral is used for adjacent and regular interactions. Pressure uses a
weighted local least-squares reconstruction of the tangential potential
gradient. The hull normal component is then imposed from impermeability,
avoiding an unverified derivative of discontinuous panel values.

The hull mesh mirrors the analytic offsets, merges the keel and stem seams,
and removes artificial centre-plane caps. The free-surface mesh is an explicit
triangulated graph with a waterplane cut-out. Its inner boundary shares the
offset waterline stations. Mesh construction rejects non-finite coordinates,
degenerate or duplicate faces, repeated indices, and non-manifold edges, and
reports area, aspect ratio, minimum angle, boundary, adjacency, and waterline
diagnostics.

## Free-surface conditions and radiation

On \(z=\eta(x,y)\), the exact kinematic and atmospheric Bernoulli conditions
are

\[
 \Phi_z-\Phi_x\eta_x-\Phi_y\eta_y=0,
 \qquad
 \tfrac12(|\nabla\Phi|^2-U^2)+g\eta=0.
\]

Linearisation about `z=0` gives

\[
 U\eta_x-\phi_z=0,
 \qquad U\phi_x+g\eta=0,
\]

and elimination of elevation gives
\(U^2\phi_{xx}/g+\phi_z=0\). The linear solver couples this row to hull
impermeability. A four-point backward streamwise derivative supplies the
no-incoming/upwind condition. Quadratic downstream and lateral sponge terms
damp the truncated boundary.

The nonlinear solver starts from the target-speed linear field. At each moved
geometry it solves exact hull and graph no-flux equations, evaluates the exact
Bernoulli residual, and advances a relaxed linear-to-exact homotopy. The
released mode is named `fixed_waterline_nonlinear`: interior graph nodes move,
while the design waterline and outer boundary remain fixed. This is an
explicit approximation, not a fully nonlinear moving-waterline solution.

## Pressure, far field, and acceptance

Hull pressure and force are

\[
p=\tfrac12\rho(U^2-|\nabla\Phi|^2),\qquad
\boldsymbol F=-\int_{S_H}p\boldsymbol n\,dS,
\qquad R_p=F_x.
\]

The independent far-field diagnostic is the linear wave-energy flux through a
downstream cut of the computed elevation,
\(P_w=\rho gU\int\eta^2dy/2\), with \(R_f=P_w/U\). It does not reuse the
legacy prescribed Michell distribution. Force balance passes only if

\[
 |R_p-R_f|\leq\max\{r_{tol}\max(|R_p|,|R_f|),F_{tol}\}.
\]

Algebraic, free-surface, force-balance, mesh-study, and domain-study gates are
independent. `accepted` is their conjunction. Histories separately record BEM,
kinematic, homotopy-dynamic, exact-dynamic, update, waterline, geometry, and
slope quantities. Mesh and domain gates are marked passed only when the caller
does not require those studies, or when a study wrapper supplies evidence.

Non-finite geometry, a non-manifold mesh, downward graph normals (loss of the
single-valued assumption), excessive slope, singular/incompatible systems,
continuation divergence, and failed force balance produce explicit failure
reasons and `accepted=False`.

## Explicit exclusions

The release excludes viscosity and total resistance, finite depth, dynamic
sinkage or trim, wet/dry/partial transoms, appendages, propulsion, air drag,
roughness, surface tension, multihulls, asymmetry, general CAD import,
separation, ventilation, spray, overturning, breaking waves, FMM, hierarchical
matrices, GPUs, and other production acceleration.
