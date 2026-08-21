# Neumann–Kelvin formulation

This document fixes every convention, sign and constant used by the solver. It is written
before the solver, because a constant transcribed from memory is a constant that comes out
quietly wrong. Where a quantity is *derived* here, the derivation is given in full. Where
a quantity is *fixed by verification*, the test that fixes it is named and no value is
asserted in advance.

## 1. Frame, scaling and the boundary-value problem

The frame is right-handed with the origin in the undisturbed free surface, $x$ forward,
$z$ upward, and the fluid occupying $z<0$ to infinite depth. The onset flow has speed $u$
in the $-x$ direction, so the fluid runs from bow to stern, upstream is $x \to +\infty$
and the wake trails to $x \to -\infty$. This matches the supplied geometry, whose stem is
at large $x$ and transom at small $x$.

The total velocity potential is split as $\Phi = -ux + \phi$, where $\phi$ is the
disturbance potential. The wavenumber scale is $k_0 \equiv g/u^2$, where $g$ is the
gravitational acceleration.

Let $S$ be the wetted hull surface, $\Gamma \equiv S \cap \{z=0\}$ its waterline, and
$\bm{n}$ the unit normal on $S$ directed into the fluid. The disturbance potential
satisfies

$$\nabla^2 \phi = 0 \quad \text{in the fluid}, \tag{1.1}$$

$$u^2 \phi_{xx} + g \phi_z = 0 \quad \text{on } z=0 \text{ outside the waterplane}, \tag{1.2}$$

$$\frac{\partial \phi}{\partial n} = u\, n_x \quad \text{on } S, \tag{1.3}$$

with $\nabla\phi \to 0$ as $z \to -\infty$ and no waves as $x \to +\infty$. Equation (1.3)
follows from $\nabla\Phi \cdot \bm{n} = 0$ and $\nabla(-ux)\cdot\bm{n} = -u n_x$; it is the
exact body condition, applied on the actual wetted surface, which is what distinguishes
the Neumann–Kelvin problem from thin-ship theory.

## 2. The Kelvin Green function, derived

Let $G(\text{P};\text{Q})$ be the potential at $\text{P}=(x,y,z)$ due to a unit source at
$\text{Q}=(\xi,\eta,\zeta)$ with $\zeta<0$, satisfying (1.1), (1.2) and the radiation
condition. Write it as a two-dimensional Fourier integral in the horizontal coordinates.
Using
$$\frac{1}{R} = \frac{1}{2\pi}\iint \frac{1}{k}\, \mathrm{e}^{\mathrm{i}\bm{k}\cdot\bm{x}}\,
\mathrm{e}^{-k|z-\zeta|}\,\mathrm{d}^2k , \qquad k \equiv |\bm{k}| , \tag{2.1}$$
the free-space source $-1/(4\pi R)$ becomes
$-(1/8\pi^2)\iint k^{-1}\mathrm{e}^{-k|z-\zeta|}\mathrm{e}^{\mathrm{i}\bm{k}\cdot\bm{x}}\mathrm{d}^2k$.
Add a homogeneous solution that decays as $z\to-\infty$, namely $\mathrm{e}^{kz}$, with an
undetermined amplitude $A(\bm{k})$:

$$G = -\frac{1}{8\pi^2}\iint \frac{1}{k}\left[\mathrm{e}^{-k|z-\zeta|} + A\,\mathrm{e}^{kz}\right]
\mathrm{e}^{\mathrm{i}\bm{k}\cdot\bm{x}}\,\mathrm{d}^2 k . \tag{2.2}$$

On $z=0$, which lies above the source, the bracket has value $\mathrm{e}^{k\zeta}+A$ and
$z$-derivative $k(A - \mathrm{e}^{k\zeta})$, while $\phi_{xx}$ contributes a factor
$-k_x^2$. Substituting into (1.2) and writing $k_x = k\cos\theta$,

$$-\frac{u^2 k_x^2}{k}\left(\mathrm{e}^{k\zeta}+A\right) + g\left(A - \mathrm{e}^{k\zeta}\right) = 0
\;\Longrightarrow\;
A = \mathrm{e}^{k\zeta}\,\frac{k_0 + k\cos^2\theta}{k_0 - k\cos^2\theta} . \tag{2.3}$$

The denominator vanishes on $k = k_0\sec^2\theta$, which is the Kelvin dispersion relation.
Splitting
$A = \mathrm{e}^{k\zeta}\left[-1 + 2k_0/(k_0 - k\cos^2\theta)\right]$
and undoing (2.1) term by term, the $-1$ piece is a negative image source at
$(\xi,\eta,-\zeta)$, so

$$G = -\frac{1}{4\pi R} + \frac{1}{4\pi R_1} + G_W , \qquad
G_W = -\frac{k_0}{4\pi^2}\int_0^{2\pi}\!\!\mathrm{d}\theta \int_0^\infty
\frac{\mathrm{e}^{k(z+\zeta)}\,\mathrm{e}^{\mathrm{i}kb}}{k_0 - k\cos^2\theta}\,\mathrm{d}k ,
\tag{2.4}$$

where $R$ and $R_1$ are the distances from P to the source and to its image, and
$b(\theta) \equiv (x-\xi)\cos\theta + (y-\eta)\sin\theta$. The non-wave part vanishes on
$z=0$, so it is the zero-potential image combination, not the rigid-wall one; $G_W$
supplies everything else.

### 2.1 Radiation condition and the pole

The problem (1.1)–(1.3) is even in $u$, so the direction of the wake is not yet fixed. Add
Rayleigh damping to the free-surface condition, $u^2\phi_{xx} + g\phi_z + 2\mu u \phi_x = 0$
with $\mu>0$. Repeating the algebra of (2.3), the denominator becomes
$u^2\left[k_0 - k\cos^2\theta + 2\mathrm{i}\mu\cos\theta/u\right]$, so the pole moves to

$$k = \sec^2\theta\left(k_0 + 2\mathrm{i}\mu\cos\theta/u\right) , \tag{2.5}$$

which lies above the real $k$ axis where $\cos\theta>0$ and below it where $\cos\theta<0$.
The $k$ integral therefore splits into a principal value and a residue,

$$\int_0^\infty \frac{\mathrm{e}^{-kw}}{k_0 - k\cos^2\theta}\,\mathrm{d}k
= -\sec^2\theta\left[\text{PV}\!\!\int_0^\infty \frac{\mathrm{e}^{-kw}}{k-\kappa}\,\mathrm{d}k
+ \mathrm{i}\pi s(\theta)\,\mathrm{e}^{-\kappa w}\right] , \tag{2.6}$$

with $\kappa \equiv k_0\sec^2\theta$, $w \equiv -(z+\zeta) - \mathrm{i}b$ so that
$\mathrm{Re}\,w>0$, and $s(\theta) = \pm 1$ according to the side the pole passes. The
principal value is the local disturbance; the residue is the free-wave field.

**The sign of $s$ is fixed by verification, not asserted here.** Which sign of the damping
term corresponds to a stern-going wake depends on a convention that is easy to invert on
paper, so the implementation evaluates $G_W$ at large $|x-\xi|$ both upstream and
downstream and keeps the choice for which the wave amplitude vanishes as
$x-\xi \to +\infty$. That is test V5b.

## 3. Representation, and why there is no waterline integral

Represent the disturbance by a source density $\sigma$ on the hull alone,

$$\phi(\text{P}) = \iint_S \sigma(\text{Q})\, G(\text{P};\text{Q})\,\mathrm{d}S_\text{Q} . \tag{3.1}$$

Because $G$ satisfies (1.2) at every point of $z=0$ for any source strictly below it, so
does $\phi$. The free-surface condition is therefore satisfied identically by
construction, on the whole plane $z=0$ and not merely on the part outside the waterplane.
Nothing remains to be imposed there.

**A waterline line integral is consequently not part of this formulation.** The waterline
term familiar from the Neumann–Kelvin literature arises in the *potential*, or
Green's-identity, formulation: applying Green's second identity to $\phi$ and $G$ over the
fluid boundary produces a free-surface integral which the free-surface condition converts
into a line integral around $\Gamma$. A source-only representation with a Green function
that already satisfies the free-surface condition never forms that integral. The earlier
project plan asserted that a waterline term would be required; that assertion was wrong
for the representation chosen here, and is withdrawn.

One genuine consequence does survive, and it is a discretisation matter rather than a
missing term. Equation (1.2) holds for sources strictly below $z=0$; the wetted surface
reaches $z=0$ along $\Gamma$, where $G_W$ has a logarithmic singularity. The
discretisation therefore keeps every source panel strictly below the free surface, and the
residual of (1.2) at field points on $z=0$ approaching $\Gamma$ is measured and reported
rather than assumed small. That is test V6.

Taking the normal derivative of (3.1) and letting P approach $S$ from the fluid gives the
integral equation

$$\tfrac{1}{2}\sigma(\text{P}) + \text{PV}\!\!\iint_S \sigma(\text{Q})\,
\frac{\partial G(\text{P};\text{Q})}{\partial n(\text{P})}\,\mathrm{d}S_\text{Q}
= u\,n_x(\text{P}) , \qquad \text{P} \in S . \tag{3.2}$$

The coefficient $+\tfrac{1}{2}$ follows from the kernel convention. A point source with
potential $-1/(4\pi R)$ has radial velocity $+1/(4\pi R^2)$ and hence unit outward flux, so
a uniform sheet of density $\sigma$ on a plane produces $\phi = \sigma|z|/2$ and a normal
derivative $+\sigma/2$ on the side into which $\bm{n}$ points. The sign is nevertheless
confirmed independently by test V3, the sphere in an unbounded stream, which fails
visibly if either the jump coefficient or the panel integration is wrong.

## 4. Far-field amplitudes, both branches

Retaining only the residue in (2.6) and substituting into (2.4), the free-wave part of the
Green function is a superposition over wave angle,

$$G_W^{\text{wave}} = \frac{\mathrm{i}k_0}{4\pi}\int \mathrm{d}\theta\; s(\theta)\sec^2\theta\;
\mathrm{e}^{k_0\sec^2\theta\,(z+\zeta)}\;
\mathrm{e}^{\mathrm{i}k_0\sec^2\theta\left[(x-\xi)\cos\theta + (y-\eta)\sin\theta\right]} . \tag{4.1}$$

The amplitude of the wave travelling at angle $\theta$ is therefore proportional to

$$A(\theta) = \iint_S \sigma(\text{Q})\,
\mathrm{e}^{k_0\sec^2\theta\,\zeta}\,
\mathrm{e}^{-\mathrm{i}k_0\sec^2\theta\left(\xi\cos\theta + \eta\sin\theta\right)}\,
\mathrm{d}S_\text{Q} . \tag{4.2}$$

The transverse phase carries $\sin\theta$, which changes sign with $\theta$. For an
asymmetric or heeled hull $A(-\theta)$ is therefore independent of $A(\theta)$, and both
must be kept. Writing $\lambda \equiv \sec\theta \ge 1$ and
$A_\pm(\lambda) \equiv A(\pm\theta)$, the wave resistance takes the form

$$R_W = C \int_1^\infty \tfrac{1}{2}\left(|A_+|^2 + |A_-|^2\right)
\frac{\lambda^2}{\sqrt{\lambda^2-1}}\,\mathrm{d}\lambda , \tag{4.3}$$

which reduces to $C\int |A|^2 \lambda^2(\lambda^2-1)^{-1/2}\mathrm{d}\lambda$ when
$A_- = A_+^*$, i.e. for a hull symmetric about $y=0$ at zero heel.

The constant $C$ follows from the thin-ship limit, and the derivation turns entirely on
which source density a thin hull carries. For a hull $y = \pm f(x,z)$ the two faces
coalesce onto $y = 0$. The linearised body condition on the starboard face is
$\varphi_y = -u f_x$, and a centreplane sheet of strength $m$ produces
$\varphi_y = \pm m/2$ at $y = 0^\pm$, so $m = -2u f_x$. Each of the two faces carries its
own density $\sigma$ and the two add on coalescence, so $2\sigma = m$ and

$$\sigma \to u\,n_x , \qquad n_x = -f_x \ \text{on the starboard face} . \tag{4.4}$$

This is emphatically **not** the zeroth iterate of (3.2), which is $\sigma = 2u\,n_x$
(§11). For a thin body the two faces are a distance $2f \to 0$ apart, so their mutual
influence through the integral operator is $O(1)$, not small; dropping that operator is
therefore not the thin-ship limit. Conflating the two is what produced an error of exactly
four in an earlier version of this document.

With (4.4) and $n_x\,\mathrm{d}S = -f_x\,\mathrm{d}x\,\mathrm{d}z$ per face, (4.2) becomes
$A \to -2u\,A_M$, where $A_M = L^2 a(\lambda)$ is Michell's amplitude in the
nondimensional form already verified in this repository. Hence
$|A|^2 \to 4u^2|A_M|^2$, and matching (4.3) to Michell's

$$R_W = \frac{4\rho g^2}{\pi u^2}\int_1^\infty |A_M|^2
\frac{\lambda^2}{\sqrt{\lambda^2-1}}\,\mathrm{d}\lambda \tag{4.5}$$

gives $4u^2 C = 4\rho g^2/(\pi u^2)$, so

$$C = \frac{\rho g^2}{\pi u^4} . \tag{4.6}$$

Michell's prefactor in (4.5) is itself anchored on published Wigley values reproduced
independently here, $10^3 C_W = 2.1413$ at $Fn = 0.300$ and $1.2362$ at $Fn = 0.345$ for
$B/L = 0.1$, $T/L = 0.0625$.

Test V2 checks (4.6) in two separate ways. Imposing (4.4) analytically and pushing it
through the far-field integral isolates the constant and the kernel from the linear solve;
it reproduces Michell to 0.12 % at 1630 panels and 0.84 % at 558 panels for $B/L = 0.02$,
$Fn = 0.30$, and converges under refinement at $Fn = 0.30$ and $0.40$ and at $B/L = 0.02$
and $0.005$. Separately, the full NK solve on thin Wigley hulls returns
$R_W/R_W^{\text{Michell}} = 1.036$ and $0.942$ at 158 and 278 panels, bracketing unity
with the first-order error in $\sigma$ that §11 quantifies. The first check verifies the
normalisation; only the second exercises the Kelvin influence matrix and the solve.

## 5. What the model does not contain

The representation (3.1) is a source distribution and therefore carries no circulation. It
cannot produce lift, and so cannot represent leeway, appendage lift or induced resistance,
at any heel angle. The free-surface condition (1.2) is linear, so wave breaking, spray and
finite-amplitude effects are absent. The attitude is prescribed; sinkage and trim are not
solved for. Depth is infinite. Viscosity is absent, so neither frictional nor form
resistance is produced, and no form factor is implied.

---

# M1 results: what the implementation established

Everything in this section was measured, not assumed. The tests that produce each number
are in `tests/test_greens.py`.

## 6. The identity that removes a quadrature level

The wavenumber integral has a closed form. Verified against direct principal-value
quadrature to 2e-16:

$$\text{PV}\!\!\int_0^\infty \frac{\mathrm{e}^{-kw}}{k-\kappa}\,\mathrm{d}k
= \mathrm{e}^{-\kappa w}\left[E_1(\kappa w) - 2\,\mathrm{Shi}(\kappa w)\right] . \tag{6.1}$$

Rescaling $p = k/\kappa$ shows the result depends on the single complex variable
$c = \kappa w$, giving $Q(c) = \text{PV}\int_0^\infty \mathrm{e}^{-pc}/(p-1)\,\mathrm{d}p$.
A second identity, verified to 3e-10 against (6.1), removes the hyperbolic sine integral
and with it the last quadrature level:

$$Q(c) = \mathrm{e}^{-c}\left[E_1(-c) - \mathrm{i}\pi\,\mathrm{sgn}(\mathrm{Im}\,c)\right] . \tag{6.2}$$

Writing $P(c) \equiv \mathrm{e}^{-c}E_1(-c)$, the integrand of (4.1) becomes
$P(c) + \mathrm{i}\pi\left[s - \mathrm{sgn}(\mathrm{Im}\,c)\right]\mathrm{e}^{-c}$, which
equals $P(c)$ wherever $\mathrm{sgn}(\mathrm{Im}\,c) = s$ and $P(c) + 2\mathrm{i}\pi s
\mathrm{e}^{-c}$ elsewhere. That switch **is** the Kelvin wake: the free-wave term
contributes only on the half of the wave-angle range where the phase condition holds. Its
real part is continuous across the switch, because $\mathrm{Im}\,c$ vanishes there and so
does $\mathrm{Im}\,\mathrm{e}^{-c}$.

$P$ is the branch whose asymptotic expansion $-\sum_{n\ge 0} n!/c^{n+1}$ is valid at any
phase. $Q$'s is not: $Q$ carries a pole term of size $\pi\mathrm{e}^{-\mathrm{Re}\,c}$
that the $1/c$ series omits, which is negligible only when $\mathrm{Re}\,c$ is large. That
distinction matters in practice, because 76 per cent of quadrature nodes on a real hull
have $|c| > 30$ and take the fast asymptotic path.

## 7. The radiation condition, settled by measurement

Section 2.1 declined to assert the branch sign. It is $s = -1$. With that choice the wave
amplitude ahead of the source is 4 per cent of the amplitude behind it over
$15 \le |X| \le 20$, and with $s = +1$ the ratio is 26 rather than 0.04. Two further
observations confirm it is the physical branch rather than an arbitrary pick: downstream
the variation decays as $|X|^{-0.47}$, which is the $x^{-1/2}$ falloff of the Kelvin waves
along the track, while upstream it decays as $|X|^{-1.9}$, i.e. there is no wave content
there at all.

An independent check ties the whole reduction back to its definition. Evaluating the raw
double integral of (2.4) with finite Rayleigh damping $\delta$, over the non-symmetrised
half range, reproduces the reduction branch for branch: $\delta<0$ matches $s=-1$,
$\delta>0$ matches $s=+1$, and the $\delta$-averaged result matches $s=0$. Integrating
instead over the full $0 \le \theta < 2\pi$ silently averages the two branches, because
the $\theta$ and $\theta+\pi$ integrands are conjugates and the damping cancels; that
average is the pole-free result and contains no waves.

## 8. The correct deep-source limit

As $k_0|z+\zeta| \to \infty$ the free-surface condition $u^2\phi_{xx} + g\phi_z = 0$ is
dominated by its second term, so it degenerates to $\phi_z = 0$ and the free surface acts
as a rigid wall. Hence

$$G \to -\frac{1}{4\pi R} - \frac{1}{4\pi R_1}, \qquad\text{i.e.}\qquad
k_0 g_w \to -\frac{1}{2\pi R_1} . \tag{8.1}$$

Measured ratio of $g_w$ to that limit: 1.027 at $|Z| = 20$, 1.010 at 50, 1.004 at 120.

The limit sometimes reached for instead — that $G$ tends to the Rankine pair at large
separation — is wrong, and the project plan asserted it. The Kelvin waves decay as
$x^{-1/2}$ along the track while the Rankine pair, being a source and its negative image,
decays as $x^{-2}$; the wave part therefore *dominates* at large separation. Measured
ratio of the wave part to the Rankine part: 6 at $X=2$, 146 at $X=10$, 6.2e4 at $X=200$.

## 9. Accuracy, and the constraint it puts on the mesh

$g_w$ diverges logarithmically as $Z \to 0$, and the quadrature follows it down only so
far. Relative error against a heavily refined evaluation, at the production setting:

| $\lvert Z\rvert$ | 2 | 1 | 0.5 | 0.2 | 0.1 | 0.05 | 0.02 | 0.005 |
|---|---|---|---|---|---|---|---|---|
| relative error | 2e-15 | 6e-13 | 3e-13 | 1e-9 | 2e-7 | 2e-4 | 1e-3 | 1e-1 |

**Consequence for the discretisation — superseded by section 15.** This table led to the
constraint $k_0|z_i + z_j| \gtrsim 0.1$, i.e. panel centroids about 7 mm below the waterline
at $Fn = 0.3$ on Sysser 01. That constraint is **withdrawn**, for two reasons. It was
derived from the accuracy of $g_w$, but the influence matrix uses $\nabla g_w$, which was
far less accurate than this table suggests; and once the gradient is fixed, the binding
quantity is the oscillation count of the *pair*, which depends on $|Y|/|Z|$ rather than on
$|Z|$ alone. Section 15 replaces both the measurement and the constraint. The numbers above
are kept because they are correct for $g_w$, and because mistaking them for a statement
about the gradient is the error that section 15 records.

## 10. Throughput

Measured on the pair distribution of an actual Sysser 01 mesh at $Fn = 0.3$, rather than
on a synthetic worst case: **928 microseconds per panel pair**, giving 594 s of assembly
for $N = 800$ panels and 2090 s for $N = 1500$. Against the plan's budget of 10 minutes of
assembly per speed, $N \approx 800$ fits and $N = 1500$ does not.

Three optimisations got there from 4.1 ms per pair. Grading the wave-angle grid to
equalise phase increment; binning points by oscillation count so that the far-apart
near-surface pairs, which are 2 per cent of pairs but seven times the unit cost, do not
set the grid for everything else; and replacing the termination-tested asymptotic loop with
a banded Horner evaluation, which cut the per-node cost from 2030 ns to 274 ns. For
comparison, `scipy.special.exp1` on complex argument costs 1423 ns per element, which is
why the asymptotic path carrying 76 per cent of nodes matters so much.

## 11. The zeroth iterate is not the thin-ship limit

Dropping the integral term from (3.2) leaves $\tfrac{1}{2}\sigma = u\,n_x$, so the zeroth
iterate of this integral equation is $\sigma = 2u\,n_x$ in this kernel convention. Whether
that coincides with Hogner's distribution depends on his normalisation, which is not
reproduced here, so it is not presented as a named method.

It is also **not** the thin-ship density, which §4 derives as $\sigma = u\,n_x$. The two
differ by a factor of two and the resistance by a factor of four. An earlier version of
this document used the zeroth iterate to pin $C$ and so reported
$C = \rho g^2/(4\pi u^4)$, a fourfold error. The zeroth iterate is legitimate as a probe
of the far-field kernel with the solve switched off, and nothing more; the constant is now
pinned by the derivation of §4, checked twice as described there.

The solved density confirms the derivation directly. On Wigley hulls at $Fn = 0.30$,
$T/L = 0.0625$, the mean of $\sigma/(u\,n_x)$ over panels with $|n_x| > 0.02$ is

| $B/L$ | 0.10 | 0.05 | 0.02 (158 panels) | 0.02 (278 panels) |
|---|---|---|---|---|
| mean $\sigma/(u\,n_x)$ | 1.028 | 1.086 | 1.077 | 1.037 |

which sits near unity, not near two. Piecewise-constant collocation is first order in the
density (§14), so a few per cent of scatter at these panel counts is expected; a factor of
two is not.

## 12. Errors found while verifying, all of which produced plausible wrong answers

Recorded because each was silent, and each is now covered by a test.

1. Truncating the wave-angle grid at $\arctan(t_{\max})$, where $t_{\max}$ is set by the
   decay of the free-wave term, omits the band up to $\pi/2$. The free-wave term is dead
   there but the local part $P(c) \sim -1/c$ decays only algebraically. Cost: 18 per cent
   of $g_w$ at $(X,Y,Z) = (2, 0.8, -0.9)$, and the truncated scheme converged happily
   under refinement to that wrong value.
2. Building the grid in absolute $\tan\theta$ rather than in a normalised variable. A point
   whose own $t_{\max}$ was far below its batch's maximum then had its entire oscillatory
   range covered by a fraction of one panel. Cost: 68 per cent at
   $(X,Y,Z) = (-12.55, 1.80, -0.152)$ in a batch, where the same point evaluated alone was
   right to nine figures.
3. Covering only $\theta > 0$. The integrand carries $Y\sin\theta$ and so is not even in
   $\theta$. Cost: 76 per cent.
4. A gap between the asymptotic magnitude bands, combined with an uninitialised output
   array, left $|c| = 29.999999999999996$ reading uninitialised memory.
5. Returning the term count at which the asymptotic series reaches a target it cannot
   reach, rather than the count at which its terms stop shrinking. The series is
   asymptotic, not convergent, so running past $n \approx |c|$ diverges: at $|c| = 30$ this
   turned a 1e-12 floor into a 100 per cent error.

---

# M2 results: constant-source panels and the Rankine solver

## 13. Influence coefficients

With the kernel $G = -1/(4\pi r)$ the potential of a panel carrying constant density
$\sigma$ is $\phi = -(\sigma/4\pi)I_0$, $I_0$ being the integral of $1/r$ over the panel.
In a frame with the panel in its own plane $\zeta = 0$,

$$\frac{\partial I_0}{\partial\zeta} = -\Omega,\qquad
\frac{\partial I_0}{\partial\xi} = -\sum_{\text{edges}} \frac{\eta_{i+1}-\eta_i}{d_i} L_i,\qquad
\frac{\partial I_0}{\partial\eta} = +\sum_{\text{edges}} \frac{\xi_{i+1}-\xi_i}{d_i} L_i, \tag{13.1}$$

with $L_i = \ln\left[(r_i + r_{i+1} + d_i)/(r_i + r_{i+1} - d_i)\right]$ and $\Omega$ the
signed solid angle. Every influence therefore integrates the kernel over the whole panel;
centroid value times area appears nowhere.

Two properties of $\Omega$ are worth having. It is computed by Van Oosterom and Strackee's
formula, which needs no case analysis and is stable for near-degenerate configurations, and
it gives exactly $4\pi$ at any point enclosed by an outward-oriented closed surface and $0$
outside — verified to 1e-9 on a subdivided icosahedron. And because the normal velocity of
a panel is exactly $\sigma\Omega/4\pi$, the jump term of the integral equation is not bolted
on: a field point in a panel's own plane but outside it gets exactly zero, as symmetry
demands, and one approaching the panel gets $\sigma/2$.

Signs and the branch of $L_i$ were fixed against direct quadrature over the panel, not
asserted; analytic and quadrature velocities agree to 1e-9 relative from four panel radii
down to 0.4.

The diagonal is nevertheless imposed rather than evaluated. At a point in the panel's own
plane the solid-angle formula has a vanishing numerator, so whether $+2\pi$ or $-2\pi$
emerges depends on the sign of a floating-point zero. The field point approaches from the
fluid side, so the required limit is $+1/2$.

## 14. The sphere, and a correction to the plan's acceptance criterion

The exact density is derived, not recalled. A surface density $\sigma_1\cos\theta$ on
$r = a$ produces the exterior potential $-\sigma_1 a^3\cos\theta/(3r^2)$; matching the
sphere-in-a-stream dipole $-u a^3\cos\theta/(2r^2)$ requires $\sigma_1 = 3u/2$, and
$n_x = \cos\theta$, so $\sigma = \tfrac{3}{2}u\,n_x$.

Measured on a subdivided icosahedron, chosen because a UV sphere's polar slivers spoil the
convergence rate for reasons unrelated to the method (its density scatter is more than
three times worse at comparable panel count):

| panels | 80 | 320 | 1280 | 5120 |
|---|---|---|---|---|
| mean $\sigma/(u n_x)$ | 1.6969 | 1.6003 | 1.5504 | 1.5256 |
| error against 3/2 | 0.197 | 0.100 | 0.0504 | 0.0256 |

The error halves with each halving of panel size: **first order**, and Richardson
extrapolation of the last two levels gives 1.5008, i.e. 3/2 to 0.05 per cent. The
condition number of the influence matrix is below 2 at every level, so nothing here is a
conditioning artefact.

**The plan's V3 criterion — 0.5 per cent at 1200 panels with second-order convergence —
was wrong, and is corrected here.** Piecewise-constant collocation on a curved body is
first order in the density, because the best piecewise-constant approximation to a smooth
density is itself only $O(h)$. Second order would need a linear density representation.
The achieved 5 per cent at 1280 panels with a first-order rate and an extrapolant good to
0.05 per cent is the correct expectation, and it confirms the jump term and the panel
integration, which is what V3 exists to do.

Separating the solve from the geometry confirms this reading. Imposing the exact density
$\tfrac{3}{2}u\,n_x$ on the faceted sphere and evaluating the exterior potential gives
errors of 1.27e-1, 3.38e-2 and 8.61e-3 against the analytic dipole — clean second order.
So the geometry and the potential evaluation are second-order accurate, and the first-order
behaviour belongs to the piecewise-constant density alone.

---

# M3 results: the full Neumann–Kelvin solve

## 15. Five errors in the wave-kernel quadrature, all of which the value hid

Section 9 measured the accuracy of $g_w$ and set the mesh constraint from it. That was the
wrong quantity. The influence matrix uses $\nabla g_w$, and the gradient integrand carries
an extra factor of $\sec^2\theta$ — $\mathrm{d}c/\mathrm{d}Z = -\sec^2\theta$ and
$\mathrm{d}c/\mathrm{d}Y = -\mathrm{i}\sec^2\theta\sin\theta$ — so it weights the fast end
of the wave-angle range far more heavily than the value does. Measured against a heavily
refined evaluation, at $(X, Y, |Z|) = (2, 0.8, 0.02)$ the production grid returned $g_w$ to
1.7e-3 and its gradient **185 per cent wrong**.

That error was silent, it converged under `refine`, and it propagated straight into the
resistance: the first Sysser 01 solve returned 18.5 N at $Fn = 0.30$, five per cent of
displacement weight, with the shallowest 3.4 per cent of the wetted area carrying 62 per
cent of the resistance and $\sigma/u$ reaching $-3.2$ against a deep-water maximum of 1.3.
It is tempting to read that as the known waterline difficulty of the Neumann–Kelvin problem
for a surface-piercing body. It was not. It was quadrature.

Four distinct causes, each found by ablation rather than by inspection:

1. **A narrow peak with nothing to resolve it.** $\operatorname{Im} c$ vanishes where
   $\tan\theta = -X/Y$, and there $|c| = \sec^2\theta\,|Z|$ takes its minimum over the
   range. The value integrand has only a logarithm there; the gradient carries $-1/c$, so
   it has a peak of height $O(1/|Z|)$ and width $O(|Z|)$. At $(2, 0.8, 0.02)$ that width is
   1.6e-3 in the normalised variable against a panel width of 0.025. The peak narrows in
   proportion to $|Z|$, so **uniform refinement cannot fix it** — reaching 1e-8 took about
   eight times the production node count. The remedy is local: the panel containing the
   peak is replaced, per point, by a partition graded towards it by successive halving,
   which is parameter-free and needs no estimate of the width. Twelve halvings resolve any
   width down to the panel over $2^{12}$. Because the replacement adds a *fixed* number of
   nodes per point, the calculation stays rectangular and vectorised.
   $Y = 0$ puts the peak at $\theta = \pm\pi/2$, outside the range, which is why the
   production grid was exact to machine precision along $Y = 0$ at every $|Z|$ and wrong
   off it. That asymmetry is what identified the mechanism.
2. **A grading that equalised one term of the phase.** The phase over the inner range is
   $c_q\xi^2 + c_l\xi$ cycles, with $c_q = \text{reach}\,|Y|/(2\pi|Z|)$ and
   $c_l = |X|t_{\max}/(2\pi)$. The grid was uniform in $\xi^2$, which equalises the
   quadratic term exactly, plus a short stretch uniform in $\xi$ — a "core" — for the
   linear term. The split between them was the fixed constant 0.15. Whenever the two terms
   were comparable the core covered a stretch that was overwhelmingly quadratic while
   grading it uniformly, and its outer panels carried several times the per-panel budget.
   At $(0.5, 2, 0.01)$ the gradient was 8.5e-4 wrong and needed four times the core
   panels; at $(0.5, 6, 0.02)$ the $\partial/\partial Z$ component was 16 per cent wrong.
   Both are now exact by construction: the whole phase is inverted in one expression,
   $$\xi_j = \frac{-c_l + \sqrt{c_l^2 + 4c_q\,j\,(c_q+c_l)/n}}{2c_q}, \tag{15.1}$$
   so every panel carries exactly $(c_q+c_l)/n$ cycles and there is no core and no split.
3. **Too few nodes per panel for the gradient.** With the grading fixed, raising the
   Gauss–Legendre order from 12 to 16 cut the worst gradient error inside the Sysser
   envelope from 7.5e-3 to 7.0e-4 for 23 per cent more nodes, while raising the *panel*
   count eightfold reached only 7e-3. Order, not panel count, was the binding knob — the
   signature of an integrand whose amplitude varies strongly within a panel.
4. **A panel cap reached without saying so.** The grid is capped at 16 000 panels, so
   beyond $16\,000 \times 1.5 = 24\,000$ cycles the per-panel budget is silently exceeded.
   At $(8, 6, 0.005)$, 7750 cycles, `refine` above 4 changed nothing because every setting
   was clamped, and the "reference" was itself unconverged. This is now an explicit
   envelope: `greens.oscillation_count` returns the count, `greens.envelope_ok` tests it,
   and `nk.influence_matrix` **raises** rather than returning a plausible number.

The cheapest lever turned out to be the one not previously questioned. `_DAMPING_REACH`
sets $t_{\max} = \sqrt{\text{reach}/|Z| - 1}$ and so the cycle count, and hence the cost,
linearly. It was 40, putting the free-wave truncation at $\mathrm{e}^{-40}$, below double
precision. Cutting it to 25 puts the floor at $\mathrm{e}^{-25} = 1.4$e-11 — which is
where the measured median error now sits, so the floor is a stated truncation rather than
an unnoticed deficiency — and pays for the higher order.

**Accuracy now.** Inside the envelope a Sysser 01 mesh at $Fn = 0.30$ actually spans
($|X| \le 11.1$, $|Y| \le 3.5$, topmost centroid 1 mm below the surface so
$|Z| \ge 0.0139$), over 400 randomly drawn panel pairs:

| | worst | median |
|---|---|---|
| $\nabla g_w$ | 7.0e-4 | 6e-11 |
| $g_w$ | 1.1e-5 | — |

against 1.85 for the worst gradient before. Throughput on an actual 280-panel Sysser mesh
is **1164 microseconds per panel pair**, against 928 before, so the whole correction cost
25 per cent. That gives 745 s of assembly at $N = 800$ and 570 s at $N = 700$, so the
plan's 10-minute budget per speed caps the mesh at about 700 panels.

**Section 9's mesh constraint is withdrawn.** It required
$k_0|z_i + z_j| \gtrsim 0.1$, from the accuracy of $g_w$. The binding constraint is the
oscillation count of the *pair*,
$$\text{cycles} = \frac{1}{2\pi}\left[\frac{\text{reach}\,|Y|}{|Z|}
 + |X|\,t_{\max}\right] \le 24\,000 , \tag{15.2}$$
which depends on $|Y|/|Z|$ and not on $|Z|$ alone. A Sysser 01 mesh at $Fn = 0.30$ with its
topmost centroid 1 mm down reaches 1699 cycles — comfortably inside — where the old
constraint would have demanded 7 mm and a mesh that cannot be built near the bow and stern.

## 16. Two routes to the resistance, and what each is good for

The plan gives the reason for taking the resistance from the far-field amplitude rather than
by integrating hull pressure: it is "far less sensitive to panel-level pressure error". The
measurements below say the opposite, and it is worth being precise about why, because the
two estimators fail in completely different ways and neither failure is obvious.

### 16.1 The far-field integral is positive-definite in the source density

$R_W = C\int \tfrac12(|A_+|^2+|A_-|^2)\lambda^2(\lambda^2-1)^{-1/2}\mathrm{d}\lambda$ is a
positive-definite quadratic form in $\sigma$. Writing $\sigma = \sigma_\text{true} + \delta$,

$$R_W = R_\text{true}
 + 2C\!\int\!\operatorname{Re}\!\left(A_\text{true}^*\,\delta A\right)(\cdots)
 + C\!\int\!|\delta A|^2(\cdots) , \tag{16.1}$$

and the last term is strictly positive. Discretisation error can therefore only *raise* the
far-field resistance; it can never cancel. Worse, $A(\lambda)$ is a strongly oscillatory
integral, so the signal carries heavy phase cancellation while error adds incoherently. The
amplification is severe.

Measured by injecting known Gaussian noise into a converged density on a thin Wigley hull,
$B/L = 0.02$, 278 panels, $Fn = 0.30$, against the Michell oracle:

| noise on $\sigma$ | 0 | 5 % of $u$ | 10 % | 20 % | 40 % |
|---|---|---|---|---|---|
| far-field / oracle | 0.936 | 1.997 | 4.165 | 26.08 | 87.59 |
| pressure / oracle | 0.958 | 1.018 | 0.333 | 1.759 | 0.515 |

**Five per cent of noise on the density doubles the far-field resistance.** The pressure
route, being a bilinear form with no definite sign, stays near unity on average and scatters
either side of it — which is its own hazard, but a different one.

This is the mechanism behind the net-source-flux diagnostic. For a closed body in a stream
$\int\sigma\,\mathrm{d}S$ vanishes, and the free surface can carry only a little, so its
size is a direct measure of the incoherent part of the error — and it feeds
$A(\lambda)$ hardest near $\lambda = 1$, which is where the resistance integrand is largest.
It is reported by every solve.

### 16.2 The quadratic Bernoulli term is not optional

Steady Bernoulli gives
$p = \rho u\varphi_x - \rho g z - \tfrac12\rho|\nabla\varphi|^2$. The hydrostatic term
carries no $x$-force — over the wetted surface closed by the $z=0$ lid the divergence
theorem gives $\int z\,n_x\,\mathrm{d}S = 0$, and on a Sysser 01 mesh it comes to 6e-5 of
its own scale, so the mesh honours the identity. The quadratic term is smaller than
$u\varphi_x$ by one order in slenderness, and the first implementation dropped it on that
basis.

That is sound for a thin hull and wrong for anything else, and the thin-hull oracle cannot
see the difference. The submerged sphere is what identified it, because it removes the
waterline entirely and the discrepancy survived. At radius 0.1 m and $k_0 d = 2$, 320
panels, with a net source flux of only 1.3e-4 — so neither the waterline nor incoherent
error was responsible:

| | far-field | pressure, linear only | pressure, full |
|---|---|---|---|
| $R_W$, N | 1.2663 | 0.7546 | 1.1459 |
| far-field / pressure | — | 1.678 | **1.105** |

The quadratic term is $+51.9$ per cent of the linear one here, against $-1.9$ per cent on the
thin Wigley hull, which is the slenderness scaling made visible. With it the two routes
agree to 10.5 per cent at 320 panels and 27 per cent at 80, so they converge towards each
other; without it they sit 68 per cent apart at 320 panels and would not converge at all,
because the missing term does not vanish under refinement.

### 16.3 What each route is for

- **Thin or slender bodies, converged density.** Either route; they agree. On the Wigley
  hull at $B/L = 0.02$, $T/L = 0.0625$, $Fn = 0.30$, 278 panels, with the resolution cap
  applied and the full Bernoulli pressure: far-field 0.936 of Michell, pressure 0.940, the
  two within **0.4 per cent** of each other. The quadratic term is only $-1.9$ per cent
  here, as slenderness predicts, and without it the two routes sit 2.3 per cent apart
  instead.

  Uncapped and with the linear pressure only — the variant the wider sweep used — the same
  hull gives far-field / Michell of 0.942 at $Fn = 0.30$ and 0.940 at $0.40$ against
  pressure 0.958 and 0.993 at 278 panels, and 0.928 against 0.979 at 558. Every one of
  these is within 7 per cent of the oracle, which is the point; the spread between the
  variants, 2 per cent, is the honest uncertainty of a 278-panel mesh. The solved density
  averages $1.04\,u\,n_x$, confirming §4's derivation.
- **Anything with a coarse density.** The pressure route, with the far-field value reported
  alongside as an upper bound and the net source flux as the reason. The far-field estimate
  is not merely noisier — it is biased one way.

Both are computed and both are reported. `nk.solve_nk` returns the far-field value with the
flux and residual diagnostics beside it, and warns when the flux exceeds one per cent.

## 17. Corrections to the plan's verification criteria

Three of the plan's acceptance criteria did not survive measurement. They are corrected here
rather than quietly reinterpreted.

**V3, "0.5 per cent at 1200 panels, second-order convergence".** Wrong rate. Piecewise-
constant collocation is first order in the density; section 14 records the measurement and
the corrected expectation.

**V4, "large separation → Rankine + image".** Wrong limit. The Kelvin waves decay as
$x^{-1/2}$ along the track while the Rankine pair decays as $x^{-2}$, so the wave part
*dominates* at large separation — measured at 6 times the Rankine part at $X = 2$ and
6.2e4 at $X = 200$. The correct limit is large *depth*, where the free surface becomes a
rigid wall; section 8 gives it.

**V11, "far-field $R_w$ vs independent pressure/energy evaluation, 5 per cent".** The
threshold is met on an oracle case — 0.4 per cent between the two routes on a thin Wigley
hull at 278 panels — but the plan's stated reason for making the far-field route primary,
that it is "far less sensitive to panel-level pressure error", is the wrong way round.
Section 16 gives the measurements: the far-field form is positive-definite in $\sigma$ and
five per cent of noise on the density doubles it, while the pressure route moves by two per
cent. V11 is therefore not only a consistency check between two roughly equal estimators.
It is the check that tells you whether the far-field number can be used at all, and the net
source flux is its early warning.

**V12's thresholds are not reachable on Sysser 01 within the compute budget.** They ask for
a body-condition residual below 1 per cent of $u$ and a net flux below
$10^{-6}\,u\,S$. Measured on the Wigley oracle the method is sound; measured on Sysser 01 at
160 to 280 panels the residual is 9 to 13 per cent of $u$ and the flux 8 per cent of
$u\,S$, and the far-field resistance falls from 45.5 N to 12.8 N over that refinement. The
resistance is therefore **not converged on Sysser 01 at any panel count the 10-minute
assembly budget allows**, and no amount of presentation changes that. Section 18 records how
far the trend goes when the budget is deliberately exceeded.

## 18. Sysser 01: what the method actually delivers, and what it does not

Everything above is either a derivation or a measurement on a case with an analytic answer.
This section is the worked hull, and the result is negative in a specific and useful way.

### 18.1 The refinement trend

Sysser 01 at $Fn = 0.30$, waterline-fitted mesh, topmost panel row's lower edge 10 mm below
the surface, displacement weight 368.86 N:

| panels | 160 | 280 | 480 |
|---|---|---|---|
| $R_W$ far-field, N | 45.52 | 12.83 | 29.11 |
| $R_W$ pressure (linear term only), N | 8.59 | 4.21 | 7.55 |
| far-field / pressure | 5.30 | 3.05 | 3.85 |
| body residual, rms as a fraction of $u$ | 0.093 | 0.098 | 0.042 |
| net source flux, fraction of $u\,S$ | 0.081 | — | — |
| max $|\sigma|/u$ | 2.39 | 1.31 | 2.26 |

The body-condition residual behaves: it more than halves from 280 to 480 panels, so the
*solution* is converging. The resistance does not. The far-field value swings by a factor of
3.5 with no sign of a limit, and it is not even monotone.

That combination is exactly what §16.1 predicts. The far-field integral is positive-definite
in $\sigma$, so it carries a $+C\int|\delta A|^2$ term; $\delta A$ falls only as fast as the
density error, while $A$ itself is a strongly cancelling oscillatory integral. A residual
falling from 10 to 4 per cent is therefore entirely compatible with a resistance that is
still dominated by its own error term.

> **Superseded by §22.1.** The reading above is half right. The one-sided bias is real, but
> the *swing* is not intrinsic to it: almost all of it was a single spurious mode, a net source
> flux the solve had no reason to suppress. With that mode constrained away both routes
> converge on this same mesh sequence, at $+5.8$ and $+2.9$ per cent on the last refinement.
> The table and conclusions in this section are the unconstrained solve, kept because they are
> what the diagnosis was built from, and because a resistance that swings while its residual
> falls is the symptom that led to the mode. Read §22 for what the method now delivers.

### 18.2 What is and is not established

Established, on cases with analytic answers:

- the formulation, the Green function and its gradient, the influence matrix, the solve, and
  both resistance routes are correct and mutually consistent, to 0.4 per cent between the
  two routes on a thin Wigley hull at 278 panels, both at 0.94 of the Michell value;
- the thin-ship density is $u\,n_x$ and the resistance constant $\rho g^2/(\pi u^4)$, both
  derived and both confirmed numerically;
- the wave-kernel quadrature is accurate to 7.0e-4 in the worst case and 6e-11 in the median
  over the region a Sysser mesh spans.

Not established *by the unconstrained solve*:

- **a converged wave resistance for Sysser 01.** At every panel count the ten-minute
  assembly budget allows, the far-field resistance is dominated by discretisation error.
  This is not a presentational matter and it is not resolved by choosing a mesh whose answer
  looks plausible. §22.1 revisits this: with the net source flux constrained, both routes do
  converge on this mesh sequence, and the pressure route lands at 0.81 of the measured
  residuary resistance.

### 18.3 The source-panel quadrature is a second, separate bias

`order` sets the quadrature over the source panel for the wave influence. Order 1, the
centroid rule, is legitimate — the wave kernel is bounded for $z < 0$ — but coarse. Measured
on the thin Wigley hull at 158 panels, $Fn = 0.30$, against the Michell oracle:

| order | far-field / oracle | change in far-field | pressure / oracle |
|---|---|---|---|
| 1 | 1.013 | — | 0.953 |
| 2 | 0.891 | −12.1 % | 1.010 |
| 3 | 0.864 | −3.0 % | 1.174 |

The far-field value converges, at $-12$ then $-3$ per cent, towards about 0.86; the pressure
value moves the other way, to 1.17. Since $B/L = 0.02$ means the true Neumann–Kelvin answer
is within $O(B/L)$, i.e. a couple of per cent, of Michell, **both** are in error at 158
panels and they bracket the oracle from either side. At 278 panels they close to 0.936 and
0.940 (§16.3), so what this table measures is not the quadrature order alone but the
combined uncertainty of a 158-panel mesh, about 15 per cent.

The honest reading is therefore narrower than "order 1 is 12 per cent low": order 1 carries a
bias of order 10 per cent on a mesh this coarse, the sign of which depends on the route, and
$(k_0h)^2/24$ would predict 1.6 per cent — the excess coming from the near-surface pairs,
where the kernel varies on the scale of $|z_i + z_j|$ rather than of the wavelength. Order 2
costs four times the assembly, which at 1164 microseconds per pair is not affordable as a
default, so order 1 stays and `solve_nk` reports the bias in its notes.

### 18.4 What would actually fix this

The obstruction is arithmetic, not conceptual. Assembly costs 1164 microseconds per panel
pair, so the influence matrix alone is 28 minutes at 1200 panels and 48 at 1500, with the
pressure route's gradient matrix costing the same again, and §18.1 gives no reason to believe
1500 panels would be enough. Three things would change the picture, in ascending order of
effort:

1. **Raise the source-panel order only for the pairs that need it.** The 12 per cent of
   §18.3 comes from near-surface pairs, a small fraction of the total, so a per-pair
   criterion would buy most of order 2 at close to the cost of order 1.
2. **Tabulate and interpolate the Kelvin kernel.** The plan defers this until pointwise
   accuracy is established. It now is, to 7e-4 worst and 6e-11 median (§15), so this is
   unblocked, and it is where the factor of ten lives.
3. **Represent the density better than piecewise constant.** This would cure the
   first-order convergence of §14 as well, and it is the only one of the three that
   addresses the actual cause rather than the cost.

All three are outside M3.

---

# M4 results: Sysser 01 against the official Delft data

## 19. The hydrostatics reconcile, except in three quantities

The primary releases were downloaded rather than inferred:

- hydrostatics — 4TU.ResearchData [10.4121/21501375](https://doi.org/10.4121/21501375), file
  `DSYHS_hydrostatics_modelscale.xlsx`, verified against the release's published MD5
  `8d041601483f5d165d2198446eb7f4de`;
- measurements — 4TU.ResearchData [10.4121/21501402](https://doi.org/10.4121/21501402), file
  `DSYHS_all_measurements_modelscale.xlsx`.

Both are CC0. Every value in `data/sysser01_reference.toml` now comes from row `Sysser = 1`
of the sheet *Canoe body hydrostatics*, and the figures supplied during review — which had no
source attached — turn out to agree with the release exactly. Three quantities are new from
the release: LCF, the midship area, and $C_B$.

**The LCB datum is resolved, and it was worth checking.** The release's own *Info* sheet
states that LCB and LCF are measured "with respect to 1/2 waterline length *in upright
condition!*". The exclamation mark is theirs, and it matters: for the heeled columns the
datum stays the upright half-waterline point rather than moving with the heeled waterline.

**The heeled columns hold displacement constant.** $\nabla_{10} = \nabla_{20} = \nabla_{30}
= \nabla_0 = 0.0376136$ m³ exactly, so the heeled hydrostatics were computed at fixed
displacement with sinkage and trim solved for. A test that rotates a hull at fixed sinkage and
expects the volume to be conserved is therefore wrong; the plan contained one and it is
removed.

Computed against published, at the reference condition set by matching the official
displacement at zero trim, on a 32 × 160 mesh:

| | computed | published | diff |
|---|---|---|---|
| $\nabla_c$ | 0.037614 m³ | 0.037614 | **−0.000 %** |
| $A_w$ | 0.558230 m² | 0.558566 | **−0.060 %** |
| $T_c$ | 0.127143 m | 0.127040 | +0.081 % |
| $A_m$ | 0.041597 m² | 0.041651 | −0.130 % |
| $C_p$ | 0.562243 | 0.564414 | −0.385 % |
| $L_{wl}$ | 1.608268 m | 1.600000 | **+0.517 %** |
| $B_{wl}$ | 0.513026 m | 0.507200 | **+1.149 %** |
| $C_m$ | 0.637722 | 0.646410 | −1.344 % |
| $C_{wp}$ | 0.676575 | 0.688296 | −1.703 % |
| $C_b$ | 0.358555 | 0.364842 | −1.723 % |
| $S_c$ | 0.656026 m² | 0.642534 | **+2.100 %** |
| LCB from midship | −0.034366 m | −0.036640 | +2.27 mm |
| LCF from midship | −0.050752 m | −0.053280 | +2.53 mm |

Displacement and waterplane area agree to better than 0.06 per cent, which is the mesh's own
uncertainty. Waterline length, waterline beam and wetted area do not, and now that the
primary release has been read, **the published table is not the explanation**. The remaining
candidate is the geometry's provenance: the IGES header path reads *"Rhino modellen na
inmeten 2012"* (Rhino models after measuring, 2012) and the single patch is named
*"Gerebuild oppervlak"* (reconstructed surface). The file is a 2012 re-measurement of the
physical model; the table describes the hull as the series was built and towed. A 1 to 2 per
cent difference in $B_{wl}$ and $S_c$ between an original lines plan and a re-measured hull
is unremarkable, and $S_c$ — the quantity most sensitive to local surface fairness — shows the
largest gap.

This is recorded as an **unresolved geometry-provenance discrepancy of stated magnitude**,
carried as a bias in every downstream comparison, and not assumed away. Earlier work checked
that attitude does not explain it: solving heave *and* trim to force $\nabla_c$ and LCB to
match converges to 0.57° bow-up and makes $L_{wl}$, $B_{wl}$ and $S_c$ all worse.

## 20. The measurements, and why Fn = 0.30 is the wrong place to look

The release gives, per speed, total resistance and the dynamic sinkage and trim. It does not
give residuary resistance: that follows from an ITTC-57 friction line at form factor zero,
which is what the series' own reduction does, so **residuary resistance is a proxy for wave
resistance and not a measurement of it** — it also contains nonlinear and viscous-form
contributions. The workbook records the tank temperature, 17.3 °C, but leaves its density and
viscosity cells at zero, so both are computed: $\rho = 998.72$ kg/m³ and
$\nu = 1.0749\times10^{-6}$ m²/s.

Sysser 01, bare hull upright:

| $Fn$ | 0.10 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|
| $R_t$, N | 0.258 | 1.108 | 1.885 | 3.019 | 4.753 | 9.793 | 19.41 | 32.70 | 51.15 |
| $R_f$, N | 0.265 | 0.911 | 1.357 | 1.885 | 2.487 | 3.162 | 3.909 | 4.727 | 6.571 |
| $R_r$, N | −0.008 | 0.198 | 0.527 | 1.134 | 2.266 | 6.632 | 15.50 | 27.98 | 44.58 |
| $R_r/\Delta$ | −0.00002 | 0.0005 | 0.0014 | **0.0031** | 0.0062 | 0.0180 | **0.0421** | 0.0759 | 0.1210 |
| sinkage, mm | −0.38 | 2.70 | 5.05 | 6.57 | 10.42 | 14.93 | 24.96 | 30.77 | 21.22 |
| trim, ° | −0.015 | 0.010 | 0.022 | −0.032 | −0.101 | −0.564 | −1.622 | −3.185 | −5.269 |

Two things follow, and both change how M3's numbers should be read.

**At $Fn = 0.30$ the quantity being predicted is 0.31 per cent of displacement weight.** Every
comparison at that speed divides by 1.13 N. M3's Sysser figures — a far-field resistance
swinging between 12.8 and 45.5 N and a pressure value between 4.2 and 8.4 N — are 4 to 40
times that, but the speed was chosen for being in the middle of the plan's range, not for
being informative. At $Fn = 0.45$ the target is 15.5 N and 4.2 per cent of weight, thirteen
times larger in absolute terms.

**The subtraction gives a negative residuary at $Fn = 0.10$**, −0.008 N. That is kept as it
comes out. It says the ITTC-57 line slightly over-predicts the friction of this model at
$Re = 5.9\times10^5$, which is information about the reduction; clamping it to zero would hide
the one speed at which the reduction visibly fails.

**The dynamic attitude is not negligible above $Fn = 0.35$.** The model sinks 25 mm and trims
1.6° bow-up by $Fn = 0.45$, against a canoe-body draught of 127 mm. Comparisons must be run
at the measured attitude, which is what `measurements.sysser01_runs` supplies, in the units
`Attitude` takes — metres positive downward and **degrees** bow-down positive. Storing the
trim in radians and letting the caller convert produced a 57-fold error in the first version
of this comparison, so the units are now asserted by a test.

The sign of the trim is corroborated rather than assumed: the workbook's $\theta$ reaches
$-5.27°$ at $Fn = 0.60$ with the model sunk 21 mm, and a yacht at that speed squats at the
stern and lifts the bow, so a negative $\theta$ must mean bow-up — which is what bow-down
positive gives. What remains genuinely unconfirmed is the reference point for $z$: the
workbook places the towing point 255 mm above the waterline and the transducers 0.53664 m
forward and 0.46336 m aft of the centre of gravity, but does not say where the sinkage was
measured. Below $Fn = 0.35$ the trim is under 0.11° so the choice hardly matters; by
$Fn = 0.50$ it is 3.2° and it matters a great deal.

## 21. The measured attitude cannot be applied above Fn ≈ 0.35

The plan requires the resistance comparison to be made at the recorded dynamic sinkage and
trim. That turns out not to be possible over most of the speed range, for a reason that is
in the data rather than in the code.

**First, an error of mine.** `Attitude` rotates about `pivot`, which defaults to the origin.
In this IGES file the origin sits at the **aft end of the waterline** — the wetted extent
runs from $x = -0.0064$ m to $x = 1.6017$ m — so the default pivot is half a waterline length
from the midpoint. Applying the measured trim about it raises the bow by
$1.6\tan(5.27°) = 148$ mm at $Fn = 0.60$, against a canoe-body draught of 127 mm, and
removes over half the immersed volume. The first sweep did exactly that and its numbers are
discarded, not reported.

**Second, and not fixable.** Rotating about pivot $P$ with trim $\theta$ and heave $s$ is the
same rigid motion as rotating about $Q$ with the same $\theta$ and heave
$s + (x_Q - x_P)\tan\theta$. The pivot is therefore not an independent choice: it is exactly
equivalent to an ambiguity in the heave, of size (pivot uncertainty) × $\tan\theta$. The
release states the sinkage and the trim but **not where the sinkage was measured**, and the
sensible candidates — the aft end of the waterline, its midpoint, the centre of buoyancy, the
centre of flotation — span 0.80 m. Hence:

| $Fn$ | 0.20 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|
| measured sinkage, mm | 2.70 | 6.57 | 10.42 | 14.93 | 24.96 | 30.77 | 21.22 |
| heave ambiguity, mm | 0.14 | 0.44 | 1.40 | 7.86 | 22.58 | 44.39 | 73.57 |
| ambiguity / sinkage | 0.05 | 0.07 | 0.13 | 0.53 | 0.90 | 1.44 | 3.47 |

Up to $Fn = 0.35$ the ambiguity is at most 1.4 mm, 13 per cent of the sinkage, and the
attitude is effectively determined. At $Fn = 0.45$ it equals the sinkage. Above $Fn = 0.50$
it exceeds it, and the attitude is simply not determined by the published data.

**The two windows do not overlap.** The speeds at which the attitude is known are the speeds
at which the measured residuary resistance is smallest — 2.27 N and 0.6 per cent of
displacement weight at $Fn = 0.35$, against 27.98 N and 7.6 per cent at $Fn = 0.50$. So the
range where the target is worth predicting is the range where the condition to predict it at
is unknown. Resolving this needs the reference point for $z$, which is a question for the
dataset's authors, not for more computation.

One caution about a test that looks like a discriminator and is not. It is tempting to pick
the pivot that best conserves the immersed volume, on the grounds that the model's weight does
not change. That reasoning is wrong. In linearised theory the hull is clipped at the
*undisturbed* plane $z = 0$ by construction, while the real hull at speed sits in a trough of
its own making, so the volume below $z = 0$ genuinely exceeds the static value and by a large
margin — 150 per cent of it at $Fn = 0.50$ with a mid-hull pivot. That is the linearisation's
own bookkeeping and carries no information about the pivot.

## 22. Constraining the net source flux, which is worth a factor of ten

Section 16.1 showed that the far-field integral is positive-definite in $\sigma$ and that
incoherent error can only inflate it, and §18.1 showed the Sysser 01 resistance swinging by
3.5 while the body residual fell steadily. This section identifies the mode responsible and
removes it.

**The mode.** A closed body in a stream emits no net source strength, and the free surface can
carry only a little, so $\int_S \sigma\,\mathrm{d}S$ should very nearly vanish. On the two
cases that verify well it does: $1.0\times10^{-4}$ of $u S$ on a thin Wigley hull at 278
panels and $1.3\times10^{-4}$ on a submerged sphere at 320. On Sysser 01 it is
$8.1\times10^{-2}$ at 160 panels and $3.4\times10^{-2}$ at 280 — two to three orders larger.

That matters far more than its size suggests. A spurious net source is a **monopole**, and its
far-field amplitude does not fall off with $\lambda$ the way a closed body's does, while the
resistance integrand carries $\lambda^2(\lambda^2-1)^{-1/2}$ and is largest exactly where the
monopole lives, at $\lambda \to 1$.

**The remedy.** The square system cannot satisfy an extra constraint exactly, so solve the
constrained least-squares problem — minimise $\|A\sigma - b\|$ subject to
$\mathbf{a}^\mathsf{T}\sigma = 0$ with $\mathbf{a}$ the panel areas — by the null-space
method. The trailing columns of the QR factorisation of $\mathbf{a}$ span the feasible
subspace; there is nothing to tune.

**What it does.** Sysser 01 at $Fn = 0.30$, static attitude, measured residuary resistance
1.134 N:

| panels | | net flux | $R_W$ far-field | $R_W$ pressure | far-field / pressure | residual |
|---|---|---|---|---|---|---|
| 160 | unconstrained | 8.1e-2 | 45.46 N | 10.15 N | 4.48 | 0.1354 |
| 160 | flux = 0 | 1.5e-17 | **4.16 N** | **0.504 N** | 8.25 | **0.0888** |
| 280 | unconstrained | 3.4e-2 | 12.82 N | 4.69 N | 2.73 | 0.0962 |
| 280 | flux = 0 | 4.5e-18 | **2.76 N** | **0.895 N** | 3.09 | **0.0910** |

**Why this is a repair and not a fudge.** Three independent reasons.

First, the off-collocation body residual **improves** — from 0.135 to 0.089 at 160 panels. A
constraint that deleted real physics would degrade the body condition, since that residual is
measured at points the solve never sees. It improves because the flux mode was violating the
body condition too.

Second, it is a **null operation where the flux is already small**. The submerged sphere's
resistance changes in the sixth significant figure, the thin Wigley hull's by 0.1 per cent,
and the residual of neither moves. A constraint that improved coarse answers by distorting
them would distort the fine ones too.

Third, the two constrained routes now **bracket the measurement** — 2.76 N and 0.895 N against
1.134 N at 280 panels — where unconstrained they sat 4 to 40 times above it. And they close on
each other under refinement, far-field over pressure going from 8.25 to 3.09.

It is therefore the default in `solve_nk`, with `zero_net_flux=False` to recover the plain
square solve. One consequence to note: removing the monopole removes low-$\lambda$ content, so
the *share* of the spectrum beyond the mesh's resolved $\lambda$ rises — from 4 to 14 per cent
on a 160-panel Sysser mesh. The absolute tail is unchanged; it is the denominator that shrank.

### 22.1 It also makes the sequence converge, which §18 said it did not

The 480-panel level, which was still running when the above was written, changes the
conclusion of §18 rather than merely improving its numbers. Sysser 01 at $Fn = 0.30$, static
attitude, measured residuary 1.134 N:

| panels | 160 | 280 | 480 | last step |
|---|---|---|---|---|
| $R_W$ far-field, unconstrained | 45.46 | 12.82 | 29.07 N | ×2.3, not monotone |
| $R_W$ far-field, flux = 0 | 4.163 | 2.761 | **2.922 N** | **+5.8 %** |
| $R_W$ pressure, unconstrained | 10.15 | 4.69 | 8.40 N | ×1.8, not monotone |
| $R_W$ pressure, flux = 0 | 0.504 | 0.895 | **0.921 N** | **+2.9 %** |
| body residual, flux = 0 | 0.0888 | 0.0910 | **0.0234** | |

**§18.1's finding is superseded.** It reported that the solution converged while the
resistance did not, and read that as the positive bias of §16.1 dominating. Half of that was
right: the bias is real and the far-field route still sits 2.58 times the measured residuary.
But the swing itself was not intrinsic — it was the flux mode, and with the mode removed both
routes converge, at $+5.8$ and $+2.9$ per cent on the last refinement. That is close to the
plan's V10 threshold of 1 per cent between successive levels rather than hopelessly far from
it, and the body residual at 480 panels is 0.0234 of $u$ against V12's 0.01.

Against the measurement, the constrained pressure route gives **0.921 N against 1.134 N
measured, a ratio of 0.81**. Two things make that better than it looks rather than worse.
Residuary resistance is not wave resistance: it carries nonlinear and viscous-form
contributions as well, so the wave resistance it brackets is *below* 1.134 N and 0.921 N is
nearer than the ratio suggests. And $Fn = 0.30$ is the speed at which the target is smallest,
0.31 per cent of displacement weight (§20), so it is where any prediction is hardest.

The far-field route remains 2.58 times the measurement, consistent with the one-sided bias of
§16.1 and with the source-panel quadrature bias of §18.3 — neither of which the flux
constraint addresses. **The pressure route is the one to quote**, with the far-field value
beside it as an upper bound.

## 23. Across speed the agreement does not hold, and the Fn = 0.30 point was luck

§22.1 reported the constrained pressure route at 0.921 N against a measured residuary
resistance of 1.134 N, a ratio of 0.81. Run at every speed on the same 480-panel mesh, static
attitude, that ratio is not representative:

| $Fn$ | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|
| $R_r$ measured, N | 0.527 | 1.134 | 2.266 | 6.632 | 15.503 | 27.976 |
| $R_W$ pressure, flux = 0, N | −0.388 | 0.904 | 7.147 | 12.756 | 7.741 | 7.859 |
| pressure / $R_r$ | −0.74 | **0.80** | 3.15 | 1.92 | 0.50 | 0.28 |
| $R_W$ far-field, flux = 0, N | 1.449 | 2.946 | 16.269 | 15.207 | 11.077 | 9.834 |
| far-field / $R_r$ | 2.75 | 2.60 | 7.18 | 2.29 | 0.71 | 0.35 |

The ratio spans 0.28 to 3.15, and the predicted curve **peaks near $Fn = 0.35$–$0.40$ and then
falls** while the measurement rises monotonically through $Fn = 0.50$. **The agreement at
$Fn = 0.30$ is therefore one point in a scatter of a factor of four, and quoting it as
validation would be wrong.** It is recorded here because the temptation to quote it is
exactly what this section exists to prevent.

Three things are wrong, and they are separable.

**The attitude is wrong at the top of the range, and known to be.** At $Fn = 0.50$ the model
sinks 30.8 mm — a quarter of the canoe-body draught — and trims 3.2° bow-up. The sweep runs at
the static attitude because §21 shows the measured one cannot be applied there: the heave
ambiguity from the unstated sinkage datum is 44 mm, larger than the sinkage itself. A hull run
27 mm too shallow makes far less wave, so the prediction *must* undershoot at high speed, and
it does, monotonically: 1.92, 0.50, 0.28 at $Fn = 0.40$, 0.45, 0.50. This is not a defect of
the method; it is the consequence of a condition the published data does not determine.

**The $Fn = 0.35$ point is an outlier and probably an artefact.** It breaks the trend from both
sides — 2.60 at $Fn = 0.30$, 7.18 at 0.35, 2.29 at 0.40 by the far-field route — where the
measured curve is smooth. A real interference hump would appear in the measurement too. The
body residual there, 0.0306, is no worse than its neighbours, so the diagnostics do not flag
it, which is itself worth knowing: **the residual and the flux do not catch everything.**

**The low-speed end is dominated by its own error.** At $Fn = 0.25$ the target is 0.527 N,
0.14 per cent of displacement weight, and the pressure route returns $-0.388$ N. A negative
wave resistance is unphysical and says plainly that the discretisation error exceeds the
signal there. It is reported rather than clipped, for the same reason §20 keeps the negative
measured residuary at $Fn = 0.10$.

### 23.1 What M4 concludes

- **The hydrostatics reconcile** against the primary release, to 0.06 per cent in displacement
  and waterplane area, with a stated and attributed geometry-provenance discrepancy of 0.5 to
  2.1 per cent in $L_{wl}$, $B_{wl}$ and $S_c$ (§19).
- **The measurements are extracted and reduced** with their conventions and assumptions
  recorded, including which are corroborated and which are not (§20).
- **The resistance comparison is not achieved.** Not because it was not attempted, but because
  two independent obstructions were found: the mesh resolution the compute budget allows leaves
  errors comparable to the signal at low $Fn$, and the published data does not determine the
  running attitude at high $Fn$. The two obstructions have disjoint ranges of validity and
  together they cover the whole useful speed range.

The single most useful next step is not a bigger mesh. It is to obtain the reference point for
the measured sinkage, which would open the $Fn \ge 0.40$ range where the target is large and
where an error in the prediction would actually be visible against it.
