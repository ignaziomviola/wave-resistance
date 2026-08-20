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
