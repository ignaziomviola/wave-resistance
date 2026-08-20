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

**The constant $C$ is fixed by verification, not asserted here.** It is determined by
requiring that the closure $\sigma = u\,n_x$, which is (3.2) with the integral term
dropped and hence the Hogner or zeroth-order slender-ship approximation, reproduce on a
thin symmetric hull the Michell value already verified independently in this repository:
$10^3 C_W = 2.1413$ at $Fn = 0.300$ and $1.2362$ at $Fn = 0.345$ for the Wigley hull with
$B/L = 0.1$ and $T/L = 0.0625$. That is test V2.

Two cautions on that closure. First, $\sigma = u\,n_x$ being the zeroth iterate of (3.2)
is a statement about this particular kernel convention, normal orientation and jump
coefficient; it is used only to pin $C$. Second, agreement of the zeroth iterate with
Michell tests normalisation alone. It exercises neither the Kelvin influence matrix nor
the solve, so it is not a validation of the method, and it is not reported as one.

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

**Consequence for the discretisation.** The NK mesh must keep $k_0|z_i + z_j| \gtrsim 0.1$.
At $Fn = 0.3$ on Sysser 01, where $k_0 = 6.9$ per metre, that means panel centroids at
least about 7 mm below the waterline, against a canoe-body draught of 127 mm. A uniform
mesh with few girth divisions puts the top row closer than that, so the NK mesh needs
grading away from the waterline. This is the concrete form taken by the caveat of
section 3: nothing is missing from the formulation, but the discretisation has to respect
where the kernel is evaluable.

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

## 11. The zeroth iterate

Dropping the integral term from (3.2) leaves $\tfrac{1}{2}\sigma = u\,n_x$, so the zeroth
iterate of this integral equation is $\sigma = 2u\,n_x$ in this kernel convention. Whether
that coincides with Hogner's distribution depends on his normalisation, which is not
reproduced here, so the quantity is used only as a probe for fixing the constant $C$ of
(4.3) and is not presented as a named method. As section 4 already noted, agreement of a
zeroth iterate with Michell tests normalisation alone: it exercises neither the Kelvin
influence matrix nor the solve.

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
