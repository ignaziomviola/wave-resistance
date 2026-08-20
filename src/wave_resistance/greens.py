"""The Kelvin (Neumann-Kelvin) Green function.

Conventions and the full derivation are in ``docs/formulation.md``.  In brief: the origin
lies in the undisturbed free surface, z is up, the fluid fills z < 0, the onset flow runs
in the -x direction so upstream is x -> +inf, and k0 = g/u^2.

For a unit source at Q = (xi, eta, zeta) with zeta < 0 and a field point P = (x, y, z),

    G = -1/(4 pi R) + 1/(4 pi R1) + k0 * g_w(X, Y, Z),
    X = k0 (x - xi),   Y = k0 (y - eta),   Z = k0 (z + zeta) <= 0,

where R and R1 are the distances to the source and to its image at (xi, eta, -zeta).
Only those three nondimensional arguments enter, which is what will later make tabulation
possible.

The wave part is a single integral over wave angle theta,

    g_w = (1 / 2 pi^2) Re integral over theta in (-pi/2, pi/2) of
              [ P(c) + i pi (s - sgn(Im c)) exp(-c) ] sec^2(theta) d(theta),
    c(theta) = sec^2(theta) [ -Z - i (X cos(theta) + Y sin(theta)) ],

with P(c) = exp(-c) E_1(-c) and s = +/-1 the side the dispersion-relation pole is passed
on, i.e. the radiation condition.

Two identities do the work here, and both were verified numerically against direct
quadrature before being used (see ``tests/test_greens.py``):

    PV integral over p in [0, inf) of exp(-p c)/(p - 1) dp
        = exp(-c) [ E_1(c) - 2 Shi(c) ]                       to 2e-16
        = exp(-c) [ E_1(-c) - i pi sgn(Im c) ]                to 3e-10

The second removes an entire quadrature level and, combined with the residue term, leaves
the integrand equal to P(c) wherever sgn(Im c) = s and to P(c) + 2 i pi s exp(-c)
elsewhere.  That switch is the Kelvin wake: the free-wave term contributes only on the
half of the wave-angle range where the phase condition holds.  The real part is continuous
across the switch, because Im c vanishes there and so does Im exp(-c).

The integrand is bounded at theta = +/- pi/2: sec^2(theta) diverges but P(c) ~ -1/c
vanishes as cos^2(theta), and the free-wave term is exponentially damped.  The endpoint
value is Re[-1/(-Z mp i Y)].
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.special import exp1

__all__ = [
    "q_function", "p_function", "wave_part", "rankine_part", "green",
    "calibrate_radiation_sign", "RADIATION_SIGN",
]

#: |c| above which the asymptotic series is used for P.  P is the "clean" branch, so its
#: asymptotic expansion is valid at any phase; Q's is not, because Q carries a pole term
#: of size pi exp(-Re c) that the 1/c series omits.
_ASYMPTOTIC_CUTOFF = 30.0
_GL_ORDER = 96

#: Branch the dispersion-relation pole is passed on.  Determined by
#: :func:`calibrate_radiation_sign` and pinned here rather than computed at import, so a
#: change is a visible edit; the test suite re-derives it.
RADIATION_SIGN = -1


@functools.lru_cache(maxsize=8)
def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


@functools.lru_cache(maxsize=64)
def _asymptotic_terms(c_min: float) -> int:
    """Smallest truncation K with K!/c_min^K below double precision, capped at 60.

    The series -sum n!/c^(n+1) is asymptotic, not convergent: its terms shrink only until
    n is about |c| and grow thereafter, so K must not exceed |c| and the accuracy floor is
    exp(-|c|).  That floor is why the cutoff below sits at |c| = 30, where it is 1e-13.
    """
    log_fact = 0.0
    best_k, best_log = 1, np.inf
    for k in range(1, 61):
        log_fact += np.log(k)
        log_term = log_fact - k * np.log(c_min)
        if log_term < np.log(1e-17):
            return k
        if log_term < best_log:
            best_k, best_log = k, log_term
    # Target unreachable: stop at the smallest term rather than running on into the
    # divergent tail, which at |c| = 30 turned a 1e-12 floor into a 100 per cent error.
    return best_k


def _asymptotic(c: np.ndarray) -> np.ndarray:
    """-sum over n >= 0 of n!/c^(n+1), by Horner in 1/c with a banded term count.

    Horner needs K complex multiply-adds where the naive loop needed K passes over the
    whole array plus a termination test; banding by octave of |c| lets the far field, where
    a handful of terms suffice, avoid paying for the near field's fifty.
    """
    out = np.zeros_like(c)
    mag = np.abs(c)
    # Bands must cover [0, inf) with no gap.  Starting at the cutoff left values that
    # rounded just below it -- |c| = 29.999999999999996 from 30 * exp(i phi) -- in
    # uninitialised memory, which read as a 100 per cent error.  Accuracy is only claimed
    # above _ASYMPTOTIC_CUTOFF; p_function enforces that.
    edges = np.array([30.0, 45.0, 70.0, 110.0, 200.0, 500.0, 2000.0, np.inf])
    lo = 0.0
    for hi in edges:
        sel = (mag >= lo) & (mag < hi)
        if not np.any(sel):
            lo = hi
            continue
        cc = c[sel]
        k = _asymptotic_terms(max(lo, 2.0))
        acc = np.ones_like(cc)
        for n in range(k, 0, -1):
            acc = 1.0 + (n / cc) * acc
        out[sel] = -acc / cc
        lo = hi
    return out


def p_function(c: np.ndarray) -> np.ndarray:
    """P(c) = exp(-c) E_1(-c), the pole-free branch, for Re c > 0."""
    c = np.asarray(c, dtype=complex)
    out = np.empty(c.shape, dtype=complex)
    big = np.abs(c) > _ASYMPTOTIC_CUTOFF
    if np.any(big):
        out[big] = _asymptotic(c[big])
    small = ~big
    if np.any(small):
        cs = c[small]
        out[small] = np.exp(-cs) * exp1(-cs)
    return out


def q_function(c: np.ndarray) -> np.ndarray:
    """Q(c) = PV integral over p in [0, inf) of exp(-p c)/(p - 1) dp, for Re c > 0.

    Retained because it is the quantity the derivation produces and the quantity the tests
    check against direct quadrature; the solver uses :func:`p_function`.
    """
    c = np.asarray(c, dtype=complex)
    return p_function(c) - 1j * np.pi * np.sign(c.imag) * np.exp(-c)


def _normalised_grid(n_inner: int, n_core: int, n_tail: int, order: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes on a normalised parameter xi in [0, 2], shared by all points.

    The grid is built in xi rather than in tan(theta) so that one grid serves every point
    in a batch.  Writing t = tan(theta) = t_max * xi for xi in [0, 1], where
    t_max = sqrt(reach/|Z| - 1) is where the damping exp(Z sec^2 theta) has died, the
    damping becomes exp(Z) exp(-reach * xi^2), which is the *same* function of xi for every
    point.  The wave phase over the inner range is
    40 |Y|/|Z| xi^2 + |X| sqrt(40/|Z|) xi, so its panel count depends only on the cycle
    count, which is what the caller bins on.

    Grading in xi is uniform in xi^2, because the phase grows quadratically; a short core
    uniform in xi resolves the linear |X| term near the origin.  xi in [1, 2] maps
    affinely onto theta in [arctan(t_max), pi/2] and closes the range: t_max bounds only
    the free-wave term, while the local part P(c) ~ -1/c decays algebraically and its
    tail is not negligible.

    Building the grid in absolute t instead is what went wrong earlier.  A point whose own
    t_max was far below the batch maximum then had its entire oscillatory range covered by
    a fraction of a panel, giving a 68 per cent error at (X, Y, Z) = (-12.55, 1.80, -0.152)
    in a batch where the same point evaluated alone was right to nine figures.
    """
    core = np.linspace(0.0, 0.15, n_core + 1)
    inner = np.sqrt(np.linspace(0.15 ** 2, 1.0, n_inner + 1)[1:])
    tail = np.linspace(1.0, 2.0, n_tail + 1)[1:]
    positive = np.concatenate([core, inner, tail])
    # Both halves of the wave-angle range are needed: the integrand carries Y sin(theta)
    # and so is not even in theta.  Covering only theta > 0 cost 76 per cent of g_w.
    edges = np.concatenate([-positive[::-1][:-1], positive])
    nodes, weights = _gauss_legendre(order)
    mid = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1:] - edges[:-1])
    return ((mid[:, None] + half[:, None] * nodes[None, :]).ravel(),
            (half[:, None] * weights[None, :]).ravel())


def _theta_of_xi(xi: np.ndarray, t_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map the normalised parameter to wave angle, with the Jacobian d(theta)/d(xi).

    ``xi`` has shape (n_nodes,) and ``t_max`` shape (n_points,); both outputs are
    (n_nodes, n_points).
    """
    xi = xi[:, None]
    tm = t_max[None, :]
    sgn = np.sign(xi)
    mag = np.abs(xi)
    theta1 = np.arctan(tm)
    t = tm * mag
    theta_in = np.arctan(t)
    jac_in = tm / (1.0 + t * t)
    span = 0.5 * np.pi - theta1
    theta_tail = theta1 + (mag - 1.0) * span
    jac_tail = np.broadcast_to(span, theta_tail.shape)
    inner = mag <= 1.0
    return (sgn * np.where(inner, theta_in, theta_tail),
            np.where(inner, jac_in, jac_tail))


def _integrand(theta: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
               sign: int) -> np.ndarray:
    """Real part of [P(c) + i pi (s - sgn(Im c)) exp(-c)] sec^2(theta).

    ``theta`` is (n_nodes, n_points) and the coordinates are (n_points,).
    """
    sec2 = 1.0 / np.cos(theta) ** 2
    c = sec2 * (-Z[None, :] - 1j * (X[None, :] * np.cos(theta) + Y[None, :] * np.sin(theta)))
    integ = p_function(c) + 1j * np.pi * (sign - np.sign(c.imag)) * np.exp(-c)
    return (integ * sec2).real


#: Coverage of the damping exp(Z sec^2 theta): beyond this the free-wave term is below
#: double precision.
_DAMPING_REACH = 40.0
#: Wave cycles one order-12 Gauss-Legendre panel is trusted to resolve.
_CYCLES_PER_PANEL = 1.5
_MAX_PANELS = 6000


def wave_part(X, Y, Z, sign: int | None = None, refine: float = 1.0) -> np.ndarray:
    """Nondimensional wave part g_w(X, Y, Z) of the Kelvin Green function.

    ``Z`` must be strictly negative: the free-surface condition holds only for sources
    strictly below z = 0, and g_w diverges logarithmically as Z -> 0.

    Points are binned by the number of wave cycles their integrand carries, and each bin
    is evaluated on one shared normalised grid.  Most panel pairs on a hull sit close
    together and carry few cycles; only the far-apart, near-surface pairs are expensive.
    ``refine`` scales every bin's node count and exists so convergence can be shown.
    """
    X, Y, Z = (np.asarray(v, dtype=float) for v in (X, Y, Z))
    shape = np.broadcast(X, Y, Z).shape
    X, Y, Z = (np.broadcast_to(v, shape).ravel().astype(float) for v in (X, Y, Z))
    if np.any(Z >= 0.0):
        raise ValueError("Z = k0 (z + zeta) must be strictly negative")
    if sign is None:
        sign = RADIATION_SIGN

    absZ = np.abs(Z)
    t_max = np.sqrt(np.maximum(_DAMPING_REACH / absZ - 1.0, 1.0))
    # Phase accumulated over the inner range, in cycles, and its linear part.
    cycles_quad = _DAMPING_REACH * np.abs(Y) / absZ / (2.0 * np.pi)
    cycles_lin = np.abs(X) * t_max / (2.0 * np.pi)
    cycles_total = cycles_quad + cycles_lin

    out = np.empty(X.size)
    key = np.zeros(X.size, dtype=np.int64)
    busy = cycles_total > 1.0
    key[busy] = np.ceil(np.log2(cycles_total[busy])).astype(np.int64)
    key = np.clip(key, 0, 18)
    for k in np.unique(key):
        sel = np.flatnonzero(key == k)
        n_inner = int(min(_MAX_PANELS, np.ceil(refine * max(
            16.0, cycles_total[sel].max() / _CYCLES_PER_PANEL))))
        n_core = int(min(1000, np.ceil(refine * max(6.0, cycles_lin[sel].max() / 4.0))))
        xi, w = _normalised_grid(n_inner, n_core, 12, 12)
        chunk = max(1, int(3e6 // max(xi.size, 1)))
        for start in range(0, sel.size, chunk):
            part = sel[start:start + chunk]
            theta, jac = _theta_of_xi(xi, t_max[part])
            vals = _integrand(theta, X[part], Y[part], Z[part], sign) * jac
            out[part] = w @ vals
    return (out / (2.0 * np.pi ** 2)).reshape(shape)


def rankine_part(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """-1/(4 pi R) + 1/(4 pi R1): the source and its negative image in z = 0.

    This pair vanishes on z = 0, so it is the zero-potential image combination rather than
    the rigid-wall one; the wave part supplies the remainder.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.linalg.norm(p - q, axis=-1)
    image = np.stack([q[..., 0], q[..., 1], -q[..., 2]], axis=-1)
    r1 = np.linalg.norm(p - image, axis=-1)
    with np.errstate(divide="ignore"):
        return -1.0 / (4.0 * np.pi * r) + 1.0 / (4.0 * np.pi * r1)


def green(p: np.ndarray, q: np.ndarray, k0: float, sign: int | None = None,
          **kwargs) -> np.ndarray:
    """Full Kelvin Green function at field points ``p`` due to unit sources at ``q``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    X = k0 * (p[..., 0] - q[..., 0])
    Y = k0 * (p[..., 1] - q[..., 1])
    Z = k0 * (p[..., 2] + q[..., 2])
    return rankine_part(p, q) + k0 * wave_part(X, Y, Z, sign=sign, **kwargs)


def calibrate_radiation_sign(depth: float = 0.4, reach: float = 30.0,
                             n: int = 160) -> int:
    """Return the branch sign for which the wave field vanishes upstream.

    Upstream is x -> +inf here, because the onset flow runs in -x.  The two candidate
    signs put the wave train on opposite sides, so comparing the far-field variation ahead
    of and behind the source settles the choice without appeal to a damping convention
    that is easy to invert on paper.
    """
    Z = -2.0 * depth
    ahead = np.linspace(0.4 * reach, reach, n)
    zeros = np.zeros_like(ahead)
    zs = np.full_like(ahead, Z)
    ratios = {}
    for sign in (+1, -1):
        up = np.ptp(wave_part(ahead, zeros, zs, sign=sign))
        down = np.ptp(wave_part(-ahead, zeros, zs, sign=sign))
        ratios[sign] = up / down if down > 0.0 else np.inf
    best = min(ratios, key=ratios.get)
    other = -best
    if not ratios[best] < 0.1 * ratios[other]:
        raise RuntimeError(
            "radiation sign unresolved: upstream/downstream wave amplitude ratios were "
            f"{ratios}; one sign should be far quieter upstream"
        )
    return best
