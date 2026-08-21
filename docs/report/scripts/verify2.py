"""Checks 4 and 5, done properly: the full Green function, and a step-scaled difference."""
import json
import numpy as np
from wave_resistance import greens

A = lambda v: np.array([float(v)])


def gw(X, Y, Z):
    return float(greens.wave_part(A(X), A(Y), A(Z))[0])


def g_total(X, Y, Z, Zeta):
    """The full scaled Green function 4 pi G / (-1) ... in scaled variables.

    With X = k0(x - xi), Y = k0(y - eta), and the source at scaled depth Zeta = k0 zeta,
    the field point is at scaled height Zf = Z - Zeta.  The Rankine pair is
    -1/(4 pi R) + 1/(4 pi R1); in scaled lengths both carry a factor k0/(4 pi).
    """
    Zf = Z - Zeta
    r = np.sqrt(X ** 2 + Y ** 2 + (Zf - Zeta) ** 2)
    r1 = np.sqrt(X ** 2 + Y ** 2 + (Zf + Zeta) ** 2)
    return (-1.0 / (4 * np.pi * r) + 1.0 / (4 * np.pi * r1)) + gw(X, Y, Z)


# --- 4. free-surface condition g_XX + g_Z = 0 at the field point z = 0 ---
# z = 0 means Zf = 0, so Z = Zeta.  Vary the field point about z = 0 at fixed source.
rng = np.random.default_rng(3)
res = []
for _ in range(30):
    X, Y = rng.uniform(-4, 4), rng.uniform(-4, 4)
    Zeta = -1.0
    e = 2e-4
    f = lambda dx, dz: g_total(X + dx, Y, Zeta + dz, Zeta)
    gxx = (f(e, 0) - 2 * f(0, 0) + f(-e, 0)) / e ** 2
    gz = (f(0, e) - f(0, -e)) / (2 * e)
    res.append(abs(gxx + gz) / max(abs(gz), abs(gxx), 1e-30))
out = {"free_surface_residual": dict(median=float(np.median(res)), worst=float(max(res)))}

# --- 5. gradient of the wave part against a step-scaled central difference ---
rng = np.random.default_rng(11)
rows = []
for _ in range(60):
    X, Y = rng.uniform(-3, 3), rng.uniform(-2, 2)
    Z = -10 ** rng.uniform(-2, 0.3)
    e = 1e-4 * abs(Z)
    an = greens.wave_part_gradient(A(X), A(Y), A(Z))[0]
    fd = np.array([(gw(X + e, Y, Z) - gw(X - e, Y, Z)) / (2 * e),
                   (gw(X, Y + e, Z) - gw(X, Y - e, Z)) / (2 * e),
                   (gw(X, Y, Z + e) - gw(X, Y, Z - e)) / (2 * e)])
    rows.append(float(np.max(np.abs(an - fd)) / max(np.max(np.abs(fd)), 1e-30)))
out["gradient_vs_fd"] = dict(median=float(np.median(rows)), worst=float(max(rows)))
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad/verify2.json", "w"), indent=1)
