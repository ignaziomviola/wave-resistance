"""Independent checks of the claims the repository makes, outside its own test suite."""
import json
import numpy as np
from wave_resistance import greens, spectrum
from wave_resistance.hull import Attitude, Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave
from wave_resistance.reference import sysser01_reference

out = {}

# 1. Displaced volume by four independent divergence-theorem routes.
hull = Hull.from_iges("data/SYSSER01_surface.igs")
shift = solve_reference_heave(hull, 0.0376136)
hull = hull.with_datum_shift(shift)
h = hydrostatics(hull.mesh(48, 240))
routes = np.array(h.volume_routes)
out["volume_routes"] = routes.tolist()
out["volume_route_rel_spread"] = float(np.ptp(routes) / routes.mean())

# 2. Published Sysser 01 hydrostatics, computed against released values.
ref = sysser01_reference()
out["hydro_vs_published"] = {
    k: dict(computed=float(getattr(h, a)), published=float(ref[k]),
            pct=100.0 * (float(getattr(h, a)) / float(ref[k]) - 1.0))
    for a, k in (("lwl", "lwl"), ("bwl", "bwl"), ("draught", "tc"), ("volume", "volume"),
                 ("wetted_area", "wetted_area"), ("waterplane_area", "waterplane_area"),
                 ("cp", "cp"), ("cb", "cb"), ("cwp", "cwp"))}

# 3. Hydrostatic mesh convergence and Richardson extrapolation, recomputed here.
seq = []
for nu, nv in ((12, 60), (24, 120), (48, 240)):
    r = hydrostatics(hull.mesh(nu, nv))
    seq.append((nu, nv, float(r.volume), float(r.lcb_from_midship)))
v = np.array([s[2] for s in seq])
out["volume_sequence"] = seq
out["volume_ratio"] = float((v[1] - v[0]) / (v[2] - v[1]))
out["volume_richardson"] = float(v[2] + (v[2] - v[1]) / 3.0)

# 4. The free-surface condition, checked directly on the Green function.
#    u^2 g_xx + g g_z = 0 on z = 0, i.e. k0 d/dZ = -d^2/dX^2 in scaled variables.
rng = np.random.default_rng(3)
err = []
for _ in range(40):
    X, Y = rng.uniform(-4, 4), rng.uniform(-4, 4)
    Zc, e = -1.0, 1e-4
    def g(x, y, z):
        return greens.wave_part(np.array([x]), np.array([y]), np.array([z]))[0]
    # field point on z = 0 means Z = k0(z + zeta) = zeta k0; use a source at Zc and take
    # the derivative combination at the surface by finite differences in the scaled Z.
    gzz = (g(X + e, Y, Zc) - 2 * g(X, Y, Zc) + g(X - e, Y, Zc)) / e ** 2
    gz = (g(X, Y, Zc + e) - g(X, Y, Zc - e)) / (2 * e)
    err.append(abs(gzz + gz) / max(abs(gz), 1e-30))
out["free_surface_residual"] = dict(median=float(np.median(err)), worst=float(max(err)))

# 5. The gradient of the wave part against central differences of the wave part itself.
rng = np.random.default_rng(11)
worst = 0.0
for _ in range(60):
    X, Y = rng.uniform(-3, 3), rng.uniform(-2, 2)
    Z = -10 ** rng.uniform(-2, 0.3)
    e = 1e-5
    an = greens.wave_part_gradient(np.array([X]), np.array([Y]), np.array([Z]))[0]
    fd = np.array([
        (greens.wave_part(np.array([X + e]), np.array([Y]), np.array([Z]))[0]
         - greens.wave_part(np.array([X - e]), np.array([Y]), np.array([Z]))[0]) / (2 * e),
        (greens.wave_part(np.array([X]), np.array([Y + e]), np.array([Z]))[0]
         - greens.wave_part(np.array([X]), np.array([Y - e]), np.array([Z]))[0]) / (2 * e),
        (greens.wave_part(np.array([X]), np.array([Y]), np.array([Z + e]))[0]
         - greens.wave_part(np.array([X]), np.array([Y]), np.array([Z - e]))[0]) / (2 * e)])
    worst = max(worst, float(np.max(np.abs(an - fd) / max(np.max(np.abs(fd)), 1e-30))))
out["gradient_vs_fd_worst"] = worst

# 6. The lambda cap can fall below the physical lower limit lambda = 1.
G = 9.80665
mesh = hull.waterline_fitted_mesh(5, 20, Attitude(), first_depth=0.010)
caps = {}
for fn in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
    k0 = G / (fn * np.sqrt(G * h.lwl)) ** 2
    caps[f"{fn:.2f}"] = float(spectrum.mesh_resolved_lambda(mesh, k0))
out["lambda_cap_by_fn"] = caps

print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad/verify.json", "w"), indent=1)
