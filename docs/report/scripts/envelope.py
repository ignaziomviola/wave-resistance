"""Gradient accuracy of the wave kernel over the panel pairs a Sysser 01 mesh really forms."""
import json
import numpy as np
from wave_resistance import greens
from wave_resistance.hull import Attitude, Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave

G = 9.80665
hull = Hull.from_iges("data/SYSSER01_surface.igs")
hull = hull.with_datum_shift(solve_reference_heave(hull, 0.0376136))
lwl = hydrostatics(hull.mesh(48, 240)).lwl
mesh = hull.waterline_fitted_mesh(6, 28, Attitude(), first_depth=0.010)
c = mesh.centroids()

out = {}
for fn in (0.30, 0.45):
    k0 = G / (fn * np.sqrt(G * lwl)) ** 2
    X = k0 * (c[:, None, 0] - c[None, :, 0])
    Y = k0 * (c[:, None, 1] - c[None, :, 1])
    Z = np.minimum(k0 * (c[:, None, 2] + c[None, :, 2]), -1e-9)
    rng = np.random.default_rng(7)
    idx = rng.choice(X.size, size=4000, replace=False)
    x, y, z = X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx]
    prod = greens.wave_part_gradient(x, y, z)
    ref = greens.wave_part_gradient(x, y, z, refine=8.0)
    scale = np.maximum(np.max(np.abs(ref), axis=1), 1e-30)
    rel = np.max(np.abs(prod - ref), axis=1) / scale
    order = np.argsort(rel)[::-1]
    out[f"fn{fn:.2f}"] = dict(
        median=float(np.median(rel)), p99=float(np.quantile(rel, 0.99)),
        worst=float(rel.max()),
        worst_point=[float(x[order[0]]), float(y[order[0]]), float(z[order[0]])],
        z_range=[float(z.min()), float(z.max())],
        abs_x_max=float(np.abs(x).max()), abs_y_max=float(np.abs(y).max()))
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad/envelope.json", "w"), indent=1)
