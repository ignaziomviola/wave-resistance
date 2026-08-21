"""End-to-end check of the resistance constant: the full NK solve against Michell."""
import json
import numpy as np
from scipy.integrate import quad
from wave_resistance.nk import pressure_resistance, solve_nk
from wave_resistance.spectrum import mesh_resolved_lambda
from wave_resistance.wigley import (michell_resistance_from_integral, wigley_mesh,
                                    wigley_michell_amplitude)

G = 9.80665
LEN, BOL, TOL = 1.0, 0.02, 0.0625
out = {}
for fn in (0.30, 0.40):
    # Michell oracle, integrated here rather than taken from the repository's test.
    f = lambda t: np.sqrt(1 + t * t) * abs(
        wigley_michell_amplitude(np.sqrt(1 + t * t), fn, BOL, TOL)) ** 2
    integral = quad(f, 0.0, 60.0, limit=400)[0]
    oracle = michell_resistance_from_integral(integral, fn, LEN)

    row = {"michell": oracle}
    for nx, nz in ((14, 5), (20, 7)):
        mesh = wigley_mesh(LEN, BOL, TOL, n_x=nx, n_z=nz)
        speed = fn * np.sqrt(G * LEN)
        k0 = G / speed ** 2
        r = solve_nk(mesh, speed, LEN, order=1, spectrum_order=3, check_points=24,
                     lambda_cap=mesh_resolved_lambda(mesh, k0))
        p = pressure_resistance(mesh, r.sigma, speed, k0, order=1)
        row[f"N{mesh.n_faces}"] = dict(far=r.resistance, press=p,
                                       far_over_oracle=r.resistance / oracle,
                                       press_over_oracle=p / oracle,
                                       flux=r.net_source_flux,
                                       resid=r.body_residual_rms)
    out[f"fn{fn:.2f}"] = row
    print(json.dumps({f"fn{fn:.2f}": row}, indent=1), flush=True)
json.dump(out, open("/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad/michell.json", "w"), indent=1)
