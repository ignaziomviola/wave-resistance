"""Neumann-Kelvin sweep over Froude number for Sysser 01, at a chosen mesh and attitude."""
import json, math, sys, time
import numpy as np
from wave_resistance.hull import Attitude, Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave
from wave_resistance.measurements import sysser01_runs
from wave_resistance.nk import pressure_resistance, solve_nk, check_envelope
from wave_resistance.spectrum import mesh_resolved_lambda

G, VOL, IGS = 9.80665, 0.0376136, "data/SYSSER01_surface.igs"
girth, stations, mode, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
want = [float(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else None

hull = Hull.from_iges(IGS)
shift = solve_reference_heave(hull, VOL)
hull = hull.with_datum_shift(shift)
fine = hydrostatics(hull.mesh(48, 240))
length = fine.lwl
runs = sysser01_runs()
if want is not None:
    runs = [min(runs, key=lambda r: abs(r.froude - w)) for w in want]
print(f"Lwl {length:.6f} vol {fine.volume:.6f} Sc {fine.wetted_area:.6f} datum {shift:.6f}",
      flush=True)

records = []
for run in runs:
    att = Attitude() if mode == "static" else Attitude(sinkage=run.sinkage, trim=run.trim)
    try:
        mesh = hull.waterline_fitted_mesh(girth, stations, att, first_depth=0.010)
        speed = run.froude * math.sqrt(G * length)
        k0 = G / speed ** 2
        worst, over = check_envelope(mesh, k0)
        cap = mesh_resolved_lambda(mesh, k0)
        t0 = time.time()
        r = solve_nk(mesh, speed, length, order=1, lambda_cap=cap, check_points=40)
        rp = pressure_resistance(mesh, r.sigma, speed, k0, order=1)
        rec = dict(fn=run.froude, speed=speed, panels=mesh.n_faces, r_far=r.resistance,
                   r_press=rp, rr_meas=run.residuary_resistance,
                   rt_meas=run.total_resistance, rf_meas=run.friction_resistance,
                   weight_fraction=run.residuary_fraction_of_weight,
                   sinkage=run.sinkage, trim=run.trim, flux=r.net_source_flux,
                   resid_rms=r.body_residual_rms, resid_max=r.body_residual_max,
                   cond=r.condition_number, lambda_cap=cap,
                   lambda_reached=r.lambda_reached,
                   spectrum_converged=r.spectrum_converged, osc_worst=worst,
                   osc_over=over, seconds=time.time() - t0, notes=r.notes)
    except Exception as exc:
        rec = dict(fn=run.froude, panels=None, error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(rec), flush=True)
    records.append(rec)
    json.dump(dict(mode=mode, girth=girth, stations=stations, length=length,
                   volume=fine.volume, wetted_area=fine.wetted_area, records=records),
              open(out, "w"), indent=1)
