"""Neumann-Kelvin sweep for any Sysser hull, at the static or the measured attitude.

Usage: sweep3.py SYSSER GIRTH STATIONS MODE OUT [FN,FN,...]
MODE is 'static', 'measured' (rotating about the origin, the old and wrong pivot) or
'pivoted' (rotating about the centre of gravity, which the release's Info sheet fixes).
"""
import json, math, sys, time
import numpy as np
from wave_resistance.hull import Attitude, Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave
from wave_resistance.measurements import hull_runs, trim_pivot_x
from wave_resistance.nk import check_envelope, pressure_resistance, solve_nk
from wave_resistance.reference import hull_reference
from wave_resistance.spectrum import mesh_resolved_lambda

G = 9.80665
sysser = int(sys.argv[1])
girth, stations, mode, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
want = [float(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else None

ref = hull_reference(sysser)
igs = f"data/SYSSER{sysser:02d}_surface.igs"
hull = Hull.from_iges(igs)
shift = solve_reference_heave(hull, ref["volume"])
hull = hull.with_datum_shift(shift)
fine = hydrostatics(hull.mesh(48, 240))
length = fine.lwl

base = hull.mesh(48, 240)
x_mid = 0.5 * (float(base.vertices[:, 0].min()) + float(base.vertices[:, 0].max()))
pivot_x = trim_pivot_x(sysser, x_mid)

runs = hull_runs(sysser)
if want is not None:
    runs = [min(runs, key=lambda r: abs(r.froude - w)) for w in want]
print(json.dumps(dict(sysser=sysser, lwl=length, volume=fine.volume,
                      wetted_area=fine.wetted_area, datum=shift, x_mid=x_mid,
                      pivot_x=pivot_x, mode=mode)), flush=True)

records = []
for run in runs:
    if mode == "static":
        att = Attitude()
    elif mode == "measured":
        att = Attitude(sinkage=run.sinkage, trim=run.trim)
    elif mode == "pivoted":
        att = Attitude(sinkage=run.sinkage, trim=run.trim, pivot=(pivot_x, 0.0, 0.0))
    else:
        raise SystemExit(f"unknown mode {mode}")
    try:
        mesh = hull.waterline_fitted_mesh(girth, stations, att, first_depth=0.010)
        wet = hydrostatics(hull.mesh(32, 160, att))
        speed = run.froude * math.sqrt(G * length)
        k0 = G / speed ** 2
        worst, over = check_envelope(mesh, k0)
        cap = mesh_resolved_lambda(mesh, k0)
        t0 = time.time()
        r = solve_nk(mesh, speed, length, order=1, lambda_cap=cap, check_points=40)
        rp = pressure_resistance(mesh, r.sigma, speed, k0, order=1)
        rec = dict(sysser=sysser, mode=mode, fn=run.froude, speed=speed,
                   panels=mesh.n_faces, r_far=r.resistance, r_press=rp,
                   rr_meas=run.residuary_resistance, rt_meas=run.total_resistance,
                   rf_meas=run.friction_resistance,
                   weight_fraction=run.residuary_fraction_of_weight,
                   sinkage=run.sinkage, trim=run.trim, pivot_x=pivot_x,
                   volume_at_attitude=wet.volume, wetted_at_attitude=wet.wetted_area,
                   flux=r.net_source_flux, resid_rms=r.body_residual_rms,
                   resid_max=r.body_residual_max, cond=r.condition_number,
                   lambda_cap=cap, lambda_reached=r.lambda_reached,
                   spectrum_converged=r.spectrum_converged, osc_worst=worst,
                   osc_over=over, seconds=time.time() - t0, notes=r.notes)
    except Exception as exc:
        rec = dict(sysser=sysser, mode=mode, fn=run.froude, panels=None,
                   error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(rec), flush=True)
    records.append(rec)
    json.dump(dict(sysser=sysser, mode=mode, girth=girth, stations=stations,
                   length=length, volume=fine.volume, wetted_area=fine.wetted_area,
                   pivot_x=pivot_x, records=records), open(out, "w"), indent=1)
