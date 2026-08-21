"""Recomputed hydrostatics of Sysser 01 and Sysser 50 against the published release."""
import json
import numpy as np
from wave_resistance.hull import Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave
from wave_resistance.reference import hull_reference

KEYS = [("volume", "volume", "m^3"), ("waterplane_area", "waterplane_area", "m^2"),
        ("wetted_area", "wetted_area", "m^2"), ("lwl", "lwl", "m"), ("bwl", "bwl", "m"),
        ("draught", "tc", "m"), ("midship_area", "midship_area", "m^2"),
        ("cp", "cp", "-"), ("cm", "cm", "-"), ("cb", "cb", "-"), ("cwp", "cwp", "-")]

out = {}
for n in (1, 50):
    ref = hull_reference(n)
    h = Hull.from_iges(f"data/SYSSER{n:02d}_surface.igs")
    shift = solve_reference_heave(h, ref["volume"])
    h = h.with_datum_shift(shift)
    seq = []
    for nu, nv in ((12, 60), (24, 120), (48, 240), (96, 480)):
        r = hydrostatics(h.mesh(nu, nv))
        seq.append(dict(nu=nu, nv=nv, faces=r.n_faces, volume=r.volume, lwl=r.lwl,
                        bwl=r.bwl, tc=r.draught, sc=r.wetted_area, aw=r.waterplane_area,
                        am=r.midship_area, lcb=r.lcb_from_midship, lcf=r.lcf_from_midship,
                        cp=r.cp, cm=r.cm, cb=r.cb, cwp=r.cwp, spread=r.volume_spread))
    r = hydrostatics(h.mesh(96, 480))
    # Richardson extrapolation on the finest three levels, second order.
    def rich(field):
        a, b, c = (s[field] for s in seq[-3:])
        return c + (c - b) / 3.0
    rows = {}
    for attr, key, unit in KEYS:
        got = float(getattr(r, attr))
        want = float(ref[key])
        rows[key] = dict(computed=got, published=want, unit=unit,
                         pct=100.0 * (got / want - 1.0))
    out[f"sysser{n:02d}"] = dict(
        datum=shift, sequence=seq, rows=rows,
        lcb=dict(computed=float(r.lcb_from_midship), published=float(ref["lcb"]),
                 mm=1000.0 * (r.lcb_from_midship - ref["lcb"])),
        lcf=dict(computed=float(r.lcf_from_midship), published=float(ref["lcf"]),
                 mm=1000.0 * (r.lcf_from_midship - ref["lcf"])),
        bwl_over_tc_published=float(ref["bwl"] / ref["tc"]),
        lcb_pct_lwl=100.0 * float(ref["lcb"]) / float(ref["lwl"]),
        richardson=dict(volume=rich("volume"), sc=rich("sc"), aw=rich("aw")),
        volume_ratio=float((seq[1]["volume"] - seq[0]["volume"])
                           / (seq[2]["volume"] - seq[1]["volume"])),
    )
    print(f"Sysser {n}: datum {shift:.6f}")
    for key, v in rows.items():
        print(f"   {key:<18}{v['computed']:14.6f}{v['published']:14.6f}{v['pct']:+9.3f} %")
    print(f"   lcb {out[f'sysser{n:02d}']['lcb']['mm']:+.2f} mm, "
          f"lcf {out[f'sysser{n:02d}']['lcf']['mm']:+.2f} mm")
json.dump(out, open("/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad/hydro_compare.json", "w"), indent=1)
