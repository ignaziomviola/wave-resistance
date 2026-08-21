"""Sysser 01 and Sysser 50 compared: body plan, profile and the panel mesh."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wave_resistance.hull import Attitude, Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave
from wave_resistance.reference import hull_reference

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.6,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def draw(ax, mesh, normal, levels, ia, ib, dx=0.0, lw=0.5, colour="k"):
    for lev in levels:
        sec = mesh.section(np.array(normal, float), float(lev))
        if sec.segments.size == 0:
            continue
        s = sec.segments
        ax.plot(np.vstack([s[:, 0, ia] - dx, s[:, 1, ia] - dx, np.full(len(s), np.nan)]),
                np.vstack([s[:, 0, ib], s[:, 1, ib], np.full(len(s), np.nan)]),
                color=colour, lw=lw)


hulls = {}
for n in (1, 50):
    ref = hull_reference(n)
    h = Hull.from_iges(f"data/SYSSER{n:02d}_surface.igs")
    h = h.with_datum_shift(solve_reference_heave(h, ref["volume"]))
    wet = h.mesh(48, 240, Attitude()).clipped_below(0.0, axis=2)
    r = hydrostatics(h.mesh(48, 240))
    v = wet.vertices
    hulls[n] = dict(hull=h, wet=wet, r=r,
                    x0=float(v[:, 0].min()), x1=float(v[:, 0].max()))

fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
for col, n in enumerate((1, 50)):
    d = hulls[n]
    xm = 0.5 * (d["x0"] + d["x1"])
    tc = d["r"].draught
    # body plan
    ax = axes[0, col]
    draw(ax, d["wet"], (1, 0, 0), np.linspace(d["x0"] + 0.01, d["x1"] - 0.01, 17), 1, 2)
    draw(ax, d["wet"], (0, 0, 1), np.linspace(-tc * 0.99, -0.001, 7), 1, 2)
    ax.axhline(0.0, color="0.45", lw=0.8, ls="--")
    ax.set_xlim(-0.34, 0.34); ax.set_ylim(-0.14, 0.012)
    ax.set_aspect("equal"); ax.set_xticks([-0.3, 0.0, 0.3])
    ax.set_yticks([-0.1, -0.05, 0.0])
    ax.set_xlabel(r"$y$ [m]")
    if col == 0:
        ax.set_ylabel(r"$z$ [m]")
    ax.set_title(f"({'ab'[col]})", loc="left")
    # plan view
    ax = axes[1, col]
    draw(ax, d["wet"], (0, 0, 1), np.linspace(-tc * 0.99, -0.001, 7), 0, 1, xm)
    draw(ax, d["wet"], (1, 0, 0), np.linspace(d["x0"], d["x1"], 15), 0, 1, xm)
    ax.plot([hull_reference(n)["lcb"]], [0.0], marker="o", ms=3.5, color="k")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.34, 0.34)
    ax.set_aspect("equal"); ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-0.3, 0.0, 0.3])
    ax.set_xlabel(r"$x$ [m]")
    if col == 0:
        ax.set_ylabel(r"$y$ [m]")
    ax.set_title(f"({'cd'[col]})", loc="left")
fig.tight_layout()
fig.savefig("docs/figures/geometry50.pdf", bbox_inches="tight")
fig.savefig("docs/figures/geometry50.png", dpi=220, bbox_inches="tight")
for n in (1, 50):
    r = hulls[n]["r"]
    print(f"Sysser {n}: Lwl {r.lwl:.4f} Bwl {r.bwl:.4f} Tc {r.draught:.5f} "
          f"Bwl/Tc {r.bwl / r.draught:.3f} Cp {r.cp:.4f} Cm {r.cm:.4f} "
          f"LCB%Lwl {100 * r.lcb_from_midship / r.lwl:+.3f}")
