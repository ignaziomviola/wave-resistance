"""Lines plan and panel mesh of Sysser 01, for the report figures."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from wave_resistance.hull import Attitude, Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.6,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

VOL = 0.0376136
hull = Hull.from_iges("data/SYSSER01_surface.igs")
shift = solve_reference_heave(hull, VOL)
hull = hull.with_datum_shift(shift)
full = hull.mesh(48, 240, Attitude())
wet = full.clipped_below(0.0, axis=2)
h = hydrostatics(full)
json.dump(dict(lwl=h.lwl, bwl=h.bwl, tc=h.draught, vol=h.volume, sc=h.wetted_area,
               awp=h.waterplane_area, cp=h.cp, cm=h.cm, cb=h.cb, cwp=h.cwp,
               lcb=h.lcb_from_midship, lcf=h.lcf_from_midship, vcb=h.vcb,
               datum=shift), open("/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad/hydro.json", "w"), indent=1)

v = wet.vertices
x0, x1 = float(v[:, 0].min()), float(v[:, 0].max())
xm = 0.5 * (x0 + x1)
lwl, bwl, tc = h.lwl, h.bwl, h.draught


def draw(ax, mesh, normal, levels, ia, ib, dx=0.0):
    for lev in levels:
        sec = mesh.section(np.array(normal, float), float(lev))
        if sec.segments.size == 0:
            continue
        s = sec.segments
        ax.plot(np.vstack([s[:, 0, ia] - dx, s[:, 1, ia] - dx, np.full(len(s), np.nan)]),
                np.vstack([s[:, 0, ib], s[:, 1, ib], np.full(len(s), np.nan)]),
                color="k", lw=0.5)


fig = plt.figure(figsize=(6.0, 6.2))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.35], hspace=0.45)

# (a) profile: buttock lines, and the reference frame
ax = fig.add_subplot(gs[0])
draw(ax, wet, (0, 1, 0), np.linspace(0.0, 0.24, 7), 0, 2, xm)
draw(ax, wet, (1, 0, 0), np.linspace(x0, x1, 15), 0, 2, xm)
ax.axhline(0.0, color="0.45", lw=0.8, ls="--")
ax.annotate("", xy=(0.34, 0.045), xytext=(0.70, 0.045),
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.text(0.72, 0.045, r"$\mathbf{u}_\infty$", va="center", ha="left")
ax.annotate("", xy=(-0.5 * lwl, -0.17), xytext=(0.5 * lwl, -0.17),
            arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(0.0, -0.163, r"$c$", ha="center", va="bottom", bbox=dict(fc="white", ec="none", pad=0.6))
ax.annotate("", xy=(-0.60, 0.0), xytext=(-0.60, -tc),
            arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(-0.585, -0.5 * tc, r"$d$", ha="left", va="center", bbox=dict(fc="white", ec="none", pad=0.6))
ax.plot([0.5 * lwl], [0.0], marker="o", ms=2.5, color="k")
ax.text(0.5 * lwl - 0.02, 0.055, "bow", ha="right", va="bottom", fontsize=8)
ax.text(-0.5 * lwl + 0.02, 0.055, "stern", ha="left", va="bottom", fontsize=8)
ax.set_xlabel(r"$x$ [m]"); ax.set_ylabel(r"$z$ [m]")
ax.set_ylim(-0.20, 0.09); ax.set_aspect("equal")
ax.set_yticks([-0.2, -0.1, 0.0]); ax.set_title("(a)", loc="left")

# (b) plan: waterlines
ax = fig.add_subplot(gs[1])
draw(ax, wet, (0, 0, 1), np.linspace(-0.125, -0.001, 8), 0, 1, xm)
draw(ax, wet, (1, 0, 0), np.linspace(x0, x1, 15), 0, 1, xm)
ax.annotate("", xy=(0.0, -0.5 * bwl), xytext=(0.0, 0.5 * bwl),
            arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(0.015, 0.0, r"$b$", ha="left", va="center", bbox=dict(fc="white", ec="none", pad=0.6))
ax.set_xlabel(r"$x$ [m]"); ax.set_ylabel(r"$y$ [m]")
ax.set_aspect("equal"); ax.set_yticks([-0.2, 0.0, 0.2])
ax.set_title("(b)", loc="left")

# (c) body plan: station sections
ax = fig.add_subplot(gs[2])
draw(ax, wet, (1, 0, 0), np.linspace(x0 + 0.01, x1 - 0.01, 17), 1, 2)
draw(ax, wet, (0, 0, 1), np.linspace(-0.125, -0.001, 8), 1, 2)
ax.axhline(0.0, color="0.45", lw=0.8, ls="--")
ax.set_xlabel(r"$y$ [m]"); ax.set_ylabel(r"$z$ [m]")
ax.set_aspect("equal"); ax.set_xticks([-0.2, 0.0, 0.2])
ax.set_yticks([-0.1, -0.05, 0.0]); ax.set_title("(c)", loc="left")

fig.savefig("docs/figures/geometry.pdf", bbox_inches="tight")
fig.savefig("docs/figures/geometry.png", dpi=200, bbox_inches="tight")

# ---- panel mesh ----
panel = hull.waterline_fitted_mesh(6, 28, Attitude(), first_depth=0.010)
print("panels", panel.n_faces, "area", panel.areas().sum(),
      "top centroid mm", -1000 * panel.centroids()[:, 2].max())
fig = plt.figure(figsize=(6.0, 3.4))
ax = fig.add_subplot(111, projection="3d")
tri = panel.triangles().copy()
tri[:, :, 0] -= xm
ax.add_collection3d(Poly3DCollection(tri, facecolors="0.85", edgecolors="0.2",
                                     linewidths=0.3))
ax.plot([-0.9, 0.9, 0.9, -0.9, -0.9], [-0.32, -0.32, 0.32, 0.32, -0.32],
        [0, 0, 0, 0, 0], color="0.55", lw=0.7, ls="--")
ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.32, 0.32); ax.set_zlim(-0.14, 0.0)
ax.set_box_aspect((1.0, 0.42, 0.30))
ax.view_init(elev=24, azim=-132)
ax.set_xlabel(r"$x$ [m]", labelpad=2); ax.set_ylabel(r"$y$ [m]", labelpad=-2)
ax.set_zlabel(r"$z$ [m]", labelpad=-6)
ax.set_xticks([-0.8, 0.0, 0.8]); ax.set_yticks([-0.25, 0.0, 0.25])
ax.set_zticks([-0.1, 0.0])
ax.tick_params(axis="x", pad=-2); ax.tick_params(axis="y", pad=-3)
ax.tick_params(axis="z", pad=-1)
fig.savefig("docs/figures/mesh.pdf", bbox_inches="tight")
fig.savefig("docs/figures/mesh.png", dpi=220, bbox_inches="tight")
print("done")
