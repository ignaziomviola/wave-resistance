"""Sysser 50 against Sysser 01, and the effect of the running attitude."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.6,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
B = "/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad/"


def load(name):
    d = json.load(open(B + name))
    return d, sorted([r for r in d["records"] if r.get("panels")], key=lambda r: r["fn"])


d01, s01 = load("s280_static.json")
d50, s50 = load("s50_static.json")
try:
    _, p50 = load("s50_pivot.json")
except Exception:
    p50 = []
try:
    _, p01 = load("s01_pivot.json")
except Exception:
    p01 = []

col = lambda rs, k: np.array([r[k] for r in rs])

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))

# (a) both hulls, resistance made nondimensional on displacement weight
ax = axes[0]
for rs, lab, ls, mk in ((s01, "Sysser 01", "-", "o"), (s50, "Sysser 50", "--", "s")):
    w = col(rs, "rr_meas") / col(rs, "weight_fraction").clip(1e-12)
    ax.plot(col(rs, "fn"), 100 * col(rs, "weight_fraction"), ls, color="k", lw=1.2,
            marker=mk, ms=3.4, mfc="k" if mk == "o" else "w", label=lab + r", $R_r$")
    ax.plot(col(rs, "fn"), 100 * col(rs, "r_press") / w, ls, color="0.45", lw=0.9,
            marker=mk, ms=3.4, mfc="w", label=lab + r", $R_W$")
ax.axhline(0.0, color="0.7", lw=0.5)
ax.set_xlabel(r"$\mathit{Fn}$"); ax.set_ylabel(r"$R/W$ [\%]")
ax.set_xlim(0.0, 0.7); ax.set_ylim(-1.0, 14.0)
ax.set_xticks([0.0, 0.2, 0.4, 0.6]); ax.set_yticks([0, 5, 10])
ax.legend(frameon=False, fontsize=6.2, loc="upper left", handlelength=2.0)
ax.set_title("(a)", loc="left")

# (b) the running attitude, applied at the pivot the release determines
ax = axes[1]
for rs, ps, lab, mk in ((s01, p01, "Sysser 01", "o"), (s50, p50, "Sysser 50", "s")):
    if not ps:
        continue
    fns = col(ps, "fn")
    meas = {round(r["fn"], 2): r for r in rs}
    stat = np.array([meas[round(f, 2)]["r_press"] for f in fns])
    rr = col(ps, "rr_meas")
    ax.plot(fns, stat / rr, "-", color="k", lw=1.0, marker=mk, ms=3.6,
            mfc="w", label=lab + ", static")
    ax.plot(fns, col(ps, "r_press") / rr, "--", color="k", lw=1.0, marker=mk, ms=3.6,
            mfc="k", label=lab + ", running")
ax.axhline(1.0, color="0.4", lw=0.7)
ax.set_xlabel(r"$\mathit{Fn}$"); ax.set_ylabel(r"$R_W/R_r$")
ax.set_xlim(0.35, 0.65); ax.set_ylim(0.0, 4.0)
ax.set_xticks([0.4, 0.5, 0.6]); ax.set_yticks([0, 1, 2, 3, 4])
ax.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=2.0)
ax.set_title("(b)", loc="left")

fig.tight_layout()
fig.savefig("docs/figures/sysser50.pdf", bbox_inches="tight")
fig.savefig("docs/figures/sysser50.png", dpi=220, bbox_inches="tight")
print("figure written;", len(s50), "static points,", len(p50), "+", len(p01), "pivoted")
