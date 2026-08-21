"""Computed and measured resistance against Froude number, one panel per hull."""
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
    rs = sorted([r for r in d["records"] if r.get("panels")], key=lambda r: r["fn"])
    return (np.array([r["fn"] for r in rs]),
            np.array([r["rr_meas"] for r in rs]),
            np.array([r["r_press"] for r in rs]),
            rs[0]["panels"])


fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharex=True, sharey=True)
for ax, (name, label, tag) in zip(axes, (("s280_static.json", "Sysser 01", "a"),
                                         ("s50_static.json", "Sysser 50", "b"))):
    fn, rr, rw, n = load(name)
    ax.plot(fn, rr, "k-", lw=1.2, marker="o", ms=3.8, label=r"$R_r$, measured")
    ax.plot(fn, rw, "k--", lw=1.0, marker="s", ms=3.8, mfc="w", label=r"$R_W$, computed")
    ax.axhline(0.0, color="0.7", lw=0.5)
    ax.set_xlim(0.0, 0.7); ax.set_ylim(-5.0, 60.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel(r"$\mathit{Fn}$")
    ax.set_title(f"({tag})", loc="left")
    ax.text(0.04, 0.93, label, transform=ax.transAxes, va="top", ha="left", fontsize=8)
axes[0].set_ylabel(r"$R$ [N]")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left",
               bbox_to_anchor=(0.02, 0.87), handlelength=2.2)
fig.tight_layout()
fig.savefig("docs/figures/resistance_by_hull.pdf", bbox_inches="tight")
fig.savefig("docs/figures/resistance_by_hull.png", dpi=220, bbox_inches="tight")

for name, label in (("s280_static.json", "Sysser 01"), ("s50_static.json", "Sysser 50")):
    fn, rr, rw, n = load(name)
    print(f"{label}: {n} panels, {len(fn)} speeds, Fn {fn[0]:.2f}-{fn[-1]:.2f}, "
          f"Rr max {rr.max():.1f} N, Rw max {rw.max():.1f} N")
