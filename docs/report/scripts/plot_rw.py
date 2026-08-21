"""Wave resistance versus Froude number for Sysser 01."""
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
    r = [x for x in d["records"] if "error" not in x and x.get("panels")]
    return d, sorted(r, key=lambda x: x["fn"])


ds, st = load("s280_static.json")
try:
    dm, me = load("s280_meas.json")
except Exception:
    dm, me = None, []

fn = np.array([r["fn"] for r in st])
far = np.array([r["r_far"] for r in st])
pre = np.array([r["r_press"] for r in st])
rr = np.array([r["rr_meas"] for r in st])
rt = np.array([r["rt_meas"] for r in st])

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))

ax = axes[0]
ax.plot(fn, rt, "k-", lw=0.9, marker="s", ms=3.2, mfc="w", label=r"$R_t$, measured")
ax.plot(fn, rr, "k-", lw=1.2, marker="o", ms=3.6, label=r"$R_r$, measured")
ax.plot(fn, far, "k--", lw=0.9, marker="^", ms=3.6, mfc="w",
        label=r"$R_W$, far field")
ax.plot(fn, pre, "k-.", lw=0.9, marker="v", ms=3.6, mfc="0.6",
        label=r"$R_W$, pressure")
if me:
    ax.plot([r["fn"] for r in me], [r["r_press"] for r in me], linestyle="none",
            marker="d", ms=3.6, mfc="k", color="k",
            label=r"$R_W$, pressure, measured attitude")
ax.axhline(0.0, color="0.6", lw=0.5)
ax.set_xlabel(r"$\mathit{Fn}$"); ax.set_ylabel(r"$R$ [N]")
ax.set_xlim(0.0, 0.6); ax.set_ylim(0.0, 60.0)
ax.set_xticks([0.0, 0.2, 0.4, 0.6]); ax.set_yticks([0, 20, 40, 60])
ax.legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=2.2)
ax.set_title("(a)", loc="left")

ax = axes[1]
m = fn > 0.12
ax.semilogy(fn[m], rr[m], "k-", lw=1.2, marker="o", ms=3.6)
ax.semilogy(fn[m], np.abs(far[m]), "k--", lw=0.9, marker="^", ms=3.6, mfc="w")
pp = np.where(pre > 0, pre, np.nan)
ax.semilogy(fn[m], pp[m], "k-.", lw=0.9, marker="v", ms=3.6, mfc="0.6")
neg = m & (pre <= 0)
if neg.any():
    ax.semilogy(fn[neg], np.abs(pre[neg]), linestyle="none", marker="x", ms=4.5,
                color="k", label=r"$R_W < 0$, pressure")
if me:
    mp = [(r["fn"], r["r_press"]) for r in me if r["r_press"] > 0]
    ax.semilogy([a for a, _ in mp], [b for _, b in mp], linestyle="none", marker="d",
                ms=3.6, mfc="k", color="k")
ax.set_xlabel(r"$\mathit{Fn}$"); ax.set_ylabel(r"$R$ [N]")
ax.set_xlim(0.1, 0.6); ax.set_ylim(0.01, 100.0)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.legend(frameon=False, fontsize=6.6, loc="lower right", handlelength=2.2)
ax.set_title("(b)", loc="left")

fig.tight_layout()
fig.savefig("docs/figures/resistance.pdf", bbox_inches="tight")
fig.savefig("docs/figures/resistance.png", dpi=220, bbox_inches="tight")

# --- diagnostics figure: ratio to measurement, and the solver's own diagnostics ---
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
ax = axes[0]
good = rr > 0.05
ax.plot(fn[good], far[good] / rr[good], "k--", lw=0.9, marker="^", ms=3.6, mfc="w",
        label="far field")
ax.plot(fn[good], pre[good] / rr[good], "k-.", lw=0.9, marker="v", ms=3.6, mfc="0.6",
        label="pressure")
ax.axhline(1.0, color="0.4", lw=0.7, ls="-")
ax.set_xlabel(r"$\mathit{Fn}$"); ax.set_ylabel(r"$R_W/R_r$")
ax.set_xlim(0.1, 0.6); ax.set_ylim(-2.0, 10.0)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticks([-2, 0, 2, 4, 6, 8, 10])
ax.legend(frameon=False, fontsize=7, handlelength=2.2)
ax.set_title("(a)", loc="left")

ax = axes[1]
res = np.array([r["resid_rms"] for r in st])
cap = np.array([r["lambda_cap"] for r in st])
ax.plot(fn, res, "k-", lw=1.0, marker="o", ms=3.4, label=r"body residual, rms$/u$")
ax.plot(fn, cap / 10.0, "k--", lw=0.9, marker="s", ms=3.4, mfc="w",
        label=r"$\lambda_\mathrm{max}/10$")
ax.axhline(0.1, color="0.6", lw=0.5, ls=":")
ax.set_xlabel(r"$\mathit{Fn}$"); ax.set_ylabel("diagnostic")
ax.set_xlim(0.0, 0.6); ax.set_ylim(0.0, 0.8)
ax.set_xticks([0.0, 0.2, 0.4, 0.6]); ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.legend(frameon=False, fontsize=7, handlelength=2.2)
ax.set_title("(b)", loc="left")
fig.tight_layout()
fig.savefig("docs/figures/diagnostics.pdf", bbox_inches="tight")
fig.savefig("docs/figures/diagnostics.png", dpi=220, bbox_inches="tight")

print(json.dumps([{k: r[k] for k in ("fn", "r_far", "r_press", "rr_meas", "resid_rms",
                                     "flux", "lambda_cap", "weight_fraction")}
                  for r in st], indent=0))
