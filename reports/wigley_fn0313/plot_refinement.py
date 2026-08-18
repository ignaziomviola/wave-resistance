"""Generate the refinement and literature-comparison figures."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
with (ROOT/"refinement_results.csv").open(newline="",encoding="utf-8") as stream:
    rows=list(csv.DictReader(stream))
literature=json.loads((ROOT/"literature_reference.json").read_text(encoding="utf-8"))
panels=np.array([int(r["total_panels"]) for r in rows])
pressure=np.array([float(r["pressure_coefficient_raw"]) for r in rows])
condition=np.array([float(r["matrix_condition"]) for r in rows])

fig,(top,bottom)=plt.subplots(2,1,figsize=(7.2,7.2),sharex=True,
                              gridspec_kw={"height_ratios":[1.1,1]})
top.axhspan(literature["experimental_cw_min"],literature["experimental_cw_max"],
            color="0.85",label="Chen--Noblesse experimental range")
top.axhline(literature["experimental_cw_mean"],color="black",linestyle="--",
            label="experimental mean")
top.axhline(literature["theoretical_cw_mean"],color="tab:blue",linestyle="--",
            label="published theoretical mean")
top.axhline(literature["michell_solver_cw"],color="tab:green",linestyle=":",
            label="repository Michell solver")
top.plot(panels,pressure,"o-",color="tab:red",label="rejected BEM pressure diagnostic")
top.set_ylim(-1e-4,2.2e-3); top.set_ylabel(r"$C_W$")
top.set_title("Wigley hull at Fn = 0.313: literature and computed resistance")
top.grid(True,alpha=.25); top.legend(fontsize=8,ncol=2)

bottom.plot(panels,1e6*pressure,"o-",color="tab:red",
            label="rejected BEM pressure diagnostic")
bottom.axhline(0,color="black",linewidth=.8)
bottom.set_ylabel(r"raw $C_{W,p}\times10^6$")
bottom.set_xlabel("Total coupled triangular panels")
bottom.grid(True,alpha=.25); bottom.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT/"literature_comparison.png",dpi=200); plt.close(fig)

with (ROOT/"double_body_refinement.csv").open(newline="",encoding="utf-8") as stream:
    double=list(csv.DictReader(stream))
hpan=np.array([int(r["hull_panels"]) for r in double])
drag=np.abs([float(r["pressure_resistance_N"]) for r in double])
impermeability=np.array([float(r["hull_impermeability_residual"]) for r in double])
eta=np.maximum(np.abs([float(r["eta_min_m"]) for r in rows]),
               np.abs([float(r["eta_max_m"]) for r in rows]))
fig,(a,b)=plt.subplots(2,1,figsize=(7.2,7.0))
a.loglog(hpan,drag,"o-",label="absolute double-body drag [N]")
a.loglog(hpan,impermeability,"s-",label="impermeability residual")
a.set(xlabel="Hull panels",ylabel="Magnitude",title="Double-body refinement")
a.grid(True,which="both",alpha=.25); a.legend(fontsize=8)
b.semilogx(panels,eta,"o-",label=r"max $|\eta|$ [m]")
b.set_yscale("log"); b.set_ylabel(r"max $|\eta|$ [m]")
c=b.twinx(); c.semilogx(panels,condition,"s--",color="tab:red",
                        label="coupled matrix condition number")
c.set_yscale("log"); c.set_ylabel("condition number",color="tab:red")
b.set(xlabel="Total coupled panels",title="Free-surface refinement")
b.grid(True,which="both",alpha=.25)
lines=b.get_lines()+c.get_lines(); b.legend(lines,[line.get_label() for line in lines],fontsize=8)
fig.tight_layout(); fig.savefig(ROOT/"refinement_diagnostics.png",dpi=200); plt.close(fig)

nonlinear=np.load(ROOT/"nonlinear_L1.npz")
fig,ax=plt.subplots(figsize=(7.2,4.2))
for key,label in (("residual_bem","BEM"),
                  ("residual_hull_impermeability","hull impermeability"),
                  ("residual_free_surface_kinematic","free-surface kinematic"),
                  ("residual_free_surface_dynamic","free-surface dynamic"),
                  ("residual_nonlinear_update","nonlinear update")):
    ax.semilogy(np.arange(len(nonlinear[key])),nonlinear[key],"o-",label=label)
ax.axhline(.05,color="black",linestyle="--",linewidth=.8,label="residual tolerance")
ax.set(xlabel="Nonlinear iteration",ylabel="Residual / update norm",
       title="Corrected 96-panel nonlinear continuation at Fn = 0.313")
ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(ROOT/"nonlinear_convergence.png",dpi=200); plt.close(fig)
