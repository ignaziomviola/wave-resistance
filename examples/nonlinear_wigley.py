#!/usr/bin/env python3
"""Restricted Rankine-BEM Wigley workflow with strict acceptance reporting."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from wave_resistance import (DoubleBodyPotentialFlowSolver, GeometrySettings,
    LinearPotentialFlowSolver, NonlinearPotentialFlowSolver, NonlinearSettings,
    wigley_hull)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fn",type=float,default=.30)
    parser.add_argument("--output",type=Path,default=Path("nonlinear_wigley_output"))
    parser.add_argument("--quick",action="store_true",help="use the smallest CI mesh")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    hull=wigley_hull(length_m=1.,nx=41,nz=17)
    geometry=(GeometrySettings(hull_nx=3,hull_nz=2,free_surface_nx=5,free_surface_ny_half=2)
              if args.quick else GeometrySettings(hull_nx=5,hull_nz=3,free_surface_nx=7,free_surface_ny_half=3))
    nonlinear=NonlinearSettings(continuation_steps=3,max_iterations=8,relaxation=.1,
                                residual_tolerance=.05)
    cases=(DoubleBodyPotentialFlowSolver(geometry=geometry).solve(hull,args.fn),
           LinearPotentialFlowSolver(geometry=geometry).solve(hull,args.fn),
           NonlinearPotentialFlowSolver(geometry=geometry,nonlinear=nonlinear).solve(hull,args.fn))
    for result in cases:
        stem=result.method_name.replace("-","_")
        result.to_json(args.output/(stem+".json")); result.to_npz(args.output/(stem+".npz"))
        print(f"{result.method_name}: accepted={result.accepted}; reasons={result.failure_reasons}")
    nonlinear_result=cases[-1]
    with (args.output/"convergence.csv").open("w",newline="",encoding="utf-8") as stream:
        keys=list(nonlinear_result.residual_history); writer=csv.writer(stream); writer.writerow(["iteration",*keys])
        count=max((len(v) for v in nonlinear_result.residual_history.values()),default=0)
        for i in range(count): writer.writerow([i,*[(v[i] if i<len(v) else "") for v in nonlinear_result.residual_history.values()]])
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    linear=cases[1]; order=np.argsort(linear.wave_cuts.get("centreline_x_m",[]))
    fig,ax=plt.subplots(); ax.plot(linear.wave_cuts["centreline_x_m"][order],linear.wave_cuts["centreline_eta_m"][order],"o-")
    ax.set(xlabel="x [m]",ylabel="eta [m]",title="Linear centreline wave cut"); fig.tight_layout(); fig.savefig(args.output/"wave_elevation.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots();
    for key in ("bem","free_surface_kinematic","free_surface_dynamic","nonlinear_update"):
        values=nonlinear_result.residual_history[key]
        if len(values): ax.semilogy(np.maximum(values,1e-16),label=key)
    ax.set(xlabel="iteration",ylabel="residual"); ax.legend(); fig.tight_layout(); fig.savefig(args.output/"convergence.png",dpi=180)

if __name__=="__main__": main()
