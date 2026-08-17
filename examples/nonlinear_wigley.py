"""Run the restricted exact-body sequence on a small Wigley verification mesh."""

from pathlib import Path

from wave_resistance import (
    BEMSettings,
    DoubleBodyPotentialFlowSolver,
    GeometrySettings,
    LinearPotentialFlowSolver,
    NonlinearPotentialFlowSolver,
    wigley_hull,
)


output = Path("nonlinear_wigley_output")
output.mkdir(exist_ok=True)
hull = wigley_hull(length_m=1.0, beam_m=0.10, draft_m=0.0625, nx=4, nz=3)
bem = BEMSettings(quadrature_order=4)
geometry = GeometrySettings(free_surface_nx=5, free_surface_ny=5)

double_body = DoubleBodyPotentialFlowSolver(hull, bem=bem).solve()
linear = LinearPotentialFlowSolver(hull, geometry=geometry, bem=bem).solve()
nonlinear = NonlinearPotentialFlowSolver(hull, geometry=geometry, bem=bem).solve()

double_body.to_json(output / "double_body_summary.json")
linear.to_json(output / "linear_summary.json")
nonlinear.to_json(output / "nonlinear_summary.json")
nonlinear.to_npz(output / "nonlinear_fields.npz")

try:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    vertices = nonlinear.free_surface_vertices
    triangulation = mtri.Triangulation(vertices[:, 0], vertices[:, 1], nonlinear.free_surface_faces)
    figure, axis = plt.subplots()
    contours = axis.tricontourf(triangulation, nonlinear.elevation, levels=30)
    figure.colorbar(contours, ax=axis, label=r"$\eta$ (m)")
    axis.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    figure.tight_layout()
    figure.savefig(output / "wave_elevation.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots()
    history = nonlinear.nonlinear_residual_history
    for name in ("bem", "kinematic", "dynamic", "exact_dynamic", "update"):
        axis.semilogy([max(float(row.get(name, float("nan"))), 1.0e-16) for row in history], label=name)
    axis.set(xlabel="nonlinear iteration", ylabel="residual", title="Nonlinear convergence")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "convergence.png", dpi=180)
    plt.close(figure)
except ImportError:
    pass

print("double body:", double_body.status)
print("linear:", linear.status)
print("nonlinear:", nonlinear.status)
print("failure reasons:", nonlinear.failure_reasons)
