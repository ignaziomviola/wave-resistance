"""V10 and V12: the Neumann-Kelvin assembly, solve and diagnostics.

These tests are deliberately small.  The wave kernel costs about 0.9 ms per panel pair, so
a solve is quadratic in panel count and a full-size case belongs in a script, not a test
suite.
"""

import numpy as np
import pytest

from _shapes import box_mesh, icosphere
from wave_resistance.hull import Hull, TriMesh
from wave_resistance.nk import (
    _mirrored_in_z, check_envelope, influence_matrix, pressure_resistance,
    rankine_with_image_matrix, solve_nk, wave_influence_matrix, x_velocity_matrix,
)
from wave_resistance.panels import rankine_normal_velocity_matrix
from wave_resistance.wigley import wigley_mesh

GRAV = 9.80665


def test_mirroring_in_z_reflects_and_reverses_winding():
    tri = np.array([[[0.0, 0.0, -1.0], [1.0, 0.0, -2.0], [0.0, 1.0, -3.0]]])
    image = _mirrored_in_z(tri)
    assert image[0, :, 2].tolist() == [3.0, 2.0, 1.0]
    # Reversed winding keeps the normal on the same side of the reflected panel.
    def normal(t):
        n = np.cross(t[1] - t[0], t[2] - t[0])
        return n / np.linalg.norm(n)
    assert normal(image[0])[2] == pytest.approx(-normal(tri[0])[2], abs=1e-14)


def test_image_term_is_not_a_small_correction():
    """The non-wave part of the Kelvin Green function is -1/(4 pi R) + 1/(4 pi R1).

    Dropping the image is not a minor omission near the free surface -- cancelling the
    potential there is exactly what it is for.  It has to be measured off the diagonal:
    the diagonal is the jump term 1/2 on both matrices and dominates the Frobenius norm, so
    comparing full norms understates the image at 5 per cent.  Off the diagonal the image
    is 18 per cent of the direct term on this mesh, and its absolute size, 0.44, is the
    same either way.
    """
    mesh = wigley_mesh(1.0, 0.05, 0.0625, n_x=14, n_z=5)
    direct = rankine_normal_velocity_matrix(mesh)
    with_image = rankine_with_image_matrix(mesh, collocated=True)
    off = ~np.eye(mesh.n_faces, dtype=bool)
    ratio = np.linalg.norm((with_image - direct)[off]) / np.linalg.norm(direct[off])
    assert ratio == pytest.approx(0.179, abs=0.02)


def test_assembled_diagonal_is_the_jump_plus_a_regular_image_term():
    mesh = wigley_mesh(1.0, 0.05, 0.0625, n_x=12, n_z=4)
    a = influence_matrix(mesh, k0=11.11, wave=False)
    # 1/2 from the jump, minus the image's own regular contribution: near but not equal.
    assert np.abs(np.diag(a) - 0.5).max() < 0.05
    assert np.abs(np.diag(a) - 0.5).max() > 0.0


def test_wave_part_vanishes_as_the_froude_number_grows():
    """k0 = g/u^2 tends to zero with increasing speed, and with it the free-surface memory,
    so the influence matrix must tend to its Rankine-with-image limit.

    Measured ratios of the wave-part norm to the Rankine-with-image norm on this mesh:
    0.3005 at Fn = 0.35, 0.1719 at 1, 0.0605 at 3, 0.0094 at 10.  The decay is roughly
    linear in k0 rather than fast, which is why the threshold is set at Fn = 10 and not at
    Fn = 3 as first written.
    """
    mesh = wigley_mesh(1.0, 0.05, 0.0625, n_x=10, n_z=4)
    base = np.linalg.norm(rankine_with_image_matrix(mesh, collocated=True))
    norms = []
    for fn in (0.35, 1.0, 3.0, 10.0):
        k0 = GRAV / (fn * np.sqrt(GRAV * 1.0)) ** 2
        norms.append(np.linalg.norm(wave_influence_matrix(mesh, k0, order=1)) / base)
    assert norms[0] > norms[1] > norms[2] > norms[3]
    assert norms[-1] < 0.05 * norms[0]


def test_a_panel_in_the_mirror_plane_is_caught_rather_than_solved_around():
    """A panel lying in y = 0 is duplicated by mirroring, giving coincident panels with
    opposite normals and an exactly singular matrix.  numpy will happily return an answer
    for that, so the solver refuses instead.

    The sphere is submerged well clear of the free surface so that the envelope check, which
    runs first, has nothing to complain about and the conditioning guard is what fires.
    """
    mesh = icosphere(0.3, subdivisions=1)
    deep = TriMesh(mesh.vertices - np.array([0.0, 0.0, 2.0]), mesh.faces)
    flat = TriMesh(np.vstack([deep.vertices, np.array(
        [[0.0, 0.0, -2.4], [0.1, 0.0, -2.4], [0.05, 0.0, -2.3]])]),
        np.vstack([deep.faces, [[len(deep.vertices), len(deep.vertices) + 1,
                                 len(deep.vertices) + 2]]]))
    doubled = flat.joined(flat.mirrored_y())
    with pytest.raises(ValueError, match="numerically singular"):
        solve_nk(doubled, speed=1.0, length=1.0, check_points=4)


def test_solve_on_a_real_hull_runs_and_reports_diagnostics(sysser01_hull):
    hull = sysser01_hull.with_datum_shift(0.127078)
    mesh = hull.mesh(4, 20).clipped_below(0.0)
    lwl = 1.608142
    speed = 0.35 * np.sqrt(GRAV * lwl)
    result = solve_nk(mesh, speed, lwl, order=1, spectrum_order=3, check_points=12)
    assert result.resistance > 0.0
    assert result.condition_number < 1e3
    assert result.spectrum_converged
    assert result.body_residual_rms < 0.2          # fraction of the onset speed
    assert result.n_panels == mesh.n_faces
    assert "centroid rule" in result.summary()


def test_waterline_fitted_mesh_reproduces_the_hydrostatics(sysser01_hull):
    """It is an alternative discretisation, so it must describe the same body."""
    from wave_resistance.hydrostatics import hydrostatics
    hull = sysser01_hull.with_datum_shift(0.127078)
    fitted = hull.waterline_fitted_mesh(10, 50)
    reference = hydrostatics(hull.mesh(48, 240))
    assert fitted.areas().sum() == pytest.approx(reference.wetted_area, rel=0.01)
    assert fitted.signed_volume() == pytest.approx(reference.volume, rel=0.01)


def test_waterline_fitted_mesh_rejects_bad_arguments(sysser01_hull):
    hull = sysser01_hull.with_datum_shift(0.127078)
    with pytest.raises(ValueError):
        hull.waterline_fitted_mesh(0, 10)
    with pytest.raises(ValueError):
        hull.waterline_fitted_mesh(4, 20, first_depth=-0.1)

def test_the_envelope_check_refuses_a_mesh_the_kernel_cannot_serve():
    """A mesh whose panels straddle the hull at the free surface leaves the quadrature's
    validity envelope, where the grid coarsens with nothing to signal it.  The assembly
    raises rather than returning the plausible number it would otherwise produce."""
    mesh = wigley_mesh(1.0, 0.3, 0.0625, n_x=8, n_z=3)
    lifted = TriMesh(mesh.vertices + np.array([0.0, 0.0, 0.0624]), mesh.faces)
    worst, over = check_envelope(lifted, k0=400.0)
    assert over > 0, f"expected pairs outside the envelope, worst count was {worst:.0f}"
    with pytest.raises(ValueError, match="validity envelope"):
        influence_matrix(lifted, k0=400.0, wave=True)


def test_a_real_mesh_stays_inside_the_envelope(sysser01_hull):
    hull = sysser01_hull.with_datum_shift(0.127078)
    mesh = hull.waterline_fitted_mesh(5, 20)
    k0 = GRAV / (0.30 * np.sqrt(GRAV * 1.6)) ** 2
    worst, over = check_envelope(mesh, k0)
    assert over == 0
    assert worst < 2000.0


def test_the_x_velocity_diagonal_carries_the_fluid_side_jump():
    """The in-plane self-velocity of a panel is finite and the formula gives it; the normal
    component is the ambiguous one, and the fluid-side limit +1/2 is imposed.

    A box has end panels with n_x = +/-1 exactly, so for those the whole jump shows up in
    the x-velocity and the diagonal must be n_x/2.  Sunk deep enough that the image
    contributes little and the wave part is negligible, the diagonal isolates the jump.
    """
    box = box_mesh(1.0, 0.4, 0.3, 0.0, n=4)
    deep = TriMesh(box.vertices - np.array([0.0, 0.0, 6.0]), box.faces)
    vx = x_velocity_matrix(deep, k0=1e-3, order=1)
    normal_x = deep.unit_normals()[:, 0]
    idx = np.arange(deep.n_faces)
    ends = np.abs(normal_x) > 0.99
    assert ends.sum() >= 8
    assert vx[idx, idx][ends] / normal_x[ends] == pytest.approx(0.5, abs=0.03)
    # Panels whose normal has no x-component see none of the *jump* in x, but their
    # in-plane self-velocity is genuinely non-zero: a triangle's centroid is not its
    # symmetric point.  It comes out at 0.038 here, alternating in sign between the two
    # triangles of each quad, which is a tenth of the jump and not an error.
    flat = np.abs(normal_x) < 1e-12
    assert flat.any()
    assert np.abs(vx[idx, idx][flat]).max() < 0.05


@pytest.mark.slow
def test_pressure_and_far_field_resistance_agree(sysser01_hull):
    """V11: two independent estimators of the same quantity.

    The far-field route uses only the source strengths; the pressure route integrates the
    linearised pressure using on-hull velocities.  They are different discretisations and
    converge at different rates, so they are compared with a tolerance.  Measured on a thin
    Wigley hull against the Michell oracle at 158 panels: far-field 1.036 of the oracle,
    pressure 0.972, the two within 6.6 per cent of each other.
    """
    hull = sysser01_hull.with_datum_shift(0.127078)
    mesh = hull.waterline_fitted_mesh(4, 14, first_depth=0.010)
    lwl = 1.600000
    speed = 0.30 * np.sqrt(GRAV * lwl)
    k0 = GRAV / speed ** 2
    result = solve_nk(mesh, speed, lwl, order=1, spectrum_order=3, check_points=16)
    pressure = pressure_resistance(mesh, result.sigma, speed, k0, order=1)
    assert pressure > 0.0
    assert result.resistance == pytest.approx(pressure, rel=0.25)
