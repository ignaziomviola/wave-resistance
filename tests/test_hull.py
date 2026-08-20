"""V8: mesh topology, orientation, exact clipping and plane sections."""

import numpy as np
import pytest

from _shapes import box_mesh, uv_sphere
from wave_resistance.hull import Attitude, TriMesh


def test_signed_volume_of_a_closed_sphere():
    r = 0.7
    mesh = uv_sphere(r, n_theta=160, n_phi=320)
    exact = 4.0 / 3.0 * np.pi * r ** 3
    assert mesh.signed_volume() == pytest.approx(exact, rel=2e-4)
    assert mesh.areas().sum() == pytest.approx(4.0 * np.pi * r ** 2, rel=2e-4)


def test_closed_body_has_no_boundary_edges():
    assert len(uv_sphere(1.0, n_theta=24, n_phi=48).boundary_vertices()) == 0


def test_clipping_is_exact_for_a_box():
    mesh = box_mesh(2.0, 0.5, 0.3, 0.2, n=4)
    below = mesh.clipped_below(0.0)
    # Bottom + four sides down to z = 0, with the lid absent.
    expected = 2.0 * 0.5 + 2 * 2.0 * 0.3 + 2 * 0.5 * 0.3
    assert below.areas().sum() == pytest.approx(expected, rel=1e-14)
    assert below.vertices[:, 2].max() == pytest.approx(0.0, abs=1e-15)


def test_clipping_is_idempotent():
    mesh = box_mesh(1.0, 0.4, 0.25, 0.15, n=5)
    once = mesh.clipped_below(0.0)
    twice = once.clipped_below(0.0)
    assert twice.areas().sum() == pytest.approx(once.areas().sum(), rel=1e-13)


def test_clipped_boundary_lies_in_the_cut_plane():
    mesh = box_mesh(1.0, 0.4, 0.25, 0.15, n=5)
    bv = mesh.clipped_below(0.0).boundary_vertices()
    assert len(bv) > 0
    assert np.abs(bv[:, 2]).max() < 1e-14


def test_mirroring_preserves_outward_orientation():
    half = uv_sphere(1.0, centre=(0.0, 2.0, 0.0), n_theta=32, n_phi=64)
    mirrored = half.mirrored_y()
    # Two disjoint spheres, each outward-oriented, so the volumes add.
    assert half.joined(mirrored).signed_volume() == pytest.approx(
        2.0 * half.signed_volume(), rel=1e-12)


def test_mirroring_negates_y_and_preserves_area():
    mesh = uv_sphere(1.0, centre=(0.0, 1.5, 0.0), n_theta=16, n_phi=32)
    m = mesh.mirrored_y()
    assert m.vertices[:, 1] == pytest.approx(-mesh.vertices[:, 1])
    assert m.areas().sum() == pytest.approx(mesh.areas().sum(), rel=1e-14)


def test_section_of_a_sphere_matches_the_analytic_circle():
    r, z0 = 1.3, 0.4
    mesh = uv_sphere(r, n_theta=200, n_phi=400)
    sec = mesh.section(np.array([0.0, 0.0, 1.0]), z0)
    assert sec.area == pytest.approx(np.pi * (r ** 2 - z0 ** 2), rel=1e-4)
    assert sec.centroid == pytest.approx([0.0, 0.0, z0], abs=1e-9)
    assert sec.perimeter == pytest.approx(2 * np.pi * np.sqrt(r ** 2 - z0 ** 2), rel=1e-4)


def test_section_area_is_independent_of_plane_orientation():
    mesh = uv_sphere(1.0, n_theta=160, n_phi=320)
    areas = [mesh.section(np.array(n, dtype=float), 0.0).area
             for n in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-2, 0.3, 1])]
    assert max(areas) - min(areas) < 1e-4 * np.pi
    for a in areas:
        assert a == pytest.approx(np.pi, rel=2e-4)


def test_a_plane_through_mesh_vertices_loses_no_boundary():
    """The x = 0 and y = 0 planes of this sphere contain whole meridians of vertices; a
    strict-inequality classifier drops segments there and the area comes out ~3e-3 low."""
    mesh = uv_sphere(1.0, n_theta=160, n_phi=320)
    on_vertices = mesh.section(np.array([1.0, 0.0, 0.0]), 0.0).area
    nudged = mesh.section(np.array([1.0, 0.0, 0.0]), 1e-9).area
    assert on_vertices == pytest.approx(nudged, rel=1e-9)


def test_clipped_mesh_has_no_unreferenced_vertices():
    below = box_mesh(1.0, 0.4, 0.25, 0.3, n=4).clipped_below(0.0)
    assert below.vertices[:, 2].max() == pytest.approx(0.0, abs=1e-15)
    assert np.unique(below.faces).size == below.vertices.shape[0]


def test_open_section_along_z0_still_gives_the_immersed_area():
    """The (1/2)(y dz - z dy) form is blind to an edge lying in z = 0, which is what
    lets an immersed section area come from the hull cut with no waterplane lid."""
    length, width, depth = 2.0, 0.6, 0.35
    below = box_mesh(length, width, depth, 0.4, n=6).clipped_below(0.0)
    sec = below.section(np.array([1.0, 0.0, 0.0]), 0.13)
    assert sec.area == pytest.approx(width * depth, rel=1e-13)


def test_identity_attitude_is_a_no_op():
    mesh = uv_sphere(1.0, n_theta=16, n_phi=32)
    rot, trans = Attitude().matrix()
    moved = mesh.transformed(rot, trans)
    assert moved.vertices == pytest.approx(mesh.vertices, abs=1e-15)


def test_sinkage_lowers_the_hull():
    mesh = box_mesh(1.0, 0.4, 0.25, 0.4, n=3)
    rot, trans = Attitude(sinkage=0.05).matrix()
    moved = mesh.transformed(rot, trans)
    assert moved.vertices[:, 2] == pytest.approx(mesh.vertices[:, 2] - 0.05)


def test_trim_is_bow_down_and_heel_is_starboard_down():
    pt = TriMesh(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0, 1, 1]]))
    rot, trans = Attitude(trim=10.0).matrix()
    assert (pt.vertices @ rot.T + trans)[0, 2] < 0.0        # bow (+x) drops
    rot, trans = Attitude(heel=10.0).matrix()
    assert (pt.vertices @ rot.T + trans)[1, 2] < 0.0        # starboard (+y) drops


def test_rotation_preserves_volume_of_a_closed_body():
    mesh = uv_sphere(1.0, n_theta=40, n_phi=80)
    rot, trans = Attitude(trim=7.0, heel=23.0, pivot=(0.3, 0.0, -0.1)).matrix()
    moved = mesh.transformed(rot, trans)
    assert moved.signed_volume() == pytest.approx(mesh.signed_volume(), rel=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-14)


def test_pivot_point_is_fixed_by_the_rotation():
    piv = np.array([0.4, 0.0, -0.2])
    rot, trans = Attitude(trim=12.0, heel=8.0, pivot=tuple(piv)).matrix()
    assert piv @ rot.T + trans == pytest.approx(piv, abs=1e-14)
