"""V1: NURBS evaluation against B-spline identities and an independent derivative surface."""

import numpy as np
import pytest

from wave_resistance import nurbs
from wave_resistance.iges import NurbsSurface


def _derivative_surface(s: NurbsSurface, axis: int) -> NurbsSurface:
    """Exact derivative patch, built independently of ``basis_and_derivative``.

    For a B-spline of degree p with knots t, the derivative is degree p-1 with control
    points Q_i = p (P_{i+1} - P_i) / (t_{i+p+1} - t_{i+1}) and knots t[1:-1].  Applying
    that along one tensor direction gives the partial derivative surface exactly, using
    only ``basis_functions`` to evaluate.
    """
    assert np.allclose(s.weights, 1.0), "construction is only valid for polynomial patches"
    p = s.degree_u if axis == 0 else s.degree_v
    t = s.knots_u if axis == 0 else s.knots_v
    dp = np.diff(s.control_points, axis=axis)
    denom = t[p + 1:-1] - t[1:-p - 1]
    q = p * dp / (denom[:, None, None] if axis == 0 else denom[None, :, None])
    return NurbsSurface(
        degree_u=p - 1 if axis == 0 else s.degree_u,
        degree_v=s.degree_v if axis == 0 else p - 1,
        knots_u=t[1:-1] if axis == 0 else s.knots_u,
        knots_v=s.knots_v if axis == 0 else t[1:-1],
        control_points=q, weights=np.ones(q.shape[:2]),
        u_range=s.u_range, v_range=s.v_range, de_pointer=-1, form=0,
    )


def test_partition_of_unity(sysser01_iges):
    s = sysser01_iges.surfaces[0]
    rng = np.random.default_rng(20260820)
    for knots, degree, n_cp, lo, hi in (
        (s.knots_u, s.degree_u, s.n_u, *s.u_range),
        (s.knots_v, s.degree_v, s.n_v, *s.v_range),
    ):
        t = rng.uniform(lo, hi, 5000)
        _, N = nurbs.basis_functions(knots, degree, n_cp, t)
        assert np.abs(N.sum(axis=-1) - 1.0).max() < 1e-13
        assert (N >= -1e-15).all()


def test_endpoint_interpolation(sysser01_iges):
    s = sysser01_iges.surfaces[0]
    u0, u1 = s.u_range
    v0, v1 = s.v_range
    assert nurbs.evaluate(s, u0, v0) == pytest.approx(s.control_points[0, 0], abs=1e-12)
    assert nurbs.evaluate(s, u1, v1) == pytest.approx(s.control_points[-1, -1], abs=1e-12)


def test_du_matches_independent_derivative_surface(sysser01_iges):
    s = sysser01_iges.surfaces[0]
    ds = _derivative_surface(s, axis=0)
    rng = np.random.default_rng(7)
    u = rng.uniform(*s.u_range, 3000)
    v = rng.uniform(*s.v_range, 3000)
    _, du, _ = nurbs.evaluate_derivatives(s, u, v)
    exact = nurbs.evaluate(ds, u, v)
    scale = np.abs(exact).max()
    assert np.abs(du - exact).max() / scale < 1e-11


def test_dv_matches_independent_derivative_surface(sysser01_iges):
    s = sysser01_iges.surfaces[0]
    ds = _derivative_surface(s, axis=1)
    rng = np.random.default_rng(11)
    u = rng.uniform(*s.u_range, 3000)
    v = rng.uniform(*s.v_range, 3000)
    _, _, dv = nurbs.evaluate_derivatives(s, u, v)
    exact = nurbs.evaluate(ds, u, v)
    assert np.abs(dv - exact).max() / np.abs(exact).max() < 1e-11


def test_derivatives_agree_with_a_plain_finite_difference(sysser01_iges):
    """Loose independent smell test.  A cubic B-spline is only C2 across its knots, so a
    finite difference cannot be pushed past a few digits here; the exact construction
    above is the real check."""
    s = sysser01_iges.surfaces[0]
    rng = np.random.default_rng(29)
    span_u = s.u_range[1] - s.u_range[0]
    span_v = s.v_range[1] - s.v_range[0]
    u = rng.uniform(s.u_range[0] + 0.05 * span_u, s.u_range[1] - 0.05 * span_u, 200)
    v = rng.uniform(s.v_range[0] + 0.05 * span_v, s.v_range[1] - 0.05 * span_v, 200)
    _, du, dv = nurbs.evaluate_derivatives(s, u, v)
    hu, hv = 1e-5 * span_u, 1e-5 * span_v
    fd_u = (nurbs.evaluate(s, u + hu, v) - nurbs.evaluate(s, u - hu, v)) / (2 * hu)
    fd_v = (nurbs.evaluate(s, u, v + hv) - nurbs.evaluate(s, u, v - hv)) / (2 * hv)
    assert np.abs(du - fd_u).max() / np.abs(fd_u).max() < 1e-5
    assert np.abs(dv - fd_v).max() / np.abs(fd_v).max() < 1e-5


def test_surface_lies_inside_the_control_net_bounding_box(sysser01_iges):
    s = sysser01_iges.surfaces[0]
    lo, hi = s.bounding_box()
    rng = np.random.default_rng(3)
    pts = nurbs.evaluate(s, rng.uniform(*s.u_range, 4000), rng.uniform(*s.v_range, 4000))
    assert (pts >= lo - 1e-12).all() and (pts <= hi + 1e-12).all()


def test_normals_are_unit_length(sysser01_iges):
    s = sysser01_iges.surfaces[0]
    rng = np.random.default_rng(5)
    span_v = s.v_range[1] - s.v_range[0]
    n = nurbs.normals(s, rng.uniform(*s.u_range, 500),
                      rng.uniform(s.v_range[0] + 0.05 * span_v, s.v_range[1], 500))
    assert np.abs(np.linalg.norm(n, axis=-1) - 1.0).max() < 1e-12
