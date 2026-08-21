"""Tests for the second hull, the parameterisation detection and the trim pivot."""
import numpy as np
import pytest

from wave_resistance.hull import Attitude, Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave
from wave_resistance.iges import read_iges
from wave_resistance.measurements import hull_runs, trim_pivot_x
from wave_resistance.reference import hull_reference


def test_transposing_a_patch_leaves_the_surface_alone():
    """Exchanging u and v must be a relabelling, not a change of geometry."""
    from wave_resistance import nurbs
    patch = read_iges("data/SYSSER50_surface.igs").surfaces[1]
    other = patch.transposed()
    a = np.array([0.13, 0.47, 0.91])
    b = np.array([0.29, 0.62, 0.05])
    span = lambda rng, t: rng[0] + t * (rng[1] - rng[0])
    p = nurbs.evaluate(patch, span(patch.u_range, a), span(patch.v_range, b))
    q = nurbs.evaluate(other, span(other.u_range, b), span(other.v_range, a))
    assert np.allclose(p, q, atol=1e-12)


def test_the_girth_parameter_is_detected_on_both_files():
    """The release disagrees with itself about which parameter runs along the hull.

    Sysser 01 puts girth first, Sysser 50 length first.  After loading, both must present
    the girth parameter first, or ``waterline_fitted_mesh`` walks the wrong direction --
    which on Sysser 50 it did, raising rather than returning a wrong answer.
    """
    for name in ("SYSSER01", "SYSSER50"):
        hull = Hull.from_iges(f"data/{name}_surface.igs")
        patch = hull.patches[hull.wetted_patch_indices[0]]
        cp = patch.control_points
        along_girth = float(np.mean(np.ptp(cp[:, :, 0], axis=0)))
        along_length = float(np.mean(np.ptp(cp[:, :, 0], axis=1)))
        assert along_length > 4.0 * along_girth, name


@pytest.mark.parametrize("sysser", [1, 50])
def test_the_body_fitted_mesh_builds_on_both_hulls(sysser):
    ref = hull_reference(sysser)
    hull = Hull.from_iges(f"data/SYSSER{sysser:02d}_surface.igs")
    hull = hull.with_datum_shift(solve_reference_heave(hull, ref["volume"]))
    mesh = hull.waterline_fitted_mesh(5, 20, Attitude(), first_depth=0.010)
    assert mesh.n_faces > 200
    assert mesh.centroids()[:, 2].max() < 0.0          # every centroid below the surface
    assert mesh.areas().sum() == pytest.approx(ref["wetted_area"], rel=0.05)


@pytest.mark.parametrize("sysser", [1, 50])
def test_matching_the_published_volume_recovers_the_published_draught(sysser):
    """The draught is not a target of the flotation solve, so it is a real check."""
    ref = hull_reference(sysser)
    hull = Hull.from_iges(f"data/SYSSER{sysser:02d}_surface.igs")
    hull = hull.with_datum_shift(solve_reference_heave(hull, ref["volume"]))
    r = hydrostatics(hull.mesh(48, 240))
    assert r.volume == pytest.approx(ref["volume"], rel=1e-3)
    assert r.draught == pytest.approx(ref["tc"], rel=3e-3)


def test_sysser50_reconciles_in_length_and_beam_where_sysser01_does_not():
    """The geometry discrepancy is specific to Sysser 01, and that is the finding.

    Sysser 50's file reproduces its published waterline length and beam to better than
    0.1 per cent at the volume-matched datum; Sysser 01's is out by 0.5 and 1.1 per cent.
    Both files carry the same 2012 re-measurement header, so the provenance attribution
    cannot be a systematic property of the release.
    """
    got = {}
    for sysser in (1, 50):
        ref = hull_reference(sysser)
        hull = Hull.from_iges(f"data/SYSSER{sysser:02d}_surface.igs")
        hull = hull.with_datum_shift(solve_reference_heave(hull, ref["volume"]))
        r = hydrostatics(hull.mesh(48, 240))
        got[sysser] = (abs(r.lwl / ref["lwl"] - 1.0), abs(r.bwl / ref["bwl"] - 1.0))
    assert got[50][0] < 1e-3 and got[50][1] < 1e-3
    assert got[1][0] > 4e-3 and got[1][1] > 8e-3


def test_the_length_to_beam_ratio_separates_shape_from_flotation():
    """The ratio barely depends on heave, so it isolates shape.

    Sysser 01's stays below its published value over a heave range spanning 13 per cent of
    the displaced volume; Sysser 50's crosses zero at the volume-matched datum.
    """
    errs = {}
    for sysser in (1, 50):
        ref = hull_reference(sysser)
        published = ref["lwl"] / ref["bwl"]
        hull = Hull.from_iges(f"data/SYSSER{sysser:02d}_surface.igs")
        hull = hull.with_datum_shift(solve_reference_heave(hull, ref["volume"]))
        errs[sysser] = [
            hydrostatics(hull.mesh(48, 240, Attitude(sinkage=dz))).lwl
            / hydrostatics(hull.mesh(48, 240, Attitude(sinkage=dz))).bwl / published - 1.0
            for dz in (-4e-3, 0.0, 4e-3)]
    assert max(errs[1]) < -3e-3                      # never reaches the published ratio
    assert min(errs[50]) < 0.0 < max(errs[50])       # brackets it


def test_the_trim_pivot_is_the_published_lcb():
    """The release's Info sheet fixes the sinkage datum at the centre of gravity, and a
    model floating at zero trim carries it over the LCB."""
    for sysser in (1, 50):
        assert trim_pivot_x(sysser, 0.0) == pytest.approx(hull_reference(sysser)["lcb"])
        assert trim_pivot_x(sysser, 1.25) == pytest.approx(1.25 + hull_reference(sysser)["lcb"])


def test_the_pivot_vertical_position_does_not_matter():
    """Only the pivot's x matters: a vertical offset is a surge plus a second-order heave."""
    ref = hull_reference(50)
    hull = Hull.from_iges("data/SYSSER50_surface.igs")
    hull = hull.with_datum_shift(solve_reference_heave(hull, ref["volume"]))
    run = max(hull_runs(50), key=lambda r: abs(r.trim))
    volumes = []
    for z in (0.0, -0.1):
        att = Attitude(sinkage=run.sinkage, trim=run.trim, pivot=(0.83, 0.0, z))
        volumes.append(hydrostatics(hull.mesh(32, 160, att)).volume)
    assert volumes[0] == pytest.approx(volumes[1], rel=2e-3)


def test_sysser50_runs_are_read_and_reduced():
    runs = hull_runs(50)
    assert len(runs) == 13
    assert runs[0].froude == pytest.approx(0.0996, abs=1e-3)
    assert runs[-1].froude == pytest.approx(0.6998, abs=1e-3)
    # Residuary resistance must be a strictly smaller, positive part of the total here.
    for r in runs:
        assert 0.0 < r.residuary_resistance < r.total_resistance
    # The measured sinkage turns round before the top of the range: deepest near Fn = 0.50.
    deepest = max(runs, key=lambda r: r.sinkage)
    assert 0.45 < deepest.froude < 0.55


def test_the_excluded_patch_is_immersed_at_the_running_attitude():
    """A limitation, asserted so that fixing it breaks this test.

    Both files carry a patch above the waterline that the wetted-surface model excludes.
    At the static attitude it clears the free surface; at the measured attitude above
    Fn = 0.45 it does not, so the panel mesh omits genuinely wetted surface there.
    """
    ref = hull_reference(50)
    hull = Hull.from_iges("data/SYSSER50_surface.igs")
    hull = hull.with_datum_shift(solve_reference_heave(hull, ref["volume"]))
    base = hull.mesh(48, 240)
    x_mid = 0.5 * (float(base.vertices[:, 0].min()) + float(base.vertices[:, 0].max()))
    pivot = (trim_pivot_x(50, x_mid), 0.0, 0.0)

    dry = hull.other_patch_mesh(12, 60, Attitude())
    assert dry is not None and dry.vertices[:, 2].min() > 0.0

    fast = next(r for r in hull_runs(50) if r.froude > 0.55)
    wet = hull.other_patch_mesh(12, 60, Attitude(sinkage=fast.sinkage, trim=fast.trim,
                                                 pivot=pivot))
    assert wet.vertices[:, 2].min() < -0.02
