"""Command line interface."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .hull import Attitude, Hull
from .hydrostatics import RHO_FRESH, hydrostatics, solve_reference_heave
from .iges import read_iges
from .nk import check_envelope, pressure_resistance, solve_nk
from .reference import sysser01_reference
from .spectrum import mesh_resolved_lambda

_COMPARABLE = [
    ("lwl", "lwl", "m"), ("bwl", "bwl", "m"), ("draught", "tc", "m"),
    ("volume", "volume", "m^3"), ("wetted_area", "wetted_area", "m^2"),
    ("waterplane_area", "waterplane_area", "m^2"),
    ("midship_area", "midship_area", "m^2"),
    ("cp", "cp", "-"), ("cm", "cm", "-"), ("cb", "cb", "-"), ("cwp", "cwp", "-"),
]


def _froude_list(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("give at least one Froude number")
    if any(v <= 0.0 for v in values):
        raise argparse.ArgumentTypeError("Froude numbers must be positive")
    return values


def _cmd_info(args: argparse.Namespace) -> int:
    print(read_iges(args.file).report())
    return 0


def _cmd_hydrostatics(args: argparse.Namespace) -> int:
    hull = Hull.from_iges(args.file)
    if args.displacement is not None and args.draught is not None:
        print("error: give either --displacement or --draught, not both", file=sys.stderr)
        return 2
    if args.displacement is not None:
        shift = solve_reference_heave(hull, args.displacement, args.nu, args.nv)
        how = f"displacement {args.displacement:g} m^3"
    elif args.draught is not None:
        shift = args.draught
        how = f"draught {args.draught:g} m above the lowest keel point"
    else:
        print("error: one of --displacement or --draught is required", file=sys.stderr)
        return 2

    hull = hull.with_datum_shift(shift)
    attitude = Attitude(sinkage=args.sinkage, trim=args.trim, heel=args.heel)
    mesh = hull.mesh(args.nu, args.nv, attitude)
    r = hydrostatics(mesh, n_sections=args.sections)

    print(f"hull: {args.file}")
    print(f"  full-hull mesh {args.nu} x {args.nv} per patch, mirrored={hull.mirrored}, "
          f"{r.n_faces} wetted faces")
    print(f"  reference condition set by {how}: datum shift {shift:.6f} m")
    print(f"  prescribed attitude: sinkage {attitude.sinkage:+.4f} m, "
          f"trim {attitude.trim:+.3f} deg, heel {attitude.heel:+.3f} deg")

    other = hull.other_patch_mesh(max(8, args.nu // 4), max(40, args.nv // 4), attitude)
    if other is not None:
        z_min = float(other.vertices[:, 2].min())
        if z_min < 0.0:
            print(f"  WARNING: a patch excluded from hydrostatics reaches z = {z_min:+.4f} m, "
                  "i.e. below the free surface; its immersion is NOT accounted for")
        else:
            print(f"  patches excluded from hydrostatics clear the free surface by "
                  f"{z_min:.4f} m")

    print(f"\n  volume            {r.volume:.9f} m^3   ({r.displacement_mass(RHO_FRESH):.3f} kg fresh)")
    print(f"  volume routes     {np.array2string(np.array(r.volume_routes), precision=9)}")
    print(f"  route spread      {r.volume_spread:.3e} m^3  ({r.volume_spread / r.volume:.2e} rel)")
    print(f"  wetted area       {r.wetted_area:.9f} m^2")
    print(f"  waterplane area   {r.waterplane_area:.9f} m^2")
    print(f"  midship area      {r.midship_area:.9f} m^2")
    print(f"  Lwl / Bwl / Tc    {r.lwl:.6f} / {r.bwl:.6f} / {r.draught:.6f} m")
    print(f"  Cp / Cm / Cb / Cwp {r.cp:.6f} / {r.cm:.6f} / {r.cb:.6f} / {r.cwp:.6f}")
    print(f"  LCB               {r.lcb:.6f} m  ({r.lcb_from_midship:+.6f} m from midship, "
          f"{100 * r.lcb_from_midship / r.lwl:+.4f} %Lwl)")
    print(f"  LCF               {r.lcf:.6f} m  ({r.lcf_from_midship:+.6f} m from midship)")
    print(f"  VCB               {r.vcb:.6f} m    TCB {r.tcb:+.3e} m   TCF {r.tcf:+.3e} m")

    if args.compare:
        ref = sysser01_reference()
        print(f"\n  comparison with published Sysser 01 ({ref['source']}):")
        print(f"  {'quantity':<18}{'computed':>14}{'published':>14}{'diff %':>10}")
        for attr, key, unit in _COMPARABLE:
            got, want = getattr(r, attr), ref[key]
            print(f"  {attr:<18}{got:>14.6f}{want:>14.6f}{100 * (got / want - 1):>+10.3f}")
        # The release states the datum: "with respect to 1/2 waterline length <in upright
        # condition!>", so both centroids are compared from the waterline midpoint.
        for attr, key in (("lcb_from_midship", "lcb"), ("lcf_from_midship", "lcf")):
            got, want = getattr(r, attr), ref[key]
            print(f"  {attr:<18}{got:>14.6f}{want:>14.6f}"
                  f"{100 * (got / want - 1):>+10.3f}")
        print("\n  Datum for LCB and LCF is the midpoint of the *upright* waterline, negative\n"
              "  aft, as stated on the release's Info sheet.")
        print("\n  Lwl, Bwl and Sc differ by more than the mesh is uncertain by, and the\n"
              "  primary release has now been checked, so the published table is not the\n"
              "  explanation. The supplied IGES is a 2012 re-measurement ('Rhino modellen na\n"
              "  inmeten 2012', patch named 'Gerebuild oppervlak' = reconstructed surface)\n"
              "  while the table describes the hull the series was built and towed as. Carry\n"
              "  the difference as a geometry-provenance bias in every comparison below.")
    return 0


def _cmd_nk(args: argparse.Namespace) -> int:
    hull = Hull.from_iges(args.file)
    if args.displacement is None and args.draught is None:
        print("give --displacement or --draught to fix the reference condition",
              file=sys.stderr)
        return 2
    if args.displacement is not None:
        shift = solve_reference_heave(hull, args.displacement)
    else:
        base = hull.with_datum_shift(0.0).mesh(args.nu, args.nv)
        shift = float(base.vertices[:, 2].min()) + args.draught
    hull = hull.with_datum_shift(shift)
    # Attitude takes trim and heel in degrees, which is also what the flags are in, so no
    # conversion.  Converting here is the mistake the hydrostatics command avoids.
    attitude = Attitude(sinkage=args.sinkage, trim=args.trim, heel=args.heel)
    mesh = hull.waterline_fitted_mesh(args.girth, args.stations, attitude,
                                      first_depth=args.first_depth)
    fine = hydrostatics(hull.mesh(args.nu, args.nv, attitude))
    length = args.length if args.length is not None else fine.lwl

    print(f"reference datum shift {shift:.6f} m; Lwl {length:.6f} m; "
          f"displaced volume {fine.volume:.6f} m^3")
    print(f"NK mesh {mesh.n_faces} panels, wetted area {mesh.areas().sum():.6f} m^2, "
          f"topmost centroid {-1000 * mesh.centroids()[:, 2].max():.2f} mm below the surface")

    gravity = 9.80665
    for froude in args.fn:
        speed = froude * np.sqrt(gravity * length)
        k0 = gravity / speed ** 2
        worst, over = check_envelope(mesh, k0)
        cap = mesh_resolved_lambda(mesh, k0)
        print(f"\nFn = {froude:.4f}")
        print(f"  worst panel-pair oscillation count {worst:.0f}, "
              f"{over} pairs outside the envelope")
        print(f"  spectrum resolved to lambda {cap:.2f} by this mesh")
        result = solve_nk(mesh, speed, length, order=args.order,
                          spectrum_order=args.spectrum_order, rtol=args.rtol,
                          check_points=args.check_points, lambda_cap=cap)
        result.pressure_resistance = pressure_resistance(mesh, result.sigma, speed, k0,
                                                         order=args.order)
        print("  " + result.summary().replace("\n", "\n  "))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wave-resistance",
        description="Neumann-Kelvin wave resistance of a bare sailing-yacht canoe body.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="report the contents of an IGES file")
    p_info.add_argument("file")
    p_info.set_defaults(func=_cmd_info)

    p_hyd = sub.add_parser("hydrostatics", help="hydrostatics at a prescribed condition")
    p_hyd.add_argument("file")
    p_hyd.add_argument("--displacement", type=float, default=None,
                       help="target displaced volume in m^3; solves for the reference heave")
    p_hyd.add_argument("--draught", type=float, default=None,
                       help="explicit draught in m above the lowest keel point")
    p_hyd.add_argument("--sinkage", type=float, default=0.0,
                       help="prescribed extra sinkage in m, positive downward")
    p_hyd.add_argument("--trim", type=float, default=0.0, help="prescribed trim in degrees, bow down positive")
    p_hyd.add_argument("--heel", type=float, default=0.0, help="prescribed heel in degrees, starboard down positive")
    p_hyd.add_argument("--nu", type=int, default=48, help="mesh divisions in the girth direction")
    p_hyd.add_argument("--nv", type=int, default=240, help="mesh divisions in the longitudinal direction")
    p_hyd.add_argument("--sections", type=int, default=240, help="station planes used for the midship area")
    p_hyd.add_argument("--compare", action="store_true",
                       help="compare against the published Sysser 01 hydrostatics")
    p_hyd.set_defaults(func=_cmd_hydrostatics)

    p_nk = sub.add_parser("nk", help="Neumann-Kelvin wave resistance at prescribed attitude")
    p_nk.add_argument("file")
    p_nk.add_argument("--displacement", type=float, default=None,
                      help="target displaced volume in m^3; solves for the reference heave")
    p_nk.add_argument("--draught", type=float, default=None,
                      help="explicit draught in m above the lowest keel point")
    p_nk.add_argument("--fn", type=_froude_list, required=True,
                      help="comma-separated Froude numbers, e.g. 0.25,0.30,0.35")
    p_nk.add_argument("--length", type=float, default=None,
                      help="reference length for Fn; defaults to the computed Lwl")
    p_nk.add_argument("--girth", type=int, default=6,
                      help="NK mesh rows from the waterline to the keel")
    p_nk.add_argument("--stations", type=int, default=28,
                      help="NK mesh stations along the length")
    p_nk.add_argument("--first-depth", dest="first_depth", type=float, default=0.010,
                      help="depth in m of the lower edge of the topmost panel row")
    p_nk.add_argument("--sinkage", type=float, default=0.0,
                      help="prescribed extra sinkage in m, positive downward")
    p_nk.add_argument("--trim", type=float, default=0.0,
                      help="prescribed trim in degrees, bow down positive")
    p_nk.add_argument("--heel", type=float, default=0.0,
                      help="prescribed heel in degrees, starboard down positive")
    p_nk.add_argument("--order", type=int, default=1,
                      help="quadrature order over the source panel for the wave kernel")
    p_nk.add_argument("--spectrum-order", dest="spectrum_order", type=int, default=4,
                      help="quadrature order over the panel for the far-field amplitude")
    p_nk.add_argument("--rtol", type=float, default=3e-5,
                      help="tail tolerance for the wave-resistance integral")
    p_nk.add_argument("--check-points", dest="check_points", type=int, default=40,
                      help="off-collocation points for the body-condition residual")
    p_nk.add_argument("--nu", type=int, default=48,
                      help="girth divisions of the fine mesh used for hydrostatics")
    p_nk.add_argument("--nv", type=int, default=240,
                      help="longitudinal divisions of the fine mesh used for hydrostatics")
    p_nk.set_defaults(func=_cmd_nk)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
