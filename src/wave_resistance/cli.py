"""Command line interface."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .hull import Attitude, Hull
from .hydrostatics import RHO_FRESH, hydrostatics, solve_reference_heave
from .iges import read_iges
from .reference import sysser01_reference

_COMPARABLE = [
    ("lwl", "lwl", "m"), ("bwl", "bwl", "m"), ("draught", "tc", "m"),
    ("volume", "volume", "m^3"), ("wetted_area", "wetted_area", "m^2"),
    ("waterplane_area", "waterplane_area", "m^2"),
    ("cp", "cp", "-"), ("cm", "cm", "-"), ("cwp", "cwp", "-"),
]


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
        print(f"  {'lcb_from_midship':<18}{r.lcb_from_midship:>14.6f}{ref['lcb']:>14.6f}"
              f"{100 * (r.lcb_from_midship / ref['lcb'] - 1):>+10.3f}")
        print("\n  Lwl, Bwl and Sc differ by more than the mesh is uncertain by. The supplied\n"
              "  IGES is a 2012 re-measurement ('Rhino modellen na inmeten 2012', patch named\n"
              "  'Gerebuild oppervlak' = reconstructed surface); the published table may derive\n"
              "  from the original 1981 lines. Unresolved until M4 checks the primary release.")
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
