#!/usr/bin/env python3
"""Plot and optionally export the analytical 2026 Dufour 39 surrogate.

The model uses Dufour's published principal dimensions and qualitative hull
description, but no builder offsets.  It is a bare, upright canoe body for the
Michell solver: keel, rudder, topsides and the real open transom are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np

from wave_resistance import HullGeometryWarning, dufour_39_approx_hull


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=201, help="longitudinal points")
    parser.add_argument("--nz", type=int, default=129, help="vertical points")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dufour_39_hull.png"),
        help="plot output path",
    )
    parser.add_argument("--offsets-csv", type=Path, help="optional offset export")
    parser.add_argument("--metadata-json", type=Path, help="optional metadata export")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    return parser.parse_args()


def export_offsets(hull, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("x_m", "z_m", "half_breadth_m"))
        for station, x_value in enumerate(hull.x):
            for level, z_value in enumerate(hull.z):
                writer.writerow(
                    (
                        "{:.12g}".format(x_value),
                        "{:.12g}".format(z_value),
                        "{:.12g}".format(hull.half_breadths[station, level]),
                    )
                )


def main() -> None:
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HullGeometryWarning)
        hull = dufour_39_approx_hull(nx=args.nx, nz=args.nz)

    if args.offsets_csv is not None:
        export_offsets(hull, args.offsets_csv)
    if args.metadata_json is not None:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        metadata_document = hull.metadata.as_dict()
        metadata_document["provenance"] = {
            "source": "Dufour Yachts, Dufour 39 specifications",
            "url": "https://www.dufour-yachts.com/en/sailboats/dufour-39/",
            "accessed": "2026-08-10",
        }
        metadata_document["published_particulars"] = {
            "length_overall_m": 12.0,
            "hull_length_m": 11.27,
            "waterline_length_m": 10.50,
            "maximum_hull_beam_m": 4.10,
            "total_draft_m": 1.95,
            "unloaded_displacement_kg": 8600.0,
        }
        metadata_document["analytical_assumptions"] = {
            "waterline_beam_m": 3.70,
            "bare_canoe_volume_m3": 8.00,
            "smooth_superelliptic_sections": True,
            "cosine_spaced_vertical_levels": int(args.nz),
            "real_transom_replaced_by_pointed_closure": True,
            "keel_rudder_and_topsides_excluded": True,
            "builder_offsets_used": False,
        }
        args.metadata_json.write_text(
            json.dumps(metadata_document, indent=2) + "\n", encoding="utf-8"
        )

    try:
        if not args.show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required; install with `python -m pip install -e '.[plot]'`"
        ) from exc

    figure = plt.figure(figsize=(11.5, 7.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.45, 1.0))
    perspective = figure.add_subplot(grid[0, :], projection="3d")
    plan = figure.add_subplot(grid[1, 0])
    body = figure.add_subplot(grid[1, 1])

    x_surface, z_surface = np.meshgrid(hull.x, hull.z, indexing="ij")
    for sign, colour in ((-1.0, "#255a78"), (1.0, "#397fa3")):
        perspective.plot_surface(
            x_surface,
            sign * hull.half_breadths,
            -z_surface,
            rstride=max(1, args.nx // 40),
            cstride=max(1, args.nz // 24),
            color=colour,
            edgecolor="#183846",
            linewidth=0.18,
            alpha=0.82,
            antialiased=True,
        )
    perspective.plot(hull.x, hull.half_breadths[:, 0], 0.0, color="#e09a33", lw=1.8)
    perspective.plot(hull.x, -hull.half_breadths[:, 0], 0.0, color="#e09a33", lw=1.8)
    perspective.set_box_aspect((4.5, 2.0, 1.0))
    perspective.view_init(elev=22.0, azim=-61.0)
    perspective.set_xlabel("aft  ←  x (m)  →  forward")
    perspective.set_ylabel("y (m)", labelpad=5)
    perspective.set_zlabel("height (m)", labelpad=5)
    perspective.set_title("Equivalent bare canoe body (vertical scale exaggerated)")

    waterline = hull.half_breadths[:, 0]
    plan.fill_between(hull.x, -waterline, waterline, color="#397fa3", alpha=0.72)
    plan.plot(hull.x, waterline, color="#183846", lw=1.0)
    plan.plot(hull.x, -waterline, color="#183846", lw=1.0)
    plan.set_aspect("equal", adjustable="box")
    plan.set_xlabel("x (m)")
    plan.set_ylabel("y (m)")
    plan.set_title("Static-waterline planform")
    plan.grid(alpha=0.2)

    fractions = np.array((0.10, 0.25, 0.43, 0.60, 0.75, 0.90))
    station_indices = [int(np.argmin(np.abs(hull.x / 10.50 - value))) for value in fractions]
    for fraction, station in zip(fractions, station_indices):
        profile = hull.half_breadths[station]
        body.plot(profile, -hull.z, lw=1.25, label="{:.0f}% LWL".format(100 * fraction))
        body.plot(-profile, -hull.z, lw=1.25)
    body.axhline(0.0, color="#e09a33", lw=1.2)
    body.set_aspect("equal", adjustable="box")
    body.set_xlabel("y (m)")
    body.set_ylabel("height from waterline (m)")
    body.set_title("Body plan (selected stations)")
    body.grid(alpha=0.2)
    body.legend(
        fontsize=7,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
    )

    figure.suptitle("2026 Dufour 39-inspired analytical hull — not builder lines")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190)
    print("Wrote {}".format(args.output))
    print(
        "LWL=10.50 m, assumed BWL=3.70 m, equivalent canoe draft={:.3f} m, "
        "volume={:.3f} m3, wetted area={:.2f} m2".format(
            hull.draft_m, hull.displaced_volume, hull.wetted_area_m2
        )
    )
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
