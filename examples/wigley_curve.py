#!/usr/bin/env python3
"""Compute and plot a Michell wave-resistance curve for a Wigley hull.

Run after installing the project with its plotting extra::

    python -m pip install -e '.[plot]'
    python examples/wigley_curve.py --output wigley_curve.png

An experimental CSV may be overlaid and scored when it represents the same
quantity and fixed hull attitude::

    python examples/wigley_curve.py \
        --validation path/to/wigley_cw.csv \
        --validation-column C_W

The CSV must have a ``Fn`` column and the coefficient column named by
``--validation-column``. No experimental data are bundled because the common
legacy Wigley datasets do not carry an explicit redistribution licence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wave_resistance import MichellSolver, wigley_hull
from wave_resistance.validation import (
    HullAttitude,
    ResistanceQuantity,
    ValidationSeries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=4.0, help="hull length in metres")
    parser.add_argument("--fn-min", type=float, default=0.15, help="minimum Froude number")
    parser.add_argument("--fn-max", type=float, default=0.45, help="maximum Froude number")
    parser.add_argument("--points", type=int, default=61, help="number of speed points")
    parser.add_argument("--nx", type=int, default=161, help="longitudinal offset points")
    parser.add_argument("--nz", type=int, default=65, help="vertical offset points")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wigley_curve.png"),
        help="plot output path",
    )
    parser.add_argument(
        "--result-csv", type=Path, help="optional path for solver outputs and diagnostics"
    )
    parser.add_argument(
        "--validation", type=Path, help="optional like-for-like experimental CSV"
    )
    parser.add_argument(
        "--validation-column",
        default="C_W",
        help="experimental coefficient column (default: C_W)",
    )
    parser.add_argument(
        "--validation-quantity",
        default="wave_making",
        choices=[item.value for item in ResistanceQuantity],
        help="physical meaning of the experimental coefficient",
    )
    parser.add_argument("--show", action="store_true", help="also open an interactive window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 2:
        raise SystemExit("--points must be at least 2")
    if not 0.0 < args.fn_min < args.fn_max:
        raise SystemExit("require 0 < --fn-min < --fn-max")

    hull = wigley_hull(length_m=args.length, nx=args.nx, nz=args.nz)
    fn = np.linspace(args.fn_min, args.fn_max, args.points)
    result = MichellSolver().solve(hull, fn, include_spectrum=False)

    if args.result_csv is not None:
        args.result_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.result_csv)

    prediction = ValidationSeries(
        result.fn,
        result.c_wave_resistance,
        quantity=ResistanceQuantity.WAVE_MAKING,
        attitude=HullAttitude.FIXED,
        label="Michell thin-ship model",
    )

    observation = None
    report = None
    if args.validation is not None:
        observation = ValidationSeries.from_csv(
            args.validation,
            quantity=args.validation_quantity,
            attitude=HullAttitude.FIXED,
            coefficient_column=args.validation_column,
        )
        report = observation.compare(prediction)
        print(report.to_text())

    try:
        if not args.show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise SystemExit(
            "matplotlib is required for this example; install with "
            "`python -m pip install -e '.[plot]'`"
        ) from exc

    figure, axis = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    axis.plot(
        result.fn,
        1.0e3 * result.c_wave_resistance,
        linewidth=2.0,
        label="Michell prediction",
    )
    if observation is not None:
        axis.scatter(
            observation.fn,
            1.0e3 * observation.coefficients,
            marker="o",
            facecolors="none",
            edgecolors="black",
            label=observation.label or "experiment",
            zorder=3,
        )
    axis.set_xlabel(r"Froude number, $Fn=U/\sqrt{gL}$")
    axis.set_ylabel(r"Wave-resistance coefficient, $10^3 C_W$")
    axis.grid(True, alpha=0.25)
    axis.legend()
    if report is not None:
        axis.set_title(f"Normalized L2 error = {report.metrics.normalized_l2:.3g}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
