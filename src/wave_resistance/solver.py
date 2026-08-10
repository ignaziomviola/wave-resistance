"""Michell thin-ship solver and reusable common-grid design operator."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from .geometry import HullMetadata, OffsetHull
from .kernels import michell_amplitude_direct, michell_amplitude_ibp
from .models import (
    BatchWaveResistanceResult,
    SolverSettings,
    SpectralDensity,
    WaterProperties,
    WaveResistanceResult,
)
from .quadrature import PhaseAwareMichellIntegrator, QuadratureResult


@dataclass
class _CachedRule:
    cutoff_lambda: float
    cycle_width: float
    fine_lambda: np.ndarray
    fine_weights: np.ndarray
    coarse_lambda: np.ndarray
    coarse_weights: np.ndarray


def _fixed_transformed_rule(
    cutoff_lambda: float, maximum_panel_width: float, order: int
) -> Tuple[np.ndarray, np.ndarray]:
    panel_count = max(1, int(math.ceil((cutoff_lambda - 1.0) / maximum_panel_width)))
    edges = np.linspace(1.0, cutoff_lambda, panel_count + 1)
    base_nodes, base_weights = leggauss(order)
    all_lambda: List[np.ndarray] = []
    all_weights: List[np.ndarray] = []
    for lambda_a, lambda_b in zip(edges[:-1], edges[1:]):
        t_a = math.sqrt(max(0.0, (lambda_a - 1.0) * (lambda_a + 1.0)))
        t_b = math.sqrt(max(0.0, (lambda_b - 1.0) * (lambda_b + 1.0)))
        half = 0.5 * (t_b - t_a)
        midpoint = 0.5 * (t_a + t_b)
        t_nodes = midpoint + half * base_nodes
        all_lambda.append(np.sqrt(1.0 + t_nodes * t_nodes))
        all_weights.append(half * base_weights)
    return np.concatenate(all_lambda), np.concatenate(all_weights)


def _estimated_wetted_area(x: np.ndarray, z: np.ndarray, y: np.ndarray) -> float:
    edge_order = 2 if x.size >= 3 and z.size >= 3 else 1
    dy_dx, dy_dz = np.gradient(y, x, z, edge_order=edge_order)
    density = np.sqrt(1.0 + dy_dx * dy_dx + dy_dz * dy_dz)
    area_z = np.sum(
        0.5 * (density[:, :-1] + density[:, 1:]) * np.diff(z)[None, :], axis=1
    )
    return float(2.0 * np.sum(0.5 * (area_z[:-1] + area_z[1:]) * np.diff(x)))


class MichellSolver:
    """Compute deep-water radiated-wave resistance from hull offsets."""

    def __init__(
        self,
        settings: Optional[SolverSettings] = None,
        water: Optional[WaterProperties] = None,
    ) -> None:
        self.settings = settings if settings is not None else SolverSettings()
        self.water = water if water is not None else WaterProperties()
        self._integrator = PhaseAwareMichellIntegrator(self.settings)

    def _make_evaluator(
        self,
        x_over_l: np.ndarray,
        z_over_l: np.ndarray,
        y_over_l: np.ndarray,
        froude: float,
    ) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, bool]]:
        settings = self.settings

        def evaluate(lambda_: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
            lambda_values = np.asarray(lambda_, dtype=float)
            ibp_amplitude, ibp_cancellation = michell_amplitude_ibp(
                lambda_values, froude, x_over_l, z_over_l, y_over_l
            )
            ibp_amplitude = np.asarray(ibp_amplitude, dtype=complex)
            ibp_cancellation = np.asarray(ibp_cancellation, dtype=float)
            if not settings.amplitude_crosscheck:
                return ibp_amplitude, ibp_cancellation, False

            # Cross-check a representative subset on every panel, plus every
            # node whose IBP cancellation could consume the requested error
            # budget.  Evaluating both formulae at every Gauss node roughly
            # doubles design-sweep cost without increasing coverage materially.
            flat_size = lambda_values.size
            representative = np.asarray(
                sorted({0, flat_size // 2, flat_size - 1}), dtype=int
            )
            severe = np.flatnonzero(
                np.finfo(float).eps * ibp_cancellation.ravel()
                > 0.1 * settings.relative_tolerance
            )
            checked = np.unique(np.concatenate((representative, severe)))
            checked_lambda = lambda_values.ravel()[checked]
            direct_amplitude, direct_cancellation = michell_amplitude_direct(
                checked_lambda, froude, x_over_l, z_over_l, y_over_l
            )
            direct_amplitude = np.asarray(direct_amplitude, dtype=complex).ravel()
            direct_cancellation = np.asarray(direct_cancellation, dtype=float).ravel()
            primary_flat = ibp_amplitude.ravel().copy()
            cancellation_flat = ibp_cancellation.ravel().copy()
            primary_checked = primary_flat[checked]
            scale = np.maximum(np.abs(primary_checked), np.abs(direct_amplitude))
            mismatch = np.abs(primary_checked - direct_amplitude) > (
                settings.amplitude_crosscheck_atol
                + settings.amplitude_crosscheck_rtol * scale
            )
            # At checked nodes, use the algebraically equivalent evaluation
            # with the smaller diagnosed cancellation.
            use_direct = direct_cancellation < cancellation_flat[checked]
            primary_flat[checked[use_direct]] = direct_amplitude[use_direct]
            cancellation_flat[checked] = np.minimum(
                direct_cancellation, cancellation_flat[checked]
            )
            return (
                primary_flat.reshape(ibp_amplitude.shape),
                cancellation_flat.reshape(ibp_cancellation.shape),
                bool(np.any(mismatch)),
            )

        return evaluate

    def _integrate_case(
        self,
        hull: OffsetHull,
        froude: float,
    ) -> Tuple[QuadratureResult, Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, bool]]]:
        x_over_l, z_over_l, y_over_l = hull.coordinates_over_length
        evaluator = self._make_evaluator(x_over_l, z_over_l, y_over_l, froude)
        x_span = float(x_over_l[-1] - x_over_l[0])
        quadrature = self._integrator.integrate(froude, x_span, evaluator)
        return quadrature, evaluator

    def _base_validity_flags(self, hull: OffsetHull, froude: float) -> List[str]:
        flags = list(hull.diagnostics.validation_warnings)
        if not (self.settings.fn_min <= froude <= self.settings.fn_max):
            flags.append(
                "outside declared Fn envelope [{:.3g}, {:.3g}]".format(
                    self.settings.fn_min, self.settings.fn_max
                )
            )
        if hull.diagnostics.maximum_longitudinal_slope > 0.30:
            flags.append(
                "max |dy/dx|={:.3g} challenges thin-ship assumptions".format(
                    hull.diagnostics.maximum_longitudinal_slope
                )
            )
        return flags

    def _physical_outputs(
        self, hull: OffsetHull, froude: float, integral: float
    ) -> Tuple[float, float, float]:
        length = hull.metadata.length_ref_m
        speed = froude * math.sqrt(self.water.g_m_s2 * length)
        resistance = (
            4.0
            * self.water.rho_kg_m3
            * self.water.g_m_s2
            * length**3
            * integral
            / (math.pi * froude**2)
        )
        area_ratio = hull.metadata.wetted_area_m2 / length**2
        coefficient = 8.0 * integral / (math.pi * froude**4 * area_ratio)
        return speed, resistance, coefficient

    def solve(
        self,
        hull: OffsetHull,
        froude_numbers: Sequence[float],
        include_spectrum: bool = False,
    ) -> WaveResistanceResult:
        """Compute a wave-resistance curve for one validated offset hull."""

        if not isinstance(hull, OffsetHull):
            raise TypeError("hull must be an OffsetHull")
        fn = np.atleast_1d(np.asarray(froude_numbers, dtype=float))
        if fn.ndim != 1 or fn.size == 0 or not np.all(np.isfinite(fn)):
            raise ValueError("froude_numbers must be a non-empty finite one-dimensional array")
        if np.any(fn <= 0.0):
            raise ValueError("Froude numbers must be positive")
        outside = (fn < self.settings.fn_min) | (fn > self.settings.fn_max)
        if np.any(outside) and not self.settings.allow_outside_envelope:
            bad = ", ".join("{:.6g}".format(value) for value in fn[outside])
            raise ValueError(
                "Froude numbers outside [{:.3g}, {:.3g}]: {}".format(
                    self.settings.fn_min, self.settings.fn_max, bad
                )
            )

        speed: List[float] = []
        resistance: List[float] = []
        coefficient: List[float] = []
        integrals: List[float] = []
        errors: List[float] = []
        tail_fractions: List[float] = []
        cancellations: List[float] = []
        converged: List[bool] = []
        flags_by_case: List[Tuple[str, ...]] = []
        spectra: Optional[Dict[float, SpectralDensity]] = {} if include_spectrum else None

        for froude in fn:
            value = float(froude)
            quadrature, evaluator = self._integrate_case(hull, value)
            case_flags = self._base_validity_flags(hull, value)
            if not quadrature.converged:
                case_flags.append("outer quadrature or tail did not converge")
            if quadrature.crosscheck_failed:
                case_flags.append("direct and integration-by-parts amplitudes disagree")
            roundoff_indicator = np.finfo(float).eps * quadrature.cancellation_ratio
            if not math.isfinite(quadrature.cancellation_ratio) or (
                roundoff_indicator > self.settings.relative_tolerance
            ):
                case_flags.append("severe complex-amplitude cancellation")
            if quadrature.integral <= self.settings.absolute_tolerance:
                case_flags.append("wave resistance is at the absolute numerical floor")
            elif quadrature.tail_fraction > self.settings.relative_tolerance:
                case_flags.append(
                    "tail accuracy is controlled by the absolute, not relative, tolerance"
                )

            case_speed, case_resistance, case_coefficient = self._physical_outputs(
                hull, value, quadrature.integral
            )
            speed.append(case_speed)
            resistance.append(case_resistance)
            coefficient.append(case_coefficient)
            integrals.append(quadrature.integral)
            errors.append(quadrature.error)
            tail_fractions.append(quadrature.tail_fraction)
            cancellations.append(quadrature.cancellation_ratio)
            converged.append(quadrature.converged)
            flags_by_case.append(tuple(case_flags))

            if spectra is not None:
                lower = 1.0 + 1.0e-8
                lambda_values = np.linspace(
                    lower, quadrature.cutoff_lambda, self.settings.spectrum_points
                )
                amplitude, _, _ = evaluator(lambda_values)
                density = (
                    lambda_values**2
                    / np.sqrt(lambda_values**2 - 1.0)
                    * np.abs(amplitude) ** 2
                )
                spectra[value] = SpectralDensity(lambda_values, density)

        return WaveResistanceResult(
            fn=fn.copy(),
            speed_m_s=np.asarray(speed),
            wave_resistance_N=np.asarray(resistance),
            c_wave_resistance=np.asarray(coefficient),
            integral=np.asarray(integrals),
            quadrature_error=np.asarray(errors),
            tail_fraction=np.asarray(tail_fractions),
            cancellation_ratio=np.asarray(cancellations),
            converged=np.asarray(converged, dtype=bool),
            validity_flags=tuple(flags_by_case),
            spectral_density=spectra,
        )


class MichellOperator:
    """Reusable quadrature operator for hulls sharing one offset grid.

    The template hull establishes certified outer cutoffs.  Fine and coarse
    transformed rules are cached independently.  A later hull that fails the
    cached error or tail checks is solved adaptively and extends the relevant
    cached rule for subsequent evaluations.
    """

    def __init__(
        self,
        template_hull: OffsetHull,
        froude_numbers: Sequence[float],
        solver: Optional[MichellSolver] = None,
    ) -> None:
        if not isinstance(template_hull, OffsetHull):
            raise TypeError("template_hull must be an OffsetHull")
        self.template_hull = template_hull
        self.solver = solver if solver is not None else MichellSolver()
        self.froude_numbers = np.atleast_1d(np.asarray(froude_numbers, dtype=float))
        if (
            self.froude_numbers.ndim != 1
            or self.froude_numbers.size == 0
            or not np.all(np.isfinite(self.froude_numbers))
            or np.any(self.froude_numbers <= 0.0)
        ):
            raise ValueError(
                "froude_numbers must be a non-empty, finite, positive one-dimensional array"
            )
        outside = (self.froude_numbers < self.solver.settings.fn_min) | (
            self.froude_numbers > self.solver.settings.fn_max
        )
        if np.any(outside) and not self.solver.settings.allow_outside_envelope:
            raise ValueError("batch Froude numbers fall outside the declared solver envelope")
        self._rules: Dict[float, _CachedRule] = {}
        self._extensions = 0
        for froude in self.froude_numbers:
            self._certify_rule(template_hull, float(froude))

    @classmethod
    def from_hull(
        cls,
        template_hull: OffsetHull,
        froude_numbers: Sequence[float],
        solver: Optional[MichellSolver] = None,
    ) -> "MichellOperator":
        return cls(template_hull, froude_numbers, solver)

    def _build_rule(self, froude: float, cutoff: float) -> _CachedRule:
        x_over_l, _, _ = self.template_hull.coordinates_over_length
        x_span = float(x_over_l[-1] - x_over_l[0])
        cycle_width = 2.0 * math.pi * froude**2 / x_span
        # The adaptive solver's certificate includes a half-panel rerun.
        width = cycle_width / (2.0 * self.solver.settings.min_panels_per_cycle)
        fine_lambda, fine_weights = _fixed_transformed_rule(
            cutoff, width, self.solver.settings.fine_order
        )
        coarse_lambda, coarse_weights = _fixed_transformed_rule(
            cutoff, width, self.solver.settings.coarse_order
        )
        return _CachedRule(
            cutoff_lambda=cutoff,
            cycle_width=cycle_width,
            fine_lambda=fine_lambda,
            fine_weights=fine_weights,
            coarse_lambda=coarse_lambda,
            coarse_weights=coarse_weights,
        )

    def _certify_rule(self, hull: OffsetHull, froude: float) -> QuadratureResult:
        quadrature, _ = self.solver._integrate_case(hull, froude)
        if not quadrature.converged:
            raise RuntimeError(
                "cannot compile a batch rule from a non-converged template at Fn={:.6g}".format(
                    froude
                )
            )
        existing = self._rules.get(froude)
        cutoff = quadrature.cutoff_lambda
        if existing is not None:
            cutoff = max(cutoff, existing.cutoff_lambda)
            self._extensions += int(cutoff > existing.cutoff_lambda)
        self._rules[froude] = self._build_rule(froude, cutoff)
        return quadrature

    def _rule_integral(
        self,
        evaluator: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, bool]],
        lambda_nodes: np.ndarray,
        weights: np.ndarray,
        tail_start: float,
    ) -> Tuple[float, float, float, bool]:
        total = 0.0
        tail = 0.0
        maximum_cancellation = 1.0
        crosscheck_failed = False
        chunk = self.solver.settings.batch_chunk_size
        for start in range(0, lambda_nodes.size, chunk):
            stop = min(start + chunk, lambda_nodes.size)
            local_lambda = lambda_nodes[start:stop]
            amplitude, cancellation, failed = evaluator(local_lambda)
            contribution = weights[start:stop] * local_lambda * np.abs(amplitude) ** 2
            total += float(np.sum(contribution))
            tail += float(np.sum(contribution[local_lambda >= tail_start]))
            maximum_cancellation = max(
                maximum_cancellation, float(np.max(cancellation))
            )
            crosscheck_failed = crosscheck_failed or failed
        return total, tail, maximum_cancellation, crosscheck_failed

    def _cached_case(
        self, hull: OffsetHull, froude: float
    ) -> Tuple[float, float, float, float, bool, bool]:
        x_over_l, z_over_l, y_over_l = hull.coordinates_over_length
        evaluator = self.solver._make_evaluator(
            x_over_l, z_over_l, y_over_l, froude
        )
        rule = self._rules[froude]
        tail_start = max(
            1.0,
            rule.cutoff_lambda
            - self.solver.settings.tail_consecutive_cycles * rule.cycle_width,
        )
        fine, tail, fine_cancel, fine_failed = self._rule_integral(
            evaluator, rule.fine_lambda, rule.fine_weights, tail_start
        )
        coarse, _, coarse_cancel, coarse_failed = self._rule_integral(
            evaluator, rule.coarse_lambda, rule.coarse_weights, tail_start
        )
        error = abs(fine - coarse)
        tolerance = max(
            self.solver.settings.absolute_tolerance,
            self.solver.settings.relative_tolerance
            * max(fine, self.solver.settings.absolute_tolerance),
        )
        tail_fraction = tail / max(fine, self.solver.settings.absolute_tolerance)
        accepted = error <= tolerance and tail <= tolerance
        return (
            fine,
            error,
            tail_fraction,
            max(fine_cancel, coarse_cancel),
            accepted,
            fine_failed or coarse_failed,
        )

    def solve_batch(
        self,
        half_breadth_batch: np.ndarray,
        wetted_areas_m2: Optional[Sequence[float]] = None,
    ) -> BatchWaveResistanceResult:
        """Evaluate dimensional half-breadth tensors on the template grid."""

        batch = np.asarray(half_breadth_batch, dtype=float)
        expected = self.template_hull.half_breadths.shape
        if batch.ndim == 2:
            batch = batch[np.newaxis, ...]
        if batch.ndim != 3 or batch.shape[1:] != expected:
            raise ValueError(
                "half_breadth_batch must have shape (n_hulls, {}, {})".format(*expected)
            )
        if not np.all(np.isfinite(batch)) or np.any(batch < 0.0):
            raise ValueError("batch half-breadths must be finite and non-negative")
        if wetted_areas_m2 is not None:
            areas = np.asarray(wetted_areas_m2, dtype=float)
            if areas.shape != (batch.shape[0],) or np.any(~np.isfinite(areas)) or np.any(
                areas <= 0.0
            ):
                raise ValueError("wetted_areas_m2 must contain one positive value per hull")
        else:
            areas = np.asarray(
                [
                    _estimated_wetted_area(
                        self.template_hull.x,
                        self.template_hull.z,
                        half_breadths,
                    )
                    for half_breadths in batch
                ]
            )

        results: List[WaveResistanceResult] = []
        extensions_before = self._extensions
        for hull_index, half_breadths in enumerate(batch):
            metadata = replace(
                self.template_hull.metadata,
                name="{} batch {}".format(self.template_hull.metadata.name, hull_index),
                wetted_area_m2=float(areas[hull_index]),
                beam_m=2.0 * float(np.max(half_breadths)),
                draft_m=float(self.template_hull.z[-1]),
            )
            hull = OffsetHull.from_tensor(
                metadata,
                self.template_hull.x,
                self.template_hull.z,
                half_breadths,
            )

            case_values = []
            fallback_needed = False
            for froude in self.froude_numbers:
                cached = self._cached_case(hull, float(froude))
                case_values.append(cached)
                fallback_needed = fallback_needed or not cached[4]
            if fallback_needed:
                # The fully adaptive result is returned for this hull; any
                # larger cutoffs become the cached basis for later hulls.
                adaptive = self.solver.solve(hull, self.froude_numbers)
                for froude, is_converged in zip(self.froude_numbers, adaptive.converged):
                    if is_converged:
                        self._certify_rule(hull, float(froude))
                results.append(adaptive)
                continue

            speed: List[float] = []
            resistance: List[float] = []
            coefficient: List[float] = []
            flags: List[Tuple[str, ...]] = []
            for froude, cached in zip(self.froude_numbers, case_values):
                integral, _, _, cancellation, _, crosscheck_failed = cached
                physical = self.solver._physical_outputs(hull, float(froude), integral)
                speed.append(physical[0])
                resistance.append(physical[1])
                coefficient.append(physical[2])
                case_flags = self.solver._base_validity_flags(hull, float(froude))
                if crosscheck_failed:
                    case_flags.append("direct and integration-by-parts amplitudes disagree")
                if np.finfo(float).eps * cancellation > self.solver.settings.relative_tolerance:
                    case_flags.append("severe complex-amplitude cancellation")
                if cached[2] > self.solver.settings.relative_tolerance:
                    case_flags.append(
                        "tail accuracy is controlled by the absolute, not relative, tolerance"
                    )
                flags.append(tuple(case_flags))
            results.append(
                WaveResistanceResult(
                    fn=self.froude_numbers.copy(),
                    speed_m_s=np.asarray(speed),
                    wave_resistance_N=np.asarray(resistance),
                    c_wave_resistance=np.asarray(coefficient),
                    integral=np.asarray([case[0] for case in case_values]),
                    quadrature_error=np.asarray([case[1] for case in case_values]),
                    tail_fraction=np.asarray([case[2] for case in case_values]),
                    cancellation_ratio=np.asarray([case[3] for case in case_values]),
                    converged=np.ones(self.froude_numbers.shape, dtype=bool),
                    validity_flags=tuple(flags),
                )
            )
        return BatchWaveResistanceResult(
            results=tuple(results),
            operator_extensions=self._extensions - extensions_before,
        )


__all__ = ["MichellOperator", "MichellSolver"]
