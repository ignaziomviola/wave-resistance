"""Phase-aware quadrature for Michell's improper outer integral."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from .models import SolverSettings


# amplitude, cancellation ratio, cross-check-failed
AmplitudeEvaluator = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, bool]]


@dataclass
class QuadratureResult:
    integral: float
    error: float
    tail_fraction: float
    cancellation_ratio: float
    converged: bool
    crosscheck_failed: bool
    cutoff_lambda: float
    lambda_nodes: np.ndarray
    t_weights: np.ndarray
    phase_cycles: int


@dataclass
class _RangeResult:
    integral: float
    error: float
    cancellation_ratio: float
    crosscheck_failed: bool
    lambda_nodes: np.ndarray
    t_weights: np.ndarray


class PhaseAwareMichellIntegrator:
    """Integrate ``lambda * |a(lambda)|^2`` in transformed ``t`` space.

    The substitution ``lambda=sqrt(1+t^2)`` removes Michell's integrable
    singularity.  Initial panels are nevertheless laid out in lambda so the
    bow--stern interference phase cannot be skipped by an adaptive estimator.
    """

    def __init__(self, settings: SolverSettings) -> None:
        self.settings = settings
        self._rules = {
            settings.coarse_order: leggauss(settings.coarse_order),
            settings.fine_order: leggauss(settings.fine_order),
        }

    @staticmethod
    def _t_from_lambda(lambda_: float) -> float:
        return math.sqrt(max(0.0, (lambda_ - 1.0) * (lambda_ + 1.0)))

    def _fixed_panel(
        self,
        lambda_a: float,
        lambda_b: float,
        order: int,
        evaluate: AmplitudeEvaluator,
    ) -> Tuple[float, float, bool, np.ndarray, np.ndarray]:
        nodes, weights = self._rules[order]
        t_a = self._t_from_lambda(lambda_a)
        t_b = self._t_from_lambda(lambda_b)
        half = 0.5 * (t_b - t_a)
        midpoint = 0.5 * (t_a + t_b)
        t_nodes = midpoint + half * nodes
        lambda_nodes = np.sqrt(1.0 + t_nodes * t_nodes)
        amplitude, cancellation, crosscheck_failed = evaluate(lambda_nodes)
        amplitude = np.asarray(amplitude, dtype=complex)
        cancellation = np.asarray(cancellation, dtype=float)
        if amplitude.shape != lambda_nodes.shape:
            raise ValueError("amplitude evaluator returned an incompatible shape")
        if cancellation.shape not in ((), lambda_nodes.shape):
            raise ValueError("cancellation diagnostic returned an incompatible shape")
        t_weights = half * weights
        values = lambda_nodes * np.abs(amplitude) ** 2
        integral = float(np.dot(t_weights, values))
        ratio = float(np.max(cancellation)) if cancellation.size else 1.0
        return integral, ratio, bool(crosscheck_failed), lambda_nodes, t_weights

    def _fixed_panels_batch(
        self,
        lambda_a: np.ndarray,
        lambda_b: np.ndarray,
        order: int,
        evaluate: AmplitudeEvaluator,
    ) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray]:
        """Evaluate several base panels in one vectorized kernel call."""

        nodes, weights = self._rules[order]
        t_a = np.sqrt(np.maximum(0.0, (lambda_a - 1.0) * (lambda_a + 1.0)))
        t_b = np.sqrt(np.maximum(0.0, (lambda_b - 1.0) * (lambda_b + 1.0)))
        half = 0.5 * (t_b - t_a)
        midpoint = 0.5 * (t_a + t_b)
        t_nodes = midpoint[:, None] + half[:, None] * nodes[None, :]
        lambda_nodes = np.sqrt(1.0 + t_nodes * t_nodes)
        amplitude, cancellation, crosscheck_failed = evaluate(lambda_nodes)
        amplitude = np.asarray(amplitude, dtype=complex)
        cancellation = np.broadcast_to(np.asarray(cancellation, dtype=float), lambda_nodes.shape)
        if amplitude.shape != lambda_nodes.shape:
            raise ValueError("amplitude evaluator returned an incompatible shape")
        t_weights = half[:, None] * weights[None, :]
        values = lambda_nodes * np.abs(amplitude) ** 2
        integrals = np.sum(t_weights * values, axis=1)
        ratios = np.max(cancellation, axis=1)
        return integrals, ratios, bool(crosscheck_failed), lambda_nodes, t_weights

    def _adaptive_panel(
        self,
        lambda_a: float,
        lambda_b: float,
        evaluate: AmplitudeEvaluator,
        absolute_budget: float,
        depth: int = 0,
    ) -> _RangeResult:
        coarse, coarse_ratio, coarse_failed, _, _ = self._fixed_panel(
            lambda_a, lambda_b, self.settings.coarse_order, evaluate
        )
        fine, fine_ratio, fine_failed, nodes, weights = self._fixed_panel(
            lambda_a, lambda_b, self.settings.fine_order, evaluate
        )
        error = abs(fine - coarse)
        tolerance = absolute_budget + self.settings.relative_tolerance * abs(fine)
        if error <= tolerance or depth >= self.settings.max_panel_refinements:
            return _RangeResult(
                integral=max(0.0, fine),
                error=error,
                cancellation_ratio=max(coarse_ratio, fine_ratio),
                crosscheck_failed=coarse_failed or fine_failed,
                lambda_nodes=nodes,
                t_weights=weights,
            )

        midpoint = 0.5 * (lambda_a + lambda_b)
        left = self._adaptive_panel(
            lambda_a, midpoint, evaluate, 0.5 * absolute_budget, depth + 1
        )
        right = self._adaptive_panel(
            midpoint, lambda_b, evaluate, 0.5 * absolute_budget, depth + 1
        )
        return self._combine((left, right))

    @staticmethod
    def _combine(parts: Tuple[_RangeResult, ...]) -> _RangeResult:
        if not parts:
            return _RangeResult(0.0, 0.0, 1.0, False, np.array([]), np.array([]))
        return _RangeResult(
            integral=float(sum(part.integral for part in parts)),
            error=float(sum(part.error for part in parts)),
            cancellation_ratio=max(part.cancellation_ratio for part in parts),
            crosscheck_failed=any(part.crosscheck_failed for part in parts),
            lambda_nodes=np.concatenate([part.lambda_nodes for part in parts]),
            t_weights=np.concatenate([part.t_weights for part in parts]),
        )

    def _integrate_range(
        self,
        lambda_a: float,
        lambda_b: float,
        maximum_panel_width: float,
        evaluate: AmplitudeEvaluator,
    ) -> _RangeResult:
        if lambda_b <= lambda_a:
            return self._combine(())
        panel_count = max(1, int(math.ceil((lambda_b - lambda_a) / maximum_panel_width)))
        edges = np.linspace(lambda_a, lambda_b, panel_count + 1)
        absolute_budget = self.settings.absolute_tolerance / panel_count
        parts: List[_RangeResult] = []
        # Batching avoids thousands of tiny geometry-kernel calls while
        # retaining an independent 16/32 estimate for every physical panel.
        panels_per_batch = 32
        for start in range(0, panel_count, panels_per_batch):
            stop = min(start + panels_per_batch, panel_count)
            left_edges = edges[start:stop]
            right_edges = edges[start + 1 : stop + 1]
            coarse = self._fixed_panels_batch(
                left_edges, right_edges, self.settings.coarse_order, evaluate
            )
            fine = self._fixed_panels_batch(
                left_edges, right_edges, self.settings.fine_order, evaluate
            )
            errors = np.abs(fine[0] - coarse[0])
            tolerances = absolute_budget + self.settings.relative_tolerance * np.abs(fine[0])
            accepted = errors <= tolerances
            if np.any(accepted):
                accepted_nodes = fine[3][accepted].reshape(-1)
                accepted_weights = fine[4][accepted].reshape(-1)
                parts.append(
                    _RangeResult(
                        integral=float(np.sum(np.maximum(0.0, fine[0][accepted]))),
                        error=float(np.sum(errors[accepted])),
                        cancellation_ratio=float(
                            max(np.max(coarse[1][accepted]), np.max(fine[1][accepted]))
                        ),
                        crosscheck_failed=coarse[2] or fine[2],
                        lambda_nodes=accepted_nodes,
                        t_weights=accepted_weights,
                    )
                )
            for local_index in np.flatnonzero(~accepted):
                parts.append(
                    self._adaptive_panel(
                        float(left_edges[local_index]),
                        float(right_edges[local_index]),
                        evaluate,
                        absolute_budget,
                    )
                )
        return self._combine(tuple(parts))

    def integrate(
        self,
        froude: float,
        x_span_over_length: float,
        evaluate: AmplitudeEvaluator,
    ) -> QuadratureResult:
        """Integrate one Froude-number case and retain the accepted rule."""

        if not math.isfinite(froude) or froude <= 0.0:
            raise ValueError("froude must be finite and positive")
        if not math.isfinite(x_span_over_length) or x_span_over_length <= 0.0:
            raise ValueError("x_span_over_length must be finite and positive")

        cycle_width = 2.0 * math.pi * froude * froude / x_span_over_length
        maximum_panel_width = cycle_width / self.settings.min_panels_per_cycle
        cycle_cap_lambda = 1.0 + self.settings.max_phase_cycles * cycle_width
        absolute_cap = min(self.settings.lambda_cap, cycle_cap_lambda)

        current = 1.0
        accumulated: List[_RangeResult] = []
        total = 0.0
        consecutive_small = 0
        cycles = 0
        crosscheck_failed = False
        maximum_cancellation = 1.0

        while current < absolute_cap and cycles < self.settings.max_phase_cycles:
            end = min(current + cycle_width, absolute_cap)
            cycle_result = self._integrate_range(
                current, end, maximum_panel_width, evaluate
            )
            accumulated.append(cycle_result)
            total += cycle_result.integral
            maximum_cancellation = max(
                maximum_cancellation, cycle_result.cancellation_ratio
            )
            crosscheck_failed = crosscheck_failed or cycle_result.crosscheck_failed
            current = end
            cycles += 1

            threshold = max(
                self.settings.absolute_tolerance,
                self.settings.relative_tolerance * max(total, self.settings.absolute_tolerance),
            )
            if cycle_result.integral <= threshold:
                consecutive_small += 1
            else:
                consecutive_small = 0

            if consecutive_small < self.settings.tail_consecutive_cycles:
                continue

            # Global half-panel re-evaluation protects against a consistently
            # aliased embedded estimate.  The extension doubles the resolved
            # lambda interval measured from the singular endpoint lambda=1.
            refined_width = 0.5 * maximum_panel_width
            refined = self._integrate_range(1.0, current, refined_width, evaluate)
            verify_end = min(1.0 + 2.0 * (current - 1.0), absolute_cap)
            extension = self._integrate_range(current, verify_end, refined_width, evaluate)
            verified_total = refined.integral + extension.integral
            verification_tolerance = max(
                self.settings.absolute_tolerance,
                self.settings.relative_tolerance
                * max(verified_total, self.settings.absolute_tolerance),
            )
            refinement_delta = abs(refined.integral - total)
            verified_error = refined.error + extension.error + refinement_delta
            maximum_cancellation = max(
                maximum_cancellation,
                refined.cancellation_ratio,
                extension.cancellation_ratio,
            )
            crosscheck_failed = (
                crosscheck_failed
                or refined.crosscheck_failed
                or extension.crosscheck_failed
            )
            converged = (
                refinement_delta <= verification_tolerance
                and extension.integral <= verification_tolerance
                and verify_end > current
            )
            if converged:
                denominator = max(verified_total, self.settings.absolute_tolerance)
                return QuadratureResult(
                    integral=max(0.0, verified_total),
                    error=verified_error,
                    tail_fraction=extension.integral / denominator,
                    cancellation_ratio=maximum_cancellation,
                    converged=True,
                    crosscheck_failed=crosscheck_failed,
                    cutoff_lambda=verify_end,
                    lambda_nodes=np.concatenate(
                        (refined.lambda_nodes, extension.lambda_nodes)
                    ),
                    t_weights=np.concatenate((refined.t_weights, extension.t_weights)),
                    phase_cycles=int(math.ceil((verify_end - 1.0) / cycle_width)),
                )

            # Keep the verified rule and continue beyond its cutoff.  This path
            # is uncommon but avoids discarding expensive evidence.
            accumulated = [refined, extension]
            total = verified_total
            current = verify_end
            maximum_panel_width = refined_width
            cycles = int(math.ceil((current - 1.0) / cycle_width))
            consecutive_small = 0

        combined = self._combine(tuple(accumulated))
        denominator = max(combined.integral, self.settings.absolute_tolerance)
        last_value = accumulated[-1].integral if accumulated else 0.0
        return QuadratureResult(
            integral=max(0.0, combined.integral),
            error=combined.error,
            tail_fraction=last_value / denominator,
            cancellation_ratio=max(maximum_cancellation, combined.cancellation_ratio),
            converged=False,
            crosscheck_failed=crosscheck_failed or combined.crosscheck_failed,
            cutoff_lambda=current,
            lambda_nodes=combined.lambda_nodes,
            t_weights=combined.t_weights,
            phase_cycles=cycles,
        )
