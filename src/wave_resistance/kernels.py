r"""Numerical kernels for Michell's thin-ship amplitude.

The geometry convention in this module is deliberately strict: ``X``, ``Z``
and ``Y`` are respectively ``x/L``, ``z/L`` and ``y/L`` for one common
reference length ``L``.  ``Z`` is positive downwards from the calm
waterline, and ``Y`` is the half-breadth.  Mixing the more usual ``x/L``,
``z/T`` and ``y/B`` normalisations gives a dimensionally incorrect kernel.

For a bilinear half-breadth field, the dimensionless Michell amplitude is

.. math::

   a(\lambda,F_n) = \int\!\!\int Y_X(X,Z)
      \exp(-\lambda^2 Z/F_n^2)
      \exp(-i\lambda X/F_n^2)\,dX\,dZ.

Both a direct cellwise evaluation and an algebraically equivalent
integration-by-parts evaluation are provided.  The latter retains the two
global end-boundary terms, so the two implementations also agree for a
non-zero end section mathematically.  Such an end section may represent an
immersed transom, but supporting the algebraic term does not cure the
physical limitations of thin-ship theory for separated transom flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = [
    "SummationDiagnostics",
    "normalized_sinc",
    "decaying_linear_basis_moments",
    "oscillatory_linear_basis_moments",
    "compensated_sum",
    "cancellation_ratio",
    "sum_with_diagnostics",
    "bilinear_cell_amplitudes_direct",
    "bilinear_cell_amplitudes_ibp",
    "michell_amplitude_direct",
    "michell_amplitude_ibp",
]


@dataclass(frozen=True)
class SummationDiagnostics:
    """Diagnostics for a compensated reduction.

    ``cancellation_ratio`` is ``sum(abs(terms)) / abs(sum(terms))``.  It is
    one when all non-zero terms have the same phase and tends to infinity as
    cancellation becomes severe.  The all-zero sum is assigned a ratio of
    one.  ``roundoff_bound`` is the conservative ordinary-summation
    ``gamma_n`` bound; compensated summation will normally be appreciably
    more accurate.
    """

    absolute_sum: Union[NDArray[np.floating], np.floating]
    cancellation_ratio: Union[NDArray[np.floating], np.floating]
    roundoff_bound: Union[NDArray[np.floating], np.floating]
    term_count: int


def normalized_sinc(argument: ArrayLike) -> Union[NDArray[np.floating], np.floating]:
    """Return ``sin(argument) / argument`` with a stable origin limit."""

    value = np.asarray(argument, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError("sinc arguments must be finite")

    result = np.empty_like(value)
    small = np.abs(value) <= 1.0e-4
    squared = value[small] * value[small]
    # Horner form through x^10; the remainder is negligible at the switch.
    result[small] = 1.0 + squared * (
        -1.0 / 6.0
        + squared
        * (
            1.0 / 120.0
            + squared
            * (-1.0 / 5040.0 + squared * (1.0 / 362880.0 - squared / 39916800.0))
        )
    )
    result[~small] = np.sin(value[~small]) / value[~small]
    return result[()] if result.ndim == 0 else result


def _horner(
    variable: NDArray, coefficients: Iterable[Union[complex, float]]
) -> NDArray:
    coefficients = tuple(coefficients)
    result = np.zeros_like(variable, dtype=np.result_type(variable, coefficients[-1]))
    result[...] = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = coefficient + variable * result
    return result


def decaying_linear_basis_moments(
    exponent: ArrayLike,
) -> tuple[
    Union[NDArray[np.floating], np.floating],
    Union[NDArray[np.floating], np.floating],
]:
    """Return exact linear-basis moments for a non-negative real exponent.

    The returned values are

    ``B0(a) = integral_0^1 (1-u) exp(-a u) du`` and
    ``B1(a) = integral_0^1 u exp(-a u) du``.

    Series are used near zero and ``expm1``-based expressions elsewhere.
    """

    value = np.asarray(exponent, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("decaying exponents must be finite and non-negative")

    b0 = np.empty_like(value)
    b1 = np.empty_like(value)
    small = value <= 1.0e-3

    # Coefficient n is 1/[n! (n+1) (n+2)] for B0 and 1/[n! (n+2)] for B1,
    # evaluated at variable -a.
    variable = -value[small]
    b0[small] = _horner(
        variable,
        (1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720, 1 / 5040, 1 / 40320),
    )
    b1[small] = _horner(
        variable,
        (1 / 2, 1 / 3, 1 / 8, 1 / 30, 1 / 144, 1 / 840, 1 / 5760),
    )

    regular = ~small
    if np.any(regular):
        a = value[regular]
        exp_minus_a = np.exp(-a)
        constant_moment = -np.expm1(-a) / a
        b1_regular = (constant_moment - exp_minus_a) / a
        b1[regular] = b1_regular
        b0[regular] = constant_moment - b1_regular

    if b0.ndim == 0:
        return b0[()], b1[()]
    return b0, b1


def oscillatory_linear_basis_moments(
    phase: ArrayLike,
) -> tuple[
    Union[NDArray[np.complexfloating], np.complexfloating],
    Union[NDArray[np.complexfloating], np.complexfloating],
]:
    """Return linear-basis moments of ``exp(-1j * phase * u)`` on [0, 1].

    The first result multiplies the left basis function ``1-u`` and the
    second the right basis function ``u``.  Small phases use power series;
    the regular expression uses the centred normalized sinc.
    """

    value = np.asarray(phase, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError("oscillatory phases must be finite")

    c0 = np.empty(value.shape, dtype=complex)
    c1 = np.empty(value.shape, dtype=complex)
    small = np.abs(value) <= 1.0e-3

    # exp(-i q u) series.  Ten terms make the truncation immaterial at the
    # switch and preserve the conjugate symmetry for negative q.
    variable = -1j * value[small]
    factorial = 1.0
    coefficients0: list[float] = []
    coefficients1: list[float] = []
    for order in range(11):
        if order:
            factorial *= order
        coefficients0.append(1.0 / (factorial * (order + 1) * (order + 2)))
        coefficients1.append(1.0 / (factorial * (order + 2)))
    c0[small] = _horner(variable, coefficients0)
    c1[small] = _horner(variable, coefficients1)

    regular = ~small
    if np.any(regular):
        q = value[regular]
        constant_moment = np.exp(-0.5j * q) * normalized_sinc(0.5 * q)
        right_moment = (constant_moment - np.exp(-1j * q)) / (1j * q)
        c1[regular] = right_moment
        c0[regular] = constant_moment - right_moment

    if c0.ndim == 0:
        return c0[()], c1[()]
    return c0, c1


def _normalise_axes(
    axis: Optional[Union[int, tuple[int, ...]]], ndim: int
) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    raw_axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes: list[int] = []
    for raw_axis in raw_axes:
        normalised = raw_axis + ndim if raw_axis < 0 else raw_axis
        if normalised < 0 or normalised >= ndim:
            raise np.AxisError(raw_axis, ndim=ndim)
        if normalised in axes:
            raise ValueError("reduction axes must be unique")
        axes.append(normalised)
    return tuple(sorted(axes))


def _flatten_reduction_axes(
    values: ArrayLike, axis: Optional[Union[int, tuple[int, ...]]]
) -> tuple[NDArray, tuple[int, ...], int]:
    array = np.asarray(values)
    axes = _normalise_axes(axis, array.ndim)
    if not axes:
        return array[..., None], array.shape, 1
    kept = tuple(index for index in range(array.ndim) if index not in axes)
    permutation = kept + axes
    moved = np.transpose(array, permutation)
    output_shape = tuple(array.shape[index] for index in kept)
    term_count = int(np.prod([array.shape[index] for index in axes], dtype=int))
    return moved.reshape(output_shape + (term_count,)), output_shape, term_count


def compensated_sum(
    values: ArrayLike, axis: Optional[Union[int, tuple[int, ...]]] = None
) -> Union[NDArray, np.number]:
    """Block-pairwise, Neumaier-compensated sum over one or more axes.

    NumPy performs each moderate block reduction in compiled pairwise code;
    the much smaller sequence of block totals is then Neumaier-compensated.
    This retains a compensation step for complex cancellation without a
    Python loop over every hull cell.
    """

    flat, output_shape, _ = _flatten_reduction_axes(values, axis)
    target_dtype = np.result_type(
        flat.dtype, np.complex128 if np.iscomplexobj(flat) else np.float64
    )
    flat = flat.astype(target_dtype, copy=False)
    if flat.shape[-1] <= 16:
        # Preserve the classic Neumaier behaviour for short, adversarial
        # sequences such as [1e16, 1, -1e16].
        block_totals = [flat[..., index] for index in range(flat.shape[-1])]
    else:
        block_size = 32
        block_totals = [
            np.sum(flat[..., start : start + block_size], axis=-1, dtype=target_dtype)
            for start in range(0, flat.shape[-1], block_size)
        ]
    total = np.zeros(output_shape, dtype=target_dtype)
    correction = np.zeros(output_shape, dtype=target_dtype)

    for term in block_totals:
        provisional = total + term
        correction += np.where(
            np.abs(total) >= np.abs(term),
            (total - provisional) + term,
            (term - provisional) + total,
        )
        total = provisional
    result = total + correction
    return result[()] if result.ndim == 0 else result


def cancellation_ratio(
    values: ArrayLike,
    total: Optional[ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
) -> Union[NDArray[np.floating], np.floating]:
    """Return ``sum(abs(values)) / abs(total)`` for a reduction."""

    absolute_sum = compensated_sum(np.abs(np.asarray(values)), axis=axis)
    if total is None:
        total = compensated_sum(values, axis=axis)
    numerator = np.asarray(absolute_sum)
    denominator = np.abs(np.asarray(total))
    ratio = np.ones(np.broadcast_shapes(numerator.shape, denominator.shape), dtype=float)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    nonzero_total = denominator > 0.0
    np.divide(numerator, denominator, out=ratio, where=nonzero_total)
    ratio[(~nonzero_total) & (numerator > 0.0)] = np.inf
    # Roundoff in the positive absolute sum can otherwise produce 1-eps.
    ratio = np.maximum(ratio, 1.0)
    return ratio[()] if ratio.ndim == 0 else ratio


def sum_with_diagnostics(
    values: ArrayLike, axis: Optional[Union[int, tuple[int, ...]]] = None
) -> tuple[Union[NDArray, np.number], SummationDiagnostics]:
    """Compensate a sum and return cancellation and roundoff diagnostics."""

    array = np.asarray(values)
    _, _, term_count = _flatten_reduction_axes(array, axis)
    total = compensated_sum(array, axis=axis)
    absolute_sum = compensated_sum(np.abs(array), axis=axis)
    ratio = cancellation_ratio(array, total=total, axis=axis)

    real_dtype = np.asarray(array.real).dtype
    if not np.issubdtype(real_dtype, np.inexact):
        real_dtype = np.dtype(float)
    epsilon = np.finfo(real_dtype).eps
    product = term_count * epsilon
    gamma = np.inf if product >= 1.0 else product / (1.0 - product)
    roundoff_bound = gamma * np.asarray(absolute_sum)
    if np.ndim(roundoff_bound) == 0:
        roundoff_bound = roundoff_bound[()]

    diagnostics = SummationDiagnostics(
        absolute_sum=absolute_sum,
        cancellation_ratio=ratio,
        roundoff_bound=roundoff_bound,
        term_count=term_count,
    )
    return total, diagnostics


def _validated_parameters(
    lambda_: ArrayLike, froude: ArrayLike
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    lambda_array = np.asarray(lambda_, dtype=float)
    froude_array = np.asarray(froude, dtype=float)
    lambda_array, froude_array = np.broadcast_arrays(lambda_array, froude_array)
    if np.any(~np.isfinite(lambda_array)) or np.any(lambda_array < 1.0):
        raise ValueError("Michell integration parameters lambda must be finite and >= 1")
    if np.any(~np.isfinite(froude_array)) or np.any(froude_array <= 0.0):
        raise ValueError("Froude numbers must be finite and positive")
    return lambda_array, froude_array


def _validated_shared_length_grid(
    x_over_l: ArrayLike, z_over_l: ArrayLike, y_over_l: ArrayLike
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    x = np.asarray(x_over_l, dtype=float)
    z = np.asarray(z_over_l, dtype=float)
    y = np.asarray(y_over_l, dtype=float)
    if x.ndim != 1 or z.ndim != 1 or x.size < 2 or z.size < 2:
        raise ValueError("X=x/L and Z=z/L must be one-dimensional nodal arrays")
    if y.shape != (x.size, z.size):
        raise ValueError("Y=y/L must have shape (len(X), len(Z))")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(z)) or np.any(~np.isfinite(y)):
        raise ValueError("the dimensionless hull grid must be finite")
    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(z) <= 0.0):
        raise ValueError("X and Z nodes must be strictly increasing")
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(z))))
    if abs(float(z[0])) > tolerance or np.any(z < -tolerance):
        raise ValueError("Z=z/L must start at the waterline Z=0 and be positive downwards")
    if np.any(y < 0.0):
        raise ValueError("Y=y/L must contain non-negative half-breadths")
    if z[0] != 0.0:
        z = z.copy()
        z[0] = 0.0
    return x, z, y


def bilinear_cell_amplitudes_direct(
    lambda_: ArrayLike,
    froude: ArrayLike,
    x_over_l: ArrayLike,
    z_over_l: ArrayLike,
    y_over_l: ArrayLike,
) -> NDArray[np.complexfloating]:
    """Return each cell's exact direct ``Y_X`` amplitude contribution.

    ``lambda_`` and ``froude`` follow ordinary NumPy broadcasting.  The
    returned shape is their broadcast shape followed by
    ``(len(x_over_l)-1, len(z_over_l)-1)``.
    """

    lambda_array, froude_array = _validated_parameters(lambda_, froude)
    x, z, y = _validated_shared_length_grid(x_over_l, z_over_l, y_over_l)

    inverse_froude_squared = 1.0 / (froude_array * froude_array)
    beta = lambda_array * inverse_froude_squared
    alpha = lambda_array * lambda_array * inverse_froude_squared

    dx = np.diff(x)
    dz = np.diff(z)
    x_centre = 0.5 * (x[:-1] + x[1:])
    q = beta[..., None] * dx
    x_moment = (
        dx
        * np.exp(-1j * beta[..., None] * x_centre)
        * normalized_sinc(0.5 * q)
    )

    decay = alpha[..., None] * dz
    b0, b1 = decaying_linear_basis_moments(decay)
    common_z = dz * np.exp(-alpha[..., None] * z[:-1])
    z_left_weight = common_z * b0
    z_right_weight = common_z * b1

    longitudinal_slopes = np.diff(y, axis=0) / dx[:, None]
    z_integral = (
        longitudinal_slopes[:, :-1] * z_left_weight[..., None, :]
        + longitudinal_slopes[:, 1:] * z_right_weight[..., None, :]
    )
    return x_moment[..., :, None] * z_integral


def bilinear_cell_amplitudes_ibp(
    lambda_: ArrayLike,
    froude: ArrayLike,
    x_over_l: ArrayLike,
    z_over_l: ArrayLike,
    y_over_l: ArrayLike,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Return exact integration-by-parts volume and end-boundary terms.

    The first array contains ``i beta`` times each bilinear cell's offset
    integral and has trailing shape ``(nx-1, nz-1)``.  The second contains
    the left and right end contributions separately and has trailing shape
    ``(2, nz-1)``.  Keeping the ends separate preserves cancellation
    diagnostics.  For a pointed hull both end arrays are exactly zero.
    """

    lambda_array, froude_array = _validated_parameters(lambda_, froude)
    x, z, y = _validated_shared_length_grid(x_over_l, z_over_l, y_over_l)

    inverse_froude_squared = 1.0 / (froude_array * froude_array)
    beta = lambda_array * inverse_froude_squared
    alpha = lambda_array * lambda_array * inverse_froude_squared

    dx = np.diff(x)
    dz = np.diff(z)
    q = beta[..., None] * dx
    c0, c1 = oscillatory_linear_basis_moments(q)
    x_phase = np.exp(-1j * beta[..., None] * x[:-1])
    x_left_weight = dx * x_phase * c0
    x_right_weight = dx * x_phase * c1

    decay = alpha[..., None] * dz
    b0, b1 = decaying_linear_basis_moments(decay)
    common_z = dz * np.exp(-alpha[..., None] * z[:-1])
    z_left_weight = common_z * b0
    z_right_weight = common_z * b1

    y00 = y[:-1, :-1]
    y10 = y[1:, :-1]
    y01 = y[:-1, 1:]
    y11 = y[1:, 1:]
    offset_integral = (
        y00 * x_left_weight[..., :, None] * z_left_weight[..., None, :]
        + y10 * x_right_weight[..., :, None] * z_left_weight[..., None, :]
        + y01 * x_left_weight[..., :, None] * z_right_weight[..., None, :]
        + y11 * x_right_weight[..., :, None] * z_right_weight[..., None, :]
    )
    volume_terms = 1j * beta[..., None, None] * offset_integral

    left_section = (
        y[0, :-1] * z_left_weight + y[0, 1:] * z_right_weight
    )
    right_section = (
        y[-1, :-1] * z_left_weight + y[-1, 1:] * z_right_weight
    )
    left_terms = -np.exp(-1j * beta * x[0])[..., None] * left_section
    right_terms = np.exp(-1j * beta * x[-1])[..., None] * right_section
    boundary_terms = np.stack((left_terms, right_terms), axis=-2)
    return volume_terms, boundary_terms


def michell_amplitude_direct(
    lambda_: ArrayLike,
    froude: ArrayLike,
    x_over_l: ArrayLike,
    z_over_l: ArrayLike,
    y_over_l: ArrayLike,
) -> tuple[
    Union[NDArray[np.complexfloating], np.complexfloating],
    Union[NDArray[np.floating], np.floating],
]:
    """Return direct Michell amplitude and its cancellation ratio.

    The amplitude shape is the NumPy broadcast shape of ``lambda_`` and
    ``froude``; in particular a one-dimensional lambda array and scalar
    Froude number return one-dimensional arrays.
    """

    lambda_array, froude_array = _validated_parameters(lambda_, froude)
    x, z, y = _validated_shared_length_grid(x_over_l, z_over_l, y_over_l)
    inverse_froude_squared = 1.0 / (froude_array * froude_array)
    beta = lambda_array * inverse_froude_squared
    alpha = lambda_array * lambda_array * inverse_froude_squared

    dx = np.diff(x)
    dz = np.diff(z)
    x_centre = 0.5 * (x[:-1] + x[1:])
    x_moment = (
        dx
        * np.exp(-1j * beta[..., None] * x_centre)
        * normalized_sinc(0.5 * beta[..., None] * dx)
    )
    decay = alpha[..., None] * dz
    b0, b1 = decaying_linear_basis_moments(decay)
    common_z = dz * np.exp(-alpha[..., None] * z[:-1])
    z_left = common_z * b0
    z_right = common_z * b1
    z_nodal = np.zeros(alpha.shape + (z.size,), dtype=float)
    z_nodal[..., :-1] += z_left
    z_nodal[..., 1:] += z_right

    slopes = np.diff(y, axis=0) / dx[:, None]
    vertical_projection = np.einsum("ij,...j->...i", slopes, z_nodal, optimize=True)
    terms = x_moment * vertical_projection
    amplitude, diagnostics = sum_with_diagnostics(terms, axis=-1)
    return amplitude, diagnostics.cancellation_ratio


def michell_amplitude_ibp(
    lambda_: ArrayLike,
    froude: ArrayLike,
    x_over_l: ArrayLike,
    z_over_l: ArrayLike,
    y_over_l: ArrayLike,
) -> tuple[
    Union[NDArray[np.complexfloating], np.complexfloating],
    Union[NDArray[np.floating], np.floating],
]:
    """Return integration-by-parts Michell amplitude and cancellation ratio."""

    lambda_array, froude_array = _validated_parameters(lambda_, froude)
    x, z, y = _validated_shared_length_grid(x_over_l, z_over_l, y_over_l)
    inverse_froude_squared = 1.0 / (froude_array * froude_array)
    beta = lambda_array * inverse_froude_squared
    alpha = lambda_array * lambda_array * inverse_froude_squared

    dx = np.diff(x)
    dz = np.diff(z)
    phase = beta[..., None] * dx
    c0, c1 = oscillatory_linear_basis_moments(phase)
    x_phase = np.exp(-1j * beta[..., None] * x[:-1])
    x_left = dx * x_phase * c0
    x_right = dx * x_phase * c1
    x_nodal = np.zeros(beta.shape + (x.size,), dtype=complex)
    x_nodal[..., :-1] += x_left
    x_nodal[..., 1:] += x_right

    decay = alpha[..., None] * dz
    b0, b1 = decaying_linear_basis_moments(decay)
    common_z = dz * np.exp(-alpha[..., None] * z[:-1])
    z_left = common_z * b0
    z_right = common_z * b1
    z_nodal = np.zeros(alpha.shape + (z.size,), dtype=float)
    z_nodal[..., :-1] += z_left
    z_nodal[..., 1:] += z_right

    vertical_projection = np.einsum("ij,...j->...i", y, z_nodal, optimize=True)
    volume_terms = 1j * beta[..., None] * x_nodal * vertical_projection
    left_section = np.einsum("j,...j->...", y[0], z_nodal, optimize=True)
    right_section = np.einsum("j,...j->...", y[-1], z_nodal, optimize=True)
    boundary_terms = np.stack(
        (
            -np.exp(-1j * beta * x[0]) * left_section,
            np.exp(-1j * beta * x[-1]) * right_section,
        ),
        axis=-1,
    )
    all_terms = np.concatenate((volume_terms, boundary_terms), axis=-1)
    amplitude, diagnostics = sum_with_diagnostics(all_terms, axis=-1)
    return amplitude, diagnostics.cancellation_ratio
