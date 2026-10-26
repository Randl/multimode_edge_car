#!/usr/bin/env python3
"""Generate data for the three conductance figures and the appendix response figure.

Uses the appendix recurrence mu[i+1] = (I + ell[i] A[i]) mu[i],
Eq. (A76) [paper:appeq:voltage-recurrence]. Temperature is in kelvin, lengths
in metres, and velocities in metres/second; KB_OVER_HBAR restores the
paper's natural units. Potentials are normalized by the source energy e*V_in.

The generic builder derives normal modes from K and U and accepts explicit
endpoint constraints. Couplings are uniform and fixed, with no momentum
mismatch or disorder; the dissipative bath is grounded. The paper presets
use equal velocities and no Gaussian integer/fractional mixing.

Numbers beside the [paper:...] markers are generated from LaTeX labels.
"""

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag, eigh
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import beta, hyp2f1

# Convert a temperature in kelvin to the angular-frequency scale k_B*T/hbar.
KB_OVER_HBAR = 1.380649e-23 / 1.054571817e-34


@lru_cache(maxsize=8)
def quadrature_rule(order):
    """Return cached Gauss-Legendre nodes and weights on the interval [0, 1].

    This is numerical quadrature for the positive spatial half of the
    pair-separation integral, Eq. (A49) [paper:appeq:current-integral].
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / 2


def time_integral(qp, qm, shift):
    """Integrate cosh(t+shift)^(-qp) * cosh(t-shift)^(-qm) over real t.

    For equal velocities qp=d+s and qm=d-s are the sums of squared vertex
    weights in the two chiralities: Eq. (A56) [paper:appeq:chiral-dimensions].
    This evaluates only the inner time integral in
    Eq. (A49) [paper:appeq:current-integral], with the denominator from
    Eq. (A45) [paper:appeq:shifted-denominator]. The hypergeometric reduction is
    Eq. (A50) [paper:appeq:equal-velocity-time-integral]. It leaves the spatial
    integral to numerical quadrature in _kernel; large spatial shifts also use
    direct time quadrature to avoid loss of floating-point accuracy.
    """
    d, q = (qp + qm) / 2, min(qp, qm)
    # At zero separation the integrand is cosh(t)^(-2d):
    # Eq. (A52) [paper:appeq:intcosh]. For a purely chiral vertex, translating t
    # removes the separation entirely, so the same result holds for every shift.
    prefactor = 2 ** (2 * d - 1) * beta(d, d)
    if q < 1e-12:
        return prefactor + np.zeros_like(shift)
    shifts = np.asarray(shift, dtype=float)
    values = np.empty_like(shifts)
    # Switch early, before 1-exp(-4*shift) approaches 1 at machine precision.
    regular = shifts <= 6
    z = shifts[regular]
    values[regular] = (
        prefactor * np.exp(-2 * q * z) * hyp2f1(q, d, 2 * d, -np.expm1(-4 * z))
    )
    for index in np.ndindex(shifts.shape):
        if not regular[index]:
            z = shifts[index]

            def integrand(t):
                """Evaluate the shifted correlator without overflowing cosh."""

                def logcosh(x):
                    """Compute log(cosh(x)) without forming an overflowing cosh."""
                    return np.logaddexp(x, -x) - np.log(2.0)

                return np.exp(-qp * logcosh(t + z) - qm * logcosh(t - z))

            values[index] = sum(
                quad(integrand, lo, hi, epsabs=1e-13, epsrel=1e-11)[0]
                for lo, hi in [(-np.inf, -z), (-z, z), (z, np.inf)]
            )
    return float(values) if values.ndim == 0 else values


@lru_cache(maxsize=100000)
def _kernel(temperature, d, spin, segment, cutoff, velocity, order):
    """Return the dimensionless coherent response A_r(T) at equal velocities.

    Implements Eq. (A48) [paper:appeq:current_final_3] and the spatial/time
    integral in Eq. (A49) [paper:appeq:current-integral]. Here d is the scaling
    dimension, spin is the conformal spin, and segment is the coherence length.
    The reduced spatial integral, Eq. (A51) [paper:appeq:equal-velocity-spatial-integral],
    is evaluated by quadrature, using the analytic time integral where stable.
    Only the fully chiral branch bypasses spatial quadrature with a closed form.
    """
    # z = k_B*T*a/(hbar*u); shift = ell_seg/ell_T with ell_T=hbar*u/(pi*k_B*T).
    z = KB_OVER_HBAR * temperature * cutoff / velocity
    ratio = segment / cutoff
    shift = np.pi * KB_OVER_HBAR * temperature * segment / velocity
    # A fully chiral vertex has only one nonzero chiral weight. Its time
    # shift can be removed, making the short-segment expression exact at all
    # lengths (discussion after Eq. (A57) [paper:appeq:current-long-segment]).
    if abs(d - abs(spin)) < 1e-12:
        return ratio / (2 * np.pi) * (2 * np.pi * z) ** (2 * d - 2) * beta(d, d)
    if shift <= 4:
        nodes, weights = quadrature_rule(order)
        # The triangular weight counts pairs of points at a given separation;
        # evenness turns the integral over [-1, 1] into 2*integral_0^1 (1-v) dv.
        integral = np.sum(
            weights * 2 * (1 - nodes) * time_integral(d + spin, d - spin, shift * nodes)
        )
    else:
        # With x=shift*v, adaptive integration resolves the narrow thermal peak.
        integral = (
            2
            / shift
            * quad(
                lambda x: (1 - x / shift) * time_integral(d + spin, d - spin, x),
                0,
                shift,
                epsabs=1e-12,
                epsrel=1e-10,
            )[0]
        )
    return ratio / (4 * np.pi) * (np.pi * z) ** (2 * d - 2) * integral


@dataclass(frozen=True)
class VertexTerms:
    """Physical vertices, couplings, and their derived normal-mode properties."""

    vectors: np.ndarray
    coefficients: np.ndarray
    weights: np.ndarray
    dimensions: np.ndarray
    spins: np.ndarray
    flux_vectors: np.ndarray


@dataclass(frozen=True)
class TransportModel:
    """A uniform Gaussian edge and its perturbations, constructed by build_model."""

    k_matrix: np.ndarray
    u_matrix: np.ndarray
    basis: np.ndarray
    velocities: np.ndarray
    chiralities: np.ndarray
    cosines: VertexTerms
    bath: VertexTerms
    boundary_start: np.ndarray
    boundary_end: np.ndarray
    boundary_values: np.ndarray
    current_weights: np.ndarray
    cutoff_length: float
    reference_velocity: float
    finger_length: float
    segment_length: float | Callable[[float], float]
    quadrature_order: int

    def coherence_length(self, temperature):
        """Evaluate the segmentation prescription in metres at temperature in K."""
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")
        length = (
            self.segment_length(temperature)
            if callable(self.segment_length)
            else self.segment_length
        )
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Segment length must be finite and positive.")
        return min(self.finger_length, length)

    def generator(self, temperature, segment=None):
        """Return the potential generator A(T, ell) in inverse metres.

        Each process contributes -rate * m (K^-1 m)^T, as in
        Eq. (A75) [paper:appeq:update-matrix]. Coherent rates use |y|^2;
        local grounded-bath rates use the dimensionless loss strength alpha.
        """
        default_segment = self.coherence_length(temperature)
        segment = default_segment if segment is None else segment
        if not np.isfinite(segment) or segment <= 0:
            raise ValueError("Segment length must be finite and positive.")
        a, u = self.cutoff_length, self.reference_velocity
        result = np.zeros_like(self.k_matrix)
        for i, coefficient in enumerate(self.cosines.coefficients):
            if coefficient == 0:
                continue
            response = vertex_kernel(
                temperature,
                self.cosines.weights[i],
                self.cosines.dimensions[i],
                self.cosines.spins[i],
                self.velocities,
                self.chiralities,
                segment,
                a,
                u,
                self.quadrature_order,
            )
            rate = 2 * np.pi / a * response * abs(coefficient) ** 2
            result -= rate * np.outer(
                self.cosines.vectors[i], self.cosines.flux_vectors[i]
            )
        z = KB_OVER_HBAR * temperature * a / u
        for i, alpha in enumerate(self.bath.coefficients):
            if alpha == 0:
                continue
            d = self.bath.dimensions[i]
            # The bath adds dimension 1/2; alpha absorbs its velocity.
            # Eq. (A70) [paper:appeq:dissipative-coefficient].
            velocity_factor = (
                (u / self.velocities[0]) ** (2 * d)
                if np.all(self.velocities == self.velocities[0])
                else np.exp(self.bath.weights[i] @ np.log(u / self.velocities))
            )
            response = (
                (2 * np.pi * z) ** (2 * d - 1)
                * velocity_factor
                * beta(d + 0.5, d + 0.5)
            )
            rate = 2 * np.pi / a * response * alpha
            result -= rate * np.outer(self.bath.vectors[i], self.bath.flux_vectors[i])
        return result

    def solve(self, temperature):
        """Return endpoint potentials and residual for the finite-segment model.

        Potentials are responses to unit source bias. Preserve extra precision
        until the current projection, since neutral potentials can cancel.
        """
        segment = self.coherence_length(temperature)
        lengths = segment_lengths(self.finger_length, segment)
        generator = self.generator(temperature, segment)
        updates, transfers = [], []
        for length in lengths:
            local = (
                generator
                if np.isclose(length, segment, rtol=1e-12, atol=0)
                else self.generator(temperature, length)
            )
            # I+ell*A is the discrete recurrence
            # Eq. (A76) [paper:appeq:voltage-recurrence].
            updates.append(length * local)
            transfers.append(np.eye(len(self.k_matrix)) + updates[-1])
        return discrete_solution(
            self.boundary_start,
            self.boundary_values,
            self.boundary_end,
            transfers,
            updates,
            preserve_precision=True,
        )

    def evaluate(self, temperature):
        """Compute G/G_Q for the supplied contacts.

        current_weights^T K^-1 mu(0) selects and orients the measured current,
        Eq. (A77) [paper:appeq:outgoing-current]. Boundary values specify the
        derivative with respect to source bias, so this gives the differential
        conductance in Eq. (A78) [paper:appeq:differential-conductance].
        """
        v0, _, _ = self.solve(temperature)
        if isinstance(v0, mp.matrix):
            with mp.workdps(60):
                charge = mp.lu_solve(mp.matrix(self.k_matrix.tolist()), v0)
                conductance = float(
                    sum(w * q for w, q in zip(self.current_weights, charge))
                )
        else:
            conductance = float(
                self.current_weights @ np.linalg.solve(self.k_matrix, v0)
            )
        return {"G_over_GQ": conductance}


def build_model(
    k_matrix,
    u_matrix,
    cosine_vectors,
    cosine_coefficients,
    *,
    boundary_start,
    boundary_end,
    boundary_values,
    current_weights,
    cutoff_length,
    reference_velocity,
    finger_length,
    segment_length,
    bath_vectors=(),
    bath_strengths=(),
    quadrature_order=96,
):
    """Build an n-mode transport model, deriving dimensions and spins from K and U.

    Parameters
    ----------
    k_matrix : array_like, shape (n, n)
        Symmetric, nonsingular integer K in the physical field basis.
        Integer-valued floating arrays are accepted; fractional entries are rejected.
    u_matrix : array_like, shape (n, n)
        Real symmetric positive-definite Gaussian interaction matrix U, in m/s,
        in the same basis: H_0 = hbar/(4*pi) integral dx (d_x phi)^T U (d_x phi).
        Mixing and unequal normal-mode velocities are supported.
    cosine_vectors : array_like, shape (r, n)
        Rows m_r in the uniform perturbations cos(m_r^T phi). Use [] for none.
        Hamiltonian vertices must have integer conformal spin.
        Identical vertices (including opposite signs) must be combined first.
    cosine_coefficients : array_like, shape (r,)
        Dimensionless real or complex y_r, with Hamiltonian density
        hbar*u_ref/(2*pi*a^2) sum_r [y_r exp(i*m_r^T phi) + h.c.].
        Thus a real y multiplies cos(m^T phi) by hbar*u_ref*y/(pi*a^2).
        Rates depend on |y_r|^2. There is no momentum mismatch or disorder.
    boundary_start, boundary_end : array_like, shape (n, n)
        General linear endpoint constraints B_0 mu(0) + B_L mu(L) = b.
        These may mix modes and couple the two endpoints. The n constraints
        must give a nonsingular boundary problem at the evaluated temperature.
    boundary_values : array_like, shape (n,)
        b above, in units of e*V_in: the imposed response to unit source bias.
        Use zero for grounded contacts and unbiased continuity constraints.
    current_weights : array_like, shape (n,)
        Signed charge weights selecting the measured current at x=0:
        G/G_Q = current_weights^T K^-1 mu(0). For bottom-edge outgoing current,
        use minus the charge vector on bottom modes and zero elsewhere.
    cutoff_length : float
        Ultraviolet cutoff a in metres, strictly positive.
    reference_velocity : float
        Positive u_ref in m/s used to normalize y_r and bath_strengths.
        Physical edge velocities are derived from K and U.
    finger_length : float
        Junction length L in metres, strictly positive.
    segment_length : float or callable
        Positive coherence length in metres, or function T[K] -> length[m].
        Clipped to L; a shorter final segment uses its own response kernel.
    bath_vectors : array_like, shape (b, n), optional
        Electron vertices coupled locally to a grounded dissipative bath.
        Use [] for no loss. The bath adds 1/2 to each edge scaling dimension.
    bath_strengths : array_like, shape (b,), optional
        Nonnegative dimensionless alpha=|w|^2/(2*pi*u_ref*v_bath), matching
        Eq. (A68) [paper:appeq:dissipation-strength]. v_bath is the bath's own
        positive velocity, independent of the edge velocities derived from K,U.
        Supply alpha after incorporating v_bath: at fixed w, doubling v_bath
        halves alpha and the bath rate. Do not include v_bath a second time.
    quadrature_order : int, optional
        Spatial Gauss-Legendre order (default 96); long segments use adaptive
        quadrature. Unequal velocities also require numerical time integration.

    Returns
    -------
    TransportModel
        Call model.generator(T) or model.evaluate(T), with T in kelvin.
        model.cosines and model.bath expose derived dimensions, spins, and
        squared normal-mode vertex weights. model.basis is W with
        W.T K W=diag(chiralities), W.T U W=diag(velocities).
        The model uses uniform, fixed couplings and the perturbative discrete
        recurrence.
    """

    def real_array(value, name):
        """Copy a finite real input so caller mutations cannot change the model."""
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real.")
        result = np.array(value, dtype=float, copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite.")
        return result

    k = real_array(k_matrix, "K")
    u = real_array(u_matrix, "U")
    if k.ndim != 2 or k.shape[0] == 0 or k.shape[0] != k.shape[1]:
        raise ValueError("K must be a nonempty square matrix.")
    if not np.array_equal(k, np.rint(k)):
        raise ValueError("K must have integer entries in the physical field basis.")
    n = len(k)
    for matrix, name in ((k, "K"), (u, "U")):
        if matrix.shape != (n, n) or not np.allclose(
            matrix, matrix.T, rtol=1e-12, atol=0
        ):
            raise ValueError(f"{name} must be symmetric and have shape {(n, n)}.")
        matrix[:] = (matrix + matrix.T) / 2
    try:
        # K v = lambda U v, v.T U v=1. Rescale by sqrt(|lambda|)
        # to obtain W.T K W=sign(lambda), W.T U W=1/|lambda|:
        # Eq. (A21) [paper:appeq:generalized-eigenproblem] and
        # Eq. (A22) [paper:appeq:generalized-eigenbasis]. Repeated eigenvalues
        # are allowed: eigh supplies a U-orthonormal basis in each eigenspace.
        eigenvalues, vectors = eigh(k, u)
        if np.any(abs(eigenvalues) <= np.finfo(float).eps * max(abs(eigenvalues)) * n):
            raise ValueError("K must be nonsingular and numerically resolvable.")
        basis = vectors / np.sqrt(abs(eigenvalues))
        velocities = 1 / abs(eigenvalues)
        chiralities = np.sign(eigenvalues)
        equal_velocities = np.allclose(velocities, velocities[0], rtol=1e-12, atol=0)
        if equal_velocities:
            # Use the basis-independent invariant u^2=tr((K^-1 U)^2)/n.
            # This avoids numerical splitting of degenerate velocities, which
            # would spuriously break the exact top/bottom symmetry of presets.
            dynamical = np.linalg.solve(k, u)
            velocities[:] = np.sqrt(np.trace(dynamical @ dynamical) / n)
    except np.linalg.LinAlgError as error:
        raise ValueError("U must be positive definite.") from error

    def vertex_terms(vertices, coefficients, name, dissipative=False):
        """Transform physical vertices and compute their dimensions and spins."""
        m = real_array(vertices, f"{name} vectors")
        if m.size == 0:
            m = np.empty((0, n))
        if m.ndim != 2 or m.shape[1] != n:
            raise ValueError(f"{name} vectors must have shape (count, {n}).")
        c = (
            real_array(coefficients, "Bath strengths")
            if dissipative
            else np.array(coefficients, dtype=complex, copy=True)
        )
        if c.shape != (len(m),) or not np.all(np.isfinite(c)):
            raise ValueError(f"{name} requires one finite coefficient per vector.")
        if dissipative and np.any(c < 0):
            raise ValueError("Bath strengths must be nonnegative.")
        if np.any(np.all(m == 0, axis=1)):
            raise ValueError(f"{name} vectors must be nonzero.")
        # lambda=W.T m and d=|lambda|^2/2:
        # Eq. (A25) [paper:appeq:current-scaling-dimension].
        weights = (m @ basis) ** 2
        dimensions = weights.sum(axis=1) / 2
        flux = np.array([np.linalg.solve(k, vector) for vector in m]).reshape(-1, n)
        if equal_velocities:
            # When all speeds equal u, W W.T=K^-1 U K^-1/u. This computes
            # the same scaling dimensions without a degenerate eigenbasis.
            dimensions = np.sum((flux @ u) * flux, axis=1) / (2 * velocities[0])
        # The signed norm is topological: s=m.T K^-1 m/2, independent of U,
        # Eq. (A46) [paper:appeq:zero-conformal-spin].
        spins = np.sum(m * flux, axis=1) / 2
        return VertexTerms(m, c, weights, dimensions, spins, flux)

    cosines = vertex_terms(cosine_vectors, cosine_coefficients, "Cosine")
    if not np.allclose(cosines.spins, np.rint(cosines.spins), rtol=0, atol=1e-10):
        raise ValueError(
            "Cosine Hamiltonian vertices must have integer conformal spin."
        )
    bath = vertex_terms(bath_vectors, bath_strengths, "Bath", dissipative=True)
    b0 = real_array(boundary_start, "boundary_start")
    bl = real_array(boundary_end, "boundary_end")
    b = real_array(boundary_values, "boundary_values")
    current = real_array(current_weights, "current_weights")
    if b0.shape != (n, n) or bl.shape != (n, n) or b.shape != (n,):
        raise ValueError(
            "Boundary matrices must have shape (n,n) and values shape (n,)."
        )
    if np.linalg.matrix_rank(np.column_stack((b0, bl))) != n:
        raise ValueError("Boundary constraints must be linearly independent.")
    if current.shape != (n,):
        raise ValueError("current_weights must have shape (n,).")
    for name, value in (
        ("cutoff_length", cutoff_length),
        ("reference_velocity", reference_velocity),
        ("finger_length", finger_length),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not callable(segment_length) and (
        not np.isfinite(segment_length) or segment_length <= 0
    ):
        raise ValueError("segment_length must be positive or a callable.")
    if not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer >= 2.")
    # Cached presets share these arrays; prevent accidental in-place edits from
    # making derived properties inconsistent with the defining matrices.
    for array in (
        k,
        u,
        basis,
        velocities,
        chiralities,
        b0,
        bl,
        b,
        current,
        *vars(cosines).values(),
        *vars(bath).values(),
    ):
        array.setflags(write=False)
    return TransportModel(
        k,
        u,
        basis,
        velocities,
        chiralities,
        cosines,
        bath,
        b0,
        bl,
        b,
        current,
        cutoff_length,
        reference_velocity,
        finger_length,
        segment_length,
        quadrature_order,
    )


def vertex_kernel(
    temperature,
    weights,
    d,
    spin,
    velocities,
    chiralities,
    segment,
    cutoff,
    reference_velocity,
    order,
):
    """Evaluate the coherent response for arbitrary Gaussian mode velocities.

    Eq. (A48) [paper:appeq:current_final_3] includes a velocity product;
    Eq. (A49) [paper:appeq:current-integral] supplies the space/time integral.
    Equal participating velocities use the analytic time reduction.
    """
    active = weights > 0
    speeds, powers, signs = velocities[active], weights[active], chiralities[active]
    if np.allclose(speeds, speeds[0], rtol=1e-12, atol=0):
        speed = float(np.mean(speeds))
        return (
            _kernel(temperature, d, abs(spin), segment, cutoff, speed, order)
            * (reference_velocity / speed) ** 2
        )
    shifts = signs * np.pi * KB_OVER_HBAR * temperature * segment / speeds

    def time_response(separation):
        """Integrate the positive correlator, resolving every shifted cosh centre."""
        offsets = shifts * separation
        # Translate all centres together to avoid missing a peak far from zero.
        offsets = offsets - (offsets.max() + offsets.min()) / 2

        def integrand(t):
            """Use log(cosh) so large thermal shifts never overflow."""
            x = t + offsets
            return np.exp(-powers @ (np.logaddexp(x, -x) - np.log(2.0)))

        points = np.r_[-np.inf, np.unique(-offsets), np.inf]
        return sum(
            quad(integrand, lo, hi, epsabs=1e-13, epsrel=1e-10)[0]
            for lo, hi in zip(points[:-1], points[1:])
        )

    if max(abs(shifts)) <= 4:
        nodes, spatial_weights = quadrature_rule(order)
        integral = sum(
            w * 2 * (1 - v) * time_response(v) for v, w in zip(nodes, spatial_weights)
        )
    else:
        integral = quad(
            lambda v: 2 * (1 - v) * time_response(v), 0, 1, epsabs=1e-12, epsrel=1e-9
        )[0]
    z = KB_OVER_HBAR * temperature * cutoff / reference_velocity
    velocity_factor = np.exp(powers @ np.log(reference_velocity / speeds))
    return (
        segment
        / cutoff
        / (4 * np.pi)
        * (np.pi * z) ** (2 * d - 2)
        * velocity_factor
        * integral
    )


def preset_model(filling, gc, gn, parameters, scenario):
    """Build a paper preset through build_model, caching temperature-independent data."""
    return _preset_model(
        filling,
        gc,
        gn,
        tuple(sorted(parameters.items())),
        tuple(sorted(scenario.items())),
    )


@lru_cache(maxsize=128)
def _preset_model(filling, gc, gn, parameter_items, scenario_items):
    """Specify physical K, U, vertices, bare couplings, and contacts for a preset."""
    p, scenario = dict(parameter_items), dict(scenario_items)
    a, u = p["cutoffLengthM"], p["modeVelocityMPerS"]
    if filling not in {"2over3", "4over3", "2over5"}:
        raise ValueError(f"Unknown filling: {filling}")
    is_jain = filling == "2over5"
    chi = -1 if filling == "2over3" else 1
    if gc <= 0 or (is_jain and (gn is None or gn <= 0)):
        raise ValueError("Luttinger parameters must be positive.")
    # Unit source bias; KFP contacts for 2/3 and physical incoming modes otherwise.
    # Eq. (A126) [paper:appeq:nu23-boundary-conditions] and
    # Eq. (A137) [paper:appeq:nu43-boundary-conditions].
    source = np.eye(4)[[0, 1] if is_jain else [0, 2]]
    rhs = np.ones(2)
    if filling == "2over3":
        source = np.array([[3, 0, -1, 0], [0, -1, 0, 1.0]])
        rhs = np.array([2.0, 0.0])
    tip = (
        np.array([[1, 0, -1, 0], [0, 1, 0, -1.0]])
        if is_jain
        else np.array([[1, -1, 0, 0], [0, 0, 1, -1.0]])
    )
    bottom = [2, 3] if is_jain else [1, 3]

    if is_jain:
        # Physical ordering (top 1, top 2, bottom 1, bottom 2).
        kedge = np.array([[3.0, 2.0], [2.0, 3.0]])
        k = block_diag(kedge, -kedge)
        # Charge/neutral rotation: K eigenvalues 5 and 1. Each top/bottom pair
        # has U=kappa*u/2*((g+1/g)I-(g-1/g)sigma_x),
        # Eq. (46) [paper:eq:Ui-parametrization].
        rotation = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
        g = np.array([gc, gn])
        diagonal = (
            rotation @ np.diag(np.array([5, 1]) * u * (g + 1 / g) / 2) @ rotation.T
        )
        cross = rotation @ np.diag(-np.array([5, 1]) * u * (g - 1 / g) / 2) @ rotation.T
        interaction = np.block([[diagonal, cross], [cross, diagonal]])
        coherent, bath = [], []
        for m in kedge:
            coherent += [
                (np.r_[m, m], scenario["fractionalTunnelingLengthM"]),
                (np.r_[m, -m], scenario["fractionalAndreevLengthM"]),
            ]
            bath += [
                (np.r_[m, 0.0, 0.0], p["fractionalDissipationLengthM"]),
                (np.r_[0.0, 0.0, -m], p["fractionalDissipationLengthM"]),
            ]
        # Four-electron inter-side neutral vertices. K_J r=r for r=(1,-1),
        # so these are local electron-bilinear transfers on each side.
        # Independent lengths allow either family to be omitted with infinity.
        root = np.array([1.0, -1.0])
        coherent += [
            (np.r_[root, root], scenario.get("neutralTunnelingLengthM", np.inf)),
            (np.r_[root, -root], scenario.get("neutralAndreevLengthM", np.inf)),
        ]

        def thermal_segment(temperature):
            """Clip the thermal coherence length to [a,L] for the Jain preset."""
            return min(
                p["fingerLengthM"], max(a, u / (np.pi * KB_OVER_HBAR * temperature))
            )

        segment = thermal_segment
    else:
        # Paired ordering (top I, bottom I, top f, bottom f):
        # Eq. (A105) [paper:appeq:integer-fractional-paired-basis].
        k = np.diag([1.0, -1.0, 3 * chi, -3 * chi])
        fractional_u = (
            3
            * u
            / 2
            * np.array([[gc + 1 / gc, -(gc - 1 / gc)], [-(gc - 1 / gc), gc + 1 / gc]])
        )
        interaction = block_diag(u * np.eye(2), fractional_u)
        coherent = [
            ([1, 1, 0, 0], scenario["integerTunnelingLengthM"]),
            ([1, -1, 0, 0], scenario["integerAndreevLengthM"]),
            ([0, 0, 3 * chi, 3 * chi], scenario["fractionalTunnelingLengthM"]),
            ([0, 0, 3 * chi, -3 * chi], scenario["fractionalAndreevLengthM"]),
        ]
        for vectors, key in [
            ([[1, 0, 0, 3 * chi], [0, 1, 3 * chi, 0]], "crossTunnelingLengthM"),
            ([[1, 0, 0, -3 * chi], [0, -1, 3 * chi, 0]], "crossAndreevLengthM"),
            (
                [[1, 0, -3 * chi, 0], [0, -1, 0, 3 * chi]],
                "equilibrationDominatedLengthM",
            ),
        ]:
            coherent += [(v, p[key]) for v in vectors]
        bath = [
            (
                np.eye(4)[i] * k[i, i],
                p["integerDissipationLengthM"]
                if i < 2
                else p["fractionalDissipationLengthM"],
            )
            for i in range(4)
        ]
        segment = p["coherentSegmentLengthM"]
    # Convert bare lengths to the dimensionless coefficients consumed by the
    # generic builder, Eq. (107) [paper:eq:numerical-bare-lengths].
    current = np.zeros(4)
    current[bottom] = -1
    return build_model(
        k,
        interaction,
        [v for v, _ in coherent],
        [np.sqrt(4 * np.pi * a / length) for _, length in coherent],
        boundary_start=np.vstack((source, np.zeros((2, 4)))),
        boundary_end=np.vstack((np.zeros((2, 4)), tip)),
        boundary_values=np.r_[rhs, 0.0, 0.0],
        current_weights=current,
        cutoff_length=a,
        reference_velocity=u,
        finger_length=p["fingerLengthM"],
        segment_length=segment,
        bath_vectors=[v for v, _ in bath],
        bath_strengths=[a / (2 * np.pi * length) for _, length in bath],
        quadrature_order=p.get("quadratureOrder", 96),
    )


def segment_lengths(length, segment):
    """Full coherence segments plus one shorter segment at the tip.

    Implements the segment discretization of Eq. (96) [paper:eq:diff].
    The last segment's kernel is evaluated at its actual length. No fractional
    matrix powers and no subdivision toward the continuum are used.
    """
    if not (
        np.isfinite(length) and np.isfinite(segment) and length > 0 and segment > 0
    ):
        raise ValueError("Finger and segment lengths must be finite and positive.")
    ratio = length / segment
    # Avoid a spurious tiny remainder when an integer ratio is rounded in float64.
    nearest = round(ratio)
    if nearest >= 1 and abs(ratio - nearest) <= 1e-12 * max(1.0, ratio):
        return np.full(nearest, length / nearest)
    count = int(np.floor(ratio))
    return np.r_[np.full(count, segment), length - count * segment]


def discrete_system(boundary_start, boundary_values, boundary_end, transfers):
    """Assemble the sparse boundary problem for all segment-end potentials.

    Unknowns are (mu_0, ..., mu_N), each with n components. The first n
    rows impose B_0 @ mu_0 + B_L @ mu_N = b using the supplied boundary
    matrices and values. Each subsequent n-row block imposes mu_(i+1)-P_i*mu_i=0.
    P_i=I+ell_i*A_i implements Eq. (A76) [paper:appeq:voltage-recurrence].
    """
    count, n = len(transfers), len(boundary_values)
    rows, cols, data = [], [], []

    def block(row, col, array):
        """Append a dense constraint block to the sparse matrix's triplet lists."""
        ii, jj = np.indices(array.shape)
        rows.extend((ii + row).ravel())
        cols.extend((jj + col).ravel())
        data.extend(array.ravel())

    block(0, 0, boundary_start)
    block(0, n * count, boundary_end)
    for i, transfer in enumerate(transfers):
        block(n * (i + 1), n * i, -transfer)
        block(n * (i + 1), n * (i + 1), np.eye(n))
    size = n * (count + 1)
    matrix = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
    values = np.zeros(size)
    values[:n] = boundary_values
    return matrix, values


def precision_transfer_solution(
    boundary_start,
    boundary_values,
    boundary_end,
    transfers,
    updates=None,
    preserve_precision=False,
):
    """Solve contact constraints using ordered transfer products at high precision.

    Generalizes the equal-segment power in
    Eq. (A101) [paper:appeq:discrete-voltage-solution] to unequal final segments.
    The product is P_(N-1)...P_0; B_0 + B_L@product gives an n-by-n
    system for mu_0, then mu_L=product@mu_0.

    A 60-digit guard remains after the sum of log10 transfer norm bounds.
    These are numerical precision choices;
    increasing precision does not improve the underlying perturbative approximation.
    Identical full segments use integer matrix powers, never exponentials.
    """
    digits = 60 + int(
        np.ceil(sum(np.log10(max(1.0, np.linalg.norm(t, np.inf))) for t in transfers))
    )
    n = len(boundary_values)
    with mp.workdps(digits):
        product = mp.eye(n)
        i = 0
        while i < len(transfers):
            j = i + 1
            while j < len(transfers) and np.array_equal(transfers[j], transfers[i]):
                j += 1
            # Form I+hA at high precision as well. Rounding its diagonal
            # additions in float64 can break exact symmetries and excite
            # enormous spurious lattice modes even if powers use 200 digits.
            local = (
                mp.eye(n) + mp.matrix(updates[i].tolist())
                if updates is not None
                else mp.matrix(transfers[i].tolist())
            )
            product = local ** (j - i) @ product
            i = j
        start_mp, end_mp = (
            mp.matrix(boundary_start.tolist()),
            mp.matrix(boundary_end.tolist()),
        )
        boundary = start_mp + end_mp @ product
        values = mp.matrix(boundary_values.tolist())
        v0 = mp.lu_solve(boundary, values)
        vl = product @ v0
        residual = max(abs(x) for x in boundary @ v0 - values)
        if residual > mp.mpf("1e-30"):
            raise ArithmeticError(
                "High-precision discrete boundary solve did not converge."
            )
        if preserve_precision:
            return v0, vl, float(residual)
        return (
            np.array(list(v0), dtype=float),
            np.array(list(vl), dtype=float),
            float(residual),
        )


def discrete_solution(
    boundary_start,
    boundary_values,
    boundary_end,
    transfers,
    updates=None,
    preserve_precision=False,
):
    """Return mu_0, mu_L, and residual from a stable discrete boundary solve.

    Large local updates trigger high-precision ordered products; otherwise a
    sparse solve treats all boundaries together. Both implement the same
    recurrence.
    """
    n = len(boundary_values)
    if max(np.linalg.norm(t - np.eye(n), np.inf) for t in transfers) > 0.5:
        return precision_transfer_solution(
            boundary_start,
            boundary_values,
            boundary_end,
            transfers,
            updates,
            preserve_precision,
        )
    matrix, values = discrete_system(
        boundary_start, boundary_values, boundary_end, transfers
    )
    voltages = spsolve(matrix, values)
    if not np.all(np.isfinite(voltages)):
        raise np.linalg.LinAlgError("Singular discrete boundary problem.")
    # Componentwise backward error includes the recurrence at every segment.
    scale = abs(matrix) @ abs(voltages) + abs(values)
    residual = np.max(abs(matrix @ voltages - values) / np.maximum(scale, 1.0))
    if residual > 1e-9:
        raise ArithmeticError(f"Discrete backward error {residual:g}")
    return voltages[:n], voltages[-n:], float(residual)


def evaluate(curve, temperature, parameters, scenario):
    """Evaluate a paper preset using the generic model and boundary solver."""
    result = preset_model(
        curve["filling"], curve["gCharge"], curve["gNeutral"], parameters, scenario
    ).evaluate(temperature)
    # The paper's normalized column is a figure convention, not model input.
    nu = {"2over3": 2 / 3, "4over3": 4 / 3, "2over5": 2 / 5}[curve["filling"]]
    return {"mathcal_G": result["G_over_GQ"] / nu, **result}


def filename(curve):
    """Encode filling, coupling scenario, and Luttinger parameters in a CSV name."""

    def token(x):
        """Encode a parameter as a filename token, for example 1.5 as 1p5."""
        return str(float(x)).replace(".0", ".").replace(".", "p")

    suffix = f"_gC_{token(curve['gCharge'])}"
    if curve["gNeutral"] is not None:
        suffix += f"_gN_{token(curve['gNeutral'])}"
    return f"conductance_nu_{curve['filling']}_{curve['scenario']}{suffix}.csv"


def write_csv(path, rows):
    """Write ordered numerical records with a header and consistent newlines."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


SEGMENT_OVER_CUTOFF = 1000.0
CASES = [(1.0, 0.0), (1.5, 0.0), (2.0, 1.0)]
SHADING_GUIDE_VERSION = 3
JAIN_SHADING_THRESHOLD_K = 2.0
TEMPERATURE_SAMPLING_VERSION = 1


def responses(x, d, spin):
    """Return exact, short-segment, and long-segment A_r for the appendix plot.

    x=ell_seg/ell_T; use a=u=1 and ell_seg/a=1000 as dimensionless reference
    scales, converting x back to temperature for the SI-aware kernel.
    The long limit requires d>|spin| (weight in both chiralities).
    """
    t = x / (np.pi * KB_OVER_HBAR * SEGMENT_OVER_CUTOFF)
    exact = _kernel(float(t), d, spin, SEGMENT_OVER_CUTOFF, 1.0, 1.0, 96)
    # Eq. (A53) [paper:appeq:current-short-segment].
    short = (
        SEGMENT_OVER_CUTOFF
        / (2 * np.pi)
        * (2 * x / SEGMENT_OVER_CUTOFF) ** (2 * d - 2)
        * beta(d, d)
    )
    # Eq. (A57) [paper:appeq:current-long-segment], with d_+=(d+s)/2, d_-=(d-s)/2.
    long = (
        (2 * x / SEGMENT_OVER_CUTOFF) ** (2 * d - 3)
        / (4 * np.pi)
        * beta((d + spin) / 2, (d + spin) / 2)
        * beta((d - spin) / 2, (d - spin) / 2)
    )
    return exact, short, long


def turning_point(config, data_dir, filling, gc, gn, scenario="sc_stronger"):
    """Locate the selected plotted minimum for the figure's shading guide."""
    curve = {
        "filling": filling,
        "scenario": scenario,
        "gCharge": gc,
        "gNeutral": gn,
    }
    with (data_dir / filename(curve)).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    temperatures = [float(row["temperature_K"]) for row in rows]
    conductances = [float(row["G_over_GQ"]) for row in rows]
    index = min(range(len(conductances)), key=conductances.__getitem__)
    if index in (0, len(conductances) - 1):
        raise ValueError(f"The selected {filling} curve has no interior minimum.")
    if filling == "2over5":
        # The Jain figure connects the saved commensurate samples. Its guide
        # must refer to their minimum, without refining between those samples.
        return temperatures[index]

    def response(temperature):
        """Evaluate the selected curve during bounded minimum refinement."""
        return evaluate(
            curve,
            temperature,
            config["parameters"],
            config["scenarioParameters"][scenario],
        )["mathcal_G"]

    result = minimize_scalar(
        response,
        bounds=(temperatures[index - 1], temperatures[index + 1]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise ArithmeticError("Turning-point refinement failed.")
    return float(result.x)


def generate_data(config, data_dir, points):
    """Write conductance CSVs, response CSV, and the metadata consumed by plotting.

    Use linear/logarithmic grids and refine extrema for fixed-length segments.
    For the Jain figure, retain the one-segment branch below T_L and sample
    only T_n=n*T_L above it, where the unclipped thermal length is L/n.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not leave metadata declaring a valid cache.
    (data_dir / "run_metadata.json").unlink(missing_ok=True)
    p = config["parameters"]
    tmin, tmax = p["temperatureMinimumK"], p["temperatureMaximumK"]
    base = np.unique(
        np.r_[np.geomspace(tmin, tmax, points), np.linspace(tmin, tmax, points)]
    )
    thermal_unit = p["modeVelocityMPerS"] / (
        np.pi * KB_OVER_HBAR * p["fingerLengthM"]
    )
    for curve in config["curves"]:
        temperatures = base
        if curve["filling"] == "2over5":
            # Sample the existing model at exact integer L/ell_T. Below T_L,
            # ell_seg=L already, so retain the resolved single-segment branch.
            # Above ell_T=a the existing cutoff still applies; do not change
            # a, L, or the finite-segment transport law to force divisibility.
            commensurate = thermal_unit * np.arange(
                1, int(np.floor(tmax / thermal_unit)) + 1
            )
            commensurate = commensurate[
                (commensurate >= tmin) & (commensurate <= tmax)
            ]
            temperatures = np.unique(
                np.r_[base[base < thermal_unit], commensurate]
            )
        scenario = config["scenarioParameters"][curve["scenario"]]
        sampled = {}

        def sample(t):
            """Memoize a transport solve and retain both conductance normalizations."""
            t = float(t)
            if t not in sampled:
                result = evaluate(curve, t, p, scenario)
                sampled[t] = {
                    "temperature_K": t,
                    "mathcal_G": result["mathcal_G"],
                    "G_over_GQ": result["G_over_GQ"],
                }
            return sampled[t]["G_over_GQ"]

        # Resolve extrema directly in G/G_Q; the logarithmic coordinate keeps
        # the bounded minimization well scaled across temperature decades.
        values = np.array([sample(t) for t in temperatures])
        for i in range(1, len(values) - 1):
            if curve["filling"] == "2over5" and temperatures[i + 1] > thermal_unit:
                # Refining a thermal-region extremum would insert off-grid
                # points and reintroduce the remainder-segment ripples.
                continue
            left, right = values[i] - values[i - 1], values[i + 1] - values[i]
            if left * right < 0 and max(abs(left), abs(right)) > 1e-9:
                sign = 1 if left < 0 else -1
                result = minimize_scalar(
                    lambda logt: sign * sample(np.exp(logt)),
                    bounds=np.log(temperatures[[i - 1, i + 1]]),
                    method="bounded",
                )
                sample(np.exp(result.x))
        name = filename(curve)
        write_csv(data_dir / name, [sampled[t] for t in sorted(sampled)])
        print(f"{name}: {len(sampled)} points", flush=True)
    integer_threshold = turning_point(config, data_dir, "4over3", 2.0, None)
    # The Jain panel uses a fixed visual low-temperature reference interval.
    # This shading is not inferred from a curve minimum or a thermal length.
    jain_threshold = JAIN_SHADING_THRESHOLD_K
    response_rows = []
    for d, spin in CASES:
        for x in np.geomspace(0.01, 100, 141):
            exact, short, long = responses(x, d, spin)
            response_rows.append(
                {
                    "d": d,
                    "spin": spin,
                    "x": float(x),
                    "exact": exact,
                    "short": short,
                    "long": long,
                }
            )
    write_csv(data_dir / "response_crossover.csv", response_rows)
    metadata = {
        "config": config,
        "points": points,
        "responseCases": CASES,
        "shadingGuideVersion": SHADING_GUIDE_VERSION,
        "temperatureSamplingVersion": TEMPERATURE_SAMPLING_VERSION,
        "temperatureSampling": {
            "2over5": {
                "rule": "single-segment branch below T_L; T_n=n*T_L at and above T_L",
                "thermalUnitK": thermal_unit,
                "thermalCutoffTemperatureK": p["modeVelocityMPerS"]
                / (np.pi * KB_OVER_HBAR * p["cutoffLengthM"]),
                "connectingLines": "guides between computed samples",
                "transportModelChanged": False,
            },
            "2over3": "adaptive linear/logarithmic grid; fixed coherence length",
            "4over3": "adaptive linear/logarithmic grid; fixed coherence length",
        },
        "shadingThresholdsK": {
            "2over3": integer_threshold,
            "4over3": integer_threshold,
            "2over5": jain_threshold,
        },
        "shadingGuideSources": {
            "2over3": "same turning-point reference as 4over3",
            "4over3": "minimum of the solid g_C=2 curve",
            "2over5": "fixed low-temperature visual reference",
        },
    }
    (data_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    """Generate data from the CLI, or reuse a complete cache for the same inputs."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=script_dir / "conductance_parameters.json"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=script_dir / "generated_data_finite"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=401,
        help="Points in each linear and logarithmic temperature grid.",
    )
    parser.add_argument(
        "--recompute-data", action="store_true", help="Recompute cached curves."
    )
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    config = json.loads(args.config.read_text())
    metadata_path = args.data_dir / "run_metadata.json"
    cached = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if (
        args.recompute_data
        or cached is None
        or cached.get("config") != config
        or cached.get("points") != args.points
        or cached.get("shadingGuideVersion") != SHADING_GUIDE_VERSION
        or cached.get("temperatureSamplingVersion") != TEMPERATURE_SAMPLING_VERSION
        or "shadingThresholdsK" not in cached
        or "responseCases" not in cached
        or not (args.data_dir / "response_crossover.csv").exists()
        or any(
            not (args.data_dir / filename(curve)).exists() for curve in config["curves"]
        )
    ):
        generate_data(config, args.data_dir, args.points)
    print(f"Data ready in {args.data_dir}")


if __name__ == "__main__":
    main()
