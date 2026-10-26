"""Independent tests of discrete transport, normalization, and continuum limit."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import beta, hyp2f1

import conductance_pipeline as pipeline

CONFIG = json.loads(
    Path(__file__).resolve().with_name("conductance_parameters.json").read_text()
)
P = CONFIG["parameters"]
KB_OVER_HBAR = 1.380649e-23 / 1.054571817e-34


NODES, WEIGHTS = np.polynomial.legendre.leggauss(96)
NODES, WEIGHTS = (NODES + 1) / 2, WEIGHTS / 2


def time_integral(qp, qm, shift):
    d = (qp + qm) / 2
    qp, qm = max(qp, qm), min(qp, qm)
    prefactor = 2 ** (2 * d - 1) * beta(d, d)
    if qm < 1e-12:
        return prefactor + np.zeros_like(shift)
    return (
        prefactor
        * np.exp(-2 * qm * shift)
        * hyp2f1(qm, d, 2 * d, -np.expm1(-4 * shift))
    )


def kernel(temperature, d, spin, segment, p):
    z = KB_OVER_HBAR * temperature * p["cutoffLengthM"] / p["modeVelocityMPerS"]
    length_ratio = segment / p["cutoffLengthM"]
    shift = np.pi * z * length_ratio
    if shift < 1e-4 or abs(d - abs(spin)) < 1e-12:
        return length_ratio / (2 * np.pi) * (2 * np.pi * z) ** (2 * d - 2) * beta(d, d)
    integral = np.sum(
        WEIGHTS * 2 * (1 - NODES) * time_integral(d + spin, d - spin, shift * NODES)
    )
    return length_ratio / (4 * np.pi) * (np.pi * z) ** (2 * d - 2) * integral


def model_and_generator(filling, gc, gn, temperature, p, scenario):
    a, u = p["cutoffLengthM"], p["modeVelocityMPerS"]
    is_jain = filling == "2over5"
    segment = (
        min(p["fingerLengthM"], max(a, u / (np.pi * KB_OVER_HBAR * temperature)))
        if is_jain
        else p["coherentSegmentLengthM"]
    )
    chi = -1 if filling == "2over3" else 1
    nu = {"2over3": 2 / 3, "4over3": 4 / 3, "2over5": 2 / 5}[filling]
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
        kedge = np.array([[3.0, 2.0], [2.0, 3.0]])
        k = np.block([[kedge, np.zeros((2, 2))], [np.zeros((2, 2)), -kedge]])
        dt, da = (5 * gc + gn) / 2, (5 / gc + 1 / gn) / 2
        coherent = []
        bath = []
        for m in kedge:
            coherent += [
                (np.r_[m, m], dt, 0, scenario["fractionalTunnelingLengthM"]),
                (np.r_[m, -m], da, 0, scenario["fractionalAndreevLengthM"]),
            ]
            bath += [
                (np.r_[m, 0.0, 0.0], (dt + da) / 4, p["fractionalDissipationLengthM"]),
                (np.r_[0.0, 0.0, -m], (dt + da) / 4, p["fractionalDissipationLengthM"]),
            ]
        coherent += [
            ([1, -1, 1, -1], 2 * gn, 0, scenario.get("neutralTunnelingLengthM", np.inf)),
            ([1, -1, -1, 1], 2 / gn, 0, scenario.get("neutralAndreevLengthM", np.inf)),
        ]
    else:
        k = np.diag([1.0, -1.0, 3 * chi, -3 * chi])
        df = 3 * (gc + 1 / gc) / 4
        coherent = [
            ([1, 1, 0, 0], 1, 0, scenario["integerTunnelingLengthM"]),
            ([1, -1, 0, 0], 1, 0, scenario["integerAndreevLengthM"]),
            (
                [0, 0, 3 * chi, 3 * chi],
                3 * gc,
                0,
                scenario["fractionalTunnelingLengthM"],
            ),
            (
                [0, 0, 3 * chi, -3 * chi],
                3 / gc,
                0,
                scenario["fractionalAndreevLengthM"],
            ),
        ]
        for vectors, key, spin in [
            (
                [[1, 0, 0, 3 * chi], [0, 1, 3 * chi, 0]],
                "crossTunnelingLengthM",
                (1 - 3 * chi) / 2,
            ),
            (
                [[1, 0, 0, -3 * chi], [0, -1, 3 * chi, 0]],
                "crossAndreevLengthM",
                (1 - 3 * chi) / 2,
            ),
            (
                [[1, 0, -3 * chi, 0], [0, -1, 0, 3 * chi]],
                "equilibrationDominatedLengthM",
                (1 + 3 * chi) / 2,
            ),
        ]:
            coherent += [(v, 0.5 + df, spin, p[key]) for v in vectors]
        bath = [
            (
                np.eye(4)[i] * k[i, i],
                0.5 if i < 2 else df,
                p["integerDissipationLengthM"]
                if i < 2
                else p["fractionalDissipationLengthM"],
            )
            for i in range(4)
        ]
    generator = np.zeros((4, 4))
    for m, d, spin, length in coherent:
        # H_r = u y_r cos(m.phi)/(pi a^2), ell_r = 4 pi a/|y_r|^2.
        y_squared = 4 * np.pi * a / length
        rate = 2 * np.pi / a * kernel(temperature, d, spin, segment, p) * y_squared
        generator -= rate * np.outer(m, np.linalg.solve(k, m))
    z = KB_OVER_HBAR * temperature * a / u
    for m, d, length in bath:
        alpha = a / (2 * np.pi * length)
        bath_kernel = (2 * np.pi * z) ** (2 * d - 1) * beta(d + 0.5, d + 0.5)
        rate = 2 * np.pi / a * bath_kernel * alpha
        generator -= rate * np.outer(m, np.linalg.solve(k, m))
    return k, source, rhs, tip, bottom, nu, generator, segment


def direct_time_integral(qp, qm, shift):
    def logcosh(x):
        return np.logaddexp(x, -x) - np.log(2)

    return quad(
        lambda t: np.exp(-qp * logcosh(t + shift) - qm * logcosh(t - shift)),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-11,
    )[0]


def continuum_solution(k, source, rhs, tip, generator, length):
    """Stable scattering composition of exp(L A), retaining all four modes."""
    order = np.r_[np.flatnonzero(np.diag(k) > 0), np.flatnonzero(np.diag(k) < 0)]
    exponent = generator[np.ix_(order, order)] * length
    steps = max(0, int(np.ceil(np.log2(max(1.0, np.linalg.norm(exponent, np.inf))))))
    transfer = expm(exponent / 2.0**steps)
    eye, zero = np.eye(2), np.zeros((2, 2))
    dd = np.linalg.solve(transfer[2:, 2:], eye)
    cc = -dd @ transfer[2:, :2]
    bb = transfer[:2, 2:] @ dd
    aa = transfer[:2, :2] + transfer[:2, 2:] @ cc
    for _ in range(steps):
        xaa = np.linalg.solve(eye - bb @ cc, aa)
        xbd = np.linalg.solve(eye - bb @ cc, bb @ dd)
        aa, bb, cc, dd = (
            aa @ xaa,
            bb + aa @ xbd,
            cc + dd @ cc @ xaa,
            dd @ dd + dd @ cc @ xbd,
        )
    start = np.block([[eye, zero], [cc, dd]])[np.argsort(order)]
    end = np.block([[aa, bb], [zero, eye]])[np.argsort(order)]
    incoming = np.linalg.solve(
        np.vstack([source @ start, tip @ end]), np.r_[rhs, 0.0, 0.0]
    )
    v0, vl = start @ incoming, end @ incoming
    residual = max(np.max(abs(source @ v0 - rhs)), np.max(abs(tip @ vl)))
    return v0, vl, float(residual)


def preset_and_generator(filling, gc, gn, temperature, p, scenario, segment=None):
    """Expose preset data in the convention used by the independent references."""
    model = pipeline.preset_model(filling, gc, gn, p, scenario)
    segment = model.coherence_length(temperature) if segment is None else segment
    return (
        model.k_matrix,
        model.boundary_start[:2],
        model.boundary_values[:2],
        model.boundary_end[2:],
        np.flatnonzero(model.current_weights).tolist(),
        {"2over3": 2 / 3, "4over3": 4 / 3, "2over5": 2 / 5}[filling],
        model.generator(temperature, segment),
        segment,
    )


class ConductanceTests(unittest.TestCase):
    def test_preset_conductance_regression(self):
        """Preserve stiff-case conductances recorded before generic model building."""
        references = [
            ("2over5", "sc_stronger", 1.5, -0.022875978054709797),
            ("2over5", "sc_stronger", 3.0, -0.013800728473805594),
            ("2over5", "tunneling_stronger", 1.5, 0.00886708924096023),
            ("2over5", "tunneling_stronger", 3.0, 0.030092263546753085),
            ("2over3", "sc_stronger", 1.5, 0.004740331001436116),
            ("2over3", "sc_stronger", 3.0, 0.0012001710064121038),
            ("2over3", "tunneling_stronger", 1.5, 0.021532303064090916),
            ("2over3", "tunneling_stronger", 3.0, 0.02168604681247652),
            ("4over3", "sc_stronger", 1.5, -0.002799546394942768),
            ("4over3", "sc_stronger", 3.0, -0.006671050575017615),
            ("4over3", "tunneling_stronger", 1.5, 0.047045960857590864),
            ("4over3", "tunneling_stronger", 3.0, 0.06037560472735624),
        ]
        for filling, scenario, gc, expected in references:
            solid = scenario == "sc_stronger"
            model = pipeline.preset_model(
                filling,
                gc,
                gc if filling == "2over5" else None,
                dict(
                    P,
                    fractionalDissipationLengthM=2e-8,
                    equilibrationDominatedLengthM=2e-8,
                    crossTunnelingLengthM=2.2e-6,
                    crossAndreevLengthM=1e-4,
                ),
                dict(
                    CONFIG["scenarioParameters"][scenario],
                    integerTunnelingLengthM=5.7e-7 if solid else 4.5e-7,
                    integerAndreevLengthM=5.5e-7,
                    fractionalTunnelingLengthM=1.1e-6 if solid else 4.5e-7,
                    fractionalAndreevLengthM=5.5e-7,
                    neutralTunnelingLengthM=np.inf,
                    neutralAndreevLengthM=np.inf,
                ),
            )
            self.assertAlmostEqual(
                model.evaluate(10.0)["G_over_GQ"], expected, delta=1e-8
            )

    def test_thermal_integral(self):
        for qp, qm in [(1, 1), (3, 1), (1, 3), (4, 0), (0, 4), (9, 9)]:
            for shift in [0.0, 0.01, 1.0, 3.3, 9.0]:
                actual = pipeline.time_integral(qp, qm, shift)
                expected = direct_time_integral(qp, qm, shift)
                self.assertAlmostEqual(actual, expected, delta=2e-11)

    def test_rates_against_independent_implementation(self):
        for curve in CONFIG["curves"]:
            for t in [1e-6, 0.2, 10.0]:
                args = (
                    curve["filling"],
                    curve["gCharge"],
                    curve["gNeutral"],
                    t,
                    P,
                    CONFIG["scenarioParameters"][curve["scenario"]],
                )
                np.testing.assert_allclose(
                    preset_and_generator(*args)[6],
                    model_and_generator(*args)[6],
                    rtol=2e-8,
                    atol=1e-8,
                )

    def test_length_partition(self):
        self.assertEqual(len(pipeline.segment_lengths(1e-6, 20e-9)), 50)
        np.testing.assert_allclose(
            pipeline.segment_lengths(1.0, 0.3), [0.3, 0.3, 0.3, 0.1]
        )
        np.testing.assert_allclose(pipeline.segment_lengths(1.0, 2.0), [1.0])
        for value in [0.0, -1.0, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                pipeline.segment_lengths(1.0, value)

    def test_zero_rates(self):
        for curve in CONFIG["curves"]:
            p = P.copy()
            s = CONFIG["scenarioParameters"][curve["scenario"]].copy()
            for mapping in [p, s]:
                for key in mapping:
                    if key.endswith("LengthM") and key not in [
                        "fingerLengthM",
                        "coherentSegmentLengthM",
                        "cutoffLengthM",
                    ]:
                        mapping[key] = np.inf
            self.assertAlmostEqual(
                pipeline.evaluate(curve, 0.2, p, s)["mathcal_G"], 1.0, places=12
            )
            k, source, rhs, tip, bottom, nu, a, _ = model_and_generator(
                curve["filling"], curve["gCharge"], curve["gNeutral"], 0.2, p, s
            )
            v0 = continuum_solution(k, source, rhs, tip, a, p["fingerLengthM"])[0]
            self.assertAlmostEqual(
                -sum(np.linalg.solve(k, v0)[bottom]) / nu, 1.0, places=12
            )

    def test_pure_bath_exact_discrete_answer(self):
        source = np.eye(4)[[0, 2]]
        rhs = np.ones(2)
        tip = np.array([[1, -1, 0, 0], [0, 0, 1, -1.0]])
        h, rate, n = 0.1, 2.0, 7
        transfer = np.diag([1 - h * rate, 1 + h * rate, 1, 1])
        v0, vl, res = pipeline.discrete_solution(
            np.vstack((source, np.zeros((2, 4)))),
            np.r_[rhs, 0.0, 0.0],
            np.vstack((np.zeros((2, 4)), tip)),
            [transfer] * n,
        )
        self.assertAlmostEqual(v0[1], ((1 - h * rate) / (1 + h * rate)) ** n, places=13)
        self.assertAlmostEqual(v0[3], 1.0, places=13)
        self.assertLess(res, 1e-14)

    def test_continuum_convergence_fixed_rates(self):
        curve = CONFIG["curves"][8]
        k, source, rhs, tip, bottom, nu, a, _ = preset_and_generator(
            curve["filling"],
            curve["gCharge"],
            curve["gNeutral"],
            0.2,
            P,
            CONFIG["scenarioParameters"][curve["scenario"]],
        )
        a = a / np.linalg.norm(a, np.inf)
        reference = continuum_solution(k, source, rhs, tip, a, 1.0)[0]
        errors = []
        for n in [10, 20, 40, 80]:
            actual = pipeline.discrete_solution(
                np.vstack((source, np.zeros((2, 4)))),
                np.r_[rhs, 0.0, 0.0],
                np.vstack((np.zeros((2, 4)), tip)),
                [np.eye(4) + a / n] * n,
            )[0]
            errors.append(np.linalg.norm(actual - reference, np.inf))
        self.assertTrue(all(b < a * 0.6 for a, b in zip(errors, errors[1:])), errors)

    def test_high_precision_transfer_reference(self):
        # Independent ordered transfer products at high precision, not a second
        # implementation of the sparse boundary solver. Includes stiff cases,
        # a single segment, noncommuting remainder, KFP contacts and both signs.
        maximum = 0.0
        # Stronger couplings can grow the reference transfer product beyond 240 digits.
        with mp.workdps(1000):
            for curve in CONFIG["curves"]:
                for t in [1e-6, 0.2, 5.0, 10.0]:
                    scenario = CONFIG["scenarioParameters"][curve["scenario"]]
                    info = preset_and_generator(
                        curve["filling"],
                        curve["gCharge"],
                        curve["gNeutral"],
                        t,
                        P,
                        scenario,
                    )
                    k, source, rhs, tip, bottom, nu, a, segment = info
                    lengths = pipeline.segment_lengths(P["fingerLengthM"], segment)
                    full = mp.eye(4) + mp.matrix((lengths[0] * a).tolist())
                    count = len(lengths)
                    remainder = not np.isclose(
                        lengths[-1], segment, rtol=1e-12, atol=0.0
                    )
                    transfer = full ** (count - int(remainder))
                    if remainder:
                        local = preset_and_generator(
                            curve["filling"],
                            curve["gCharge"],
                            curve["gNeutral"],
                            t,
                            P,
                            scenario,
                            segment=lengths[-1],
                        )[6]
                        transfer = (
                            mp.eye(4) + mp.matrix((lengths[-1] * local).tolist())
                        ) * transfer
                    boundary = mp.matrix(4, 4)
                    last = mp.matrix(tip.tolist()) * transfer
                    for i in range(2):
                        for j in range(4):
                            boundary[i, j] = source[i, j]
                            boundary[i + 2, j] = last[i, j]
                    v0 = mp.lu_solve(boundary, mp.matrix([*rhs, 0.0, 0.0]))
                    charge = mp.lu_solve(mp.matrix(k.tolist()), v0)
                    reference = float(-sum(charge[i] for i in bottom) / nu)
                    actual = pipeline.evaluate(curve, t, P, scenario)["mathcal_G"]
                    error = abs(actual - reference)
                    maximum = max(maximum, error)
                    self.assertLess(
                        error, 1e-8 * max(1, abs(reference)), (curve, t, error)
                    )
        print(
            f"High-precision discrete transfer: 96 checks, maximum absolute difference {maximum:.6g}"
        )

    def test_response_limits(self):
        """Independent double quadrature, plus short/long and chiral limits."""
        for d, spin in pipeline.CASES:
            for x in (0.03, 1.0, 30.0):

                def spatial(v):
                    shift = x * v

                    def temporal(t):
                        def lc(z):
                            return np.logaddexp(z, -z) - np.log(2.0)

                        return np.exp(
                            -(d + spin) * lc(t + shift) - (d - spin) * lc(t - shift)
                        )

                    return (
                        2
                        * (1 - v)
                        * quad(temporal, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-10)[0]
                    )

                integral = quad(spatial, 0, 1, epsabs=1e-11, epsrel=1e-9)[0]
                independent = (
                    pipeline.SEGMENT_OVER_CUTOFF
                    / (4 * np.pi)
                    * (x / pipeline.SEGMENT_OVER_CUTOFF) ** (2 * d - 2)
                    * integral
                )
                exact = pipeline.responses(x, d, spin)[0]
                error = abs(exact / independent - 1)
                self.assertLess(error, 1e-7, (d, spin, x, error))
            exact, short, _ = pipeline.responses(0.001, d, spin)
            self.assertLess(abs(exact / short - 1), 1e-6)
            exact, _, long = pipeline.responses(1000.0, d, spin)
            self.assertLess(abs(exact / long - 1), 0.002)
        for x in (0.01, 1.0, 100.0):
            t = x / (np.pi * pipeline.KB_OVER_HBAR * pipeline.SEGMENT_OVER_CUTOFF)
            exact = pipeline._kernel(
                t, 1.0, 1.0, pipeline.SEGMENT_OVER_CUTOFF, 1.0, 1.0, 96
            )
            self.assertTrue(
                np.isclose(exact, pipeline.SEGMENT_OVER_CUTOFF / (2 * np.pi))
            )


class GenericModelTests(unittest.TestCase):
    @staticmethod
    def inputs(n=2):
        """A small custom model with explicit mixed endpoint constraints."""
        return dict(
            k_matrix=np.diag([1 if i % 2 == 0 else -1 for i in range(n)]),
            u_matrix=np.eye(n) * 25000.0,
            cosine_vectors=[],
            cosine_coefficients=[],
            boundary_start=np.eye(n),
            boundary_end=np.eye(n) * 0.2,
            boundary_values=np.arange(1.0, n + 1),
            current_weights=np.ones(n),
            cutoff_length=7e-9,
            reference_velocity=25000.0,
            finger_length=1e-6,
            segment_length=3e-7,
            quadrature_order=32,
        )

    def test_dimensions_and_spins_from_interactions(self):
        """Recover analytic Laughlin dimensions and signed spins without supplying them."""
        g, u = 1.7, 25000.0
        inputs = self.inputs()
        inputs.update(
            k_matrix=np.diag([3.0, -3.0]),
            u_matrix=3
            * u
            / 2
            * np.array([[g + 1 / g, -(g - 1 / g)], [-(g - 1 / g), g + 1 / g]]),
            cosine_vectors=[[3, 3], [3, -3]],
            cosine_coefficients=[0.01, 0.02j],
            bath_vectors=[[3, 0], [0, -3]],
            bath_strengths=[0.001, 0.002],
        )
        model = pipeline.build_model(**inputs)
        np.testing.assert_allclose(model.cosines.dimensions, [3 * g, 3 / g], rtol=1e-13)
        np.testing.assert_allclose(model.cosines.spins, [0, 0], atol=1e-14)
        np.testing.assert_allclose(
            model.bath.dimensions, [3 * (g + 1 / g) / 4] * 2, rtol=1e-13
        )
        np.testing.assert_allclose(model.bath.spins, [1.5, -1.5], rtol=1e-13)
        np.testing.assert_allclose(model.velocities, u, rtol=1e-13)
        np.testing.assert_allclose(
            model.basis.T @ model.k_matrix @ model.basis,
            np.diag(model.chiralities),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            model.basis.T @ model.u_matrix @ model.basis,
            np.diag(model.velocities),
            atol=1e-10,
        )

    def test_unequal_velocities_against_shift_identity(self):
        """Two unequal-speed cosh factors reduce by a common time translation.

        Includes co-propagating vertices: their spin equals d but their spatial
        response is not constant when the velocities differ.
        """
        speeds = np.array([13000.0, 41000.0])
        for signs in ([1, -1], [1, 1]):
            inputs = self.inputs()
            inputs.update(
                k_matrix=np.diag(signs),
                u_matrix=np.diag(speeds),
                cosine_vectors=[[1, 1]],
                cosine_coefficients=[0.03j],
            )
            model = pipeline.build_model(**inputs)
            for temperature, length in ((0.02, 1e-7), (0.2, 1e-6)):
                shift = (
                    np.pi
                    * KB_OVER_HBAR
                    * temperature
                    * length
                    * abs(signs[0] / speeds[0] - signs[1] / speeds[1])
                    / 2
                )
                # Independently integrate the translated time correlator.
                integral = quad(
                    lambda v: 2 * (1 - v) * direct_time_integral(1, 1, shift * v),
                    0,
                    1,
                    epsabs=1e-11,
                    epsrel=1e-10,
                )[0]
                response = (
                    length
                    / inputs["cutoff_length"]
                    / (4 * np.pi)
                    * inputs["reference_velocity"] ** 2
                    / np.prod(speeds)
                    * integral
                )
                expected = (
                    -2
                    * np.pi
                    / inputs["cutoff_length"]
                    * 0.03**2
                    * response
                    * np.outer([1, 1], signs)
                )
                np.testing.assert_allclose(
                    model.generator(temperature, length), expected, rtol=2e-9
                )

    def test_basis_covariance_and_reference_normalization(self):
        """Changing field basis or the coupling reference speed preserves transport."""
        inputs = self.inputs(3)
        inputs.update(
            u_matrix=np.diag([13000.0, 25000.0, 41000.0]),
            cosine_vectors=np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
            cosine_coefficients=np.array([0.01 + 0.02j, -0.03]),
            bath_vectors=np.eye(3),
            bath_strengths=np.array([0.001, 0.002, 0.003]),
        )
        base = pipeline.build_model(**inputs)
        # A unimodular change preserves the physical integer K lattice.
        transform = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        inverse = np.linalg.inv(transform)
        changed = dict(inputs)
        changed.update(
            k_matrix=inverse.T @ inputs["k_matrix"] @ inverse,
            u_matrix=inverse.T @ inputs["u_matrix"] @ inverse,
            cosine_vectors=inputs["cosine_vectors"] @ inverse,
            bath_vectors=inputs["bath_vectors"] @ inverse,
            boundary_start=inputs["boundary_start"] @ transform.T,
            boundary_end=inputs["boundary_end"] @ transform.T,
            current_weights=inverse.T @ inputs["current_weights"],
        )
        rotated = pipeline.build_model(**changed)
        np.testing.assert_allclose(
            rotated.cosines.dimensions, base.cosines.dimensions, rtol=1e-12
        )
        np.testing.assert_allclose(
            rotated.cosines.spins, base.cosines.spins, atol=1e-13
        )
        np.testing.assert_allclose(
            rotated.generator(0.2),
            inverse.T @ base.generator(0.2) @ transform.T,
            rtol=1e-10,
            atol=1e-8,
        )
        self.assertAlmostEqual(
            rotated.evaluate(0.2)["G_over_GQ"],
            base.evaluate(0.2)["G_over_GQ"],
            places=10,
        )
        renormalized = dict(
            inputs,
            reference_velocity=50000.0,
            cosine_coefficients=inputs["cosine_coefficients"] / 2,
            bath_strengths=inputs["bath_strengths"] / 2,
        )
        np.testing.assert_allclose(
            pipeline.build_model(**renormalized).generator(0.2),
            base.generator(0.2),
            rtol=1e-12,
            atol=1e-8,
        )

    def test_arbitrary_mode_counts_and_coupled_boundaries(self):
        """Check sparse and high-precision paths against a diagonal exact recurrence."""
        for n in (1, 2, 6):
            for strength in (0.2, 4.0):
                inputs = self.inputs(n)
                lengths = np.array([0.3, 0.3, 0.15])
                rates = strength * np.linspace(0.7, 1.3, n)
                inputs.update(
                    finger_length=0.75,
                    segment_length=lambda t: 0.3,
                    cutoff_length=0.01,
                    bath_vectors=np.eye(n),
                    bath_strengths=rates * 0.01 / (2 * np.pi),
                    boundary_end=0.2 * np.eye(n) + 0.03 * np.ones((n, n)),
                )
                model = pipeline.build_model(**inputs)
                signs = np.diag(inputs["k_matrix"])
                transfer = np.diag(
                    np.prod(1 - lengths[:, None] * rates * signs, axis=0)
                )
                expected = np.linalg.solve(
                    inputs["boundary_start"] + inputs["boundary_end"] @ transfer,
                    inputs["boundary_values"],
                )
                v0, vl, residual = model.solve(0.2)
                np.testing.assert_allclose(
                    np.array(list(v0), dtype=float), expected, atol=1e-12
                )
                np.testing.assert_allclose(
                    np.array(list(vl), dtype=float), transfer @ expected, atol=1e-12
                )
                self.assertLess(residual, 1e-12)
                result = model.evaluate(0.2)
                self.assertAlmostEqual(
                    result["G_over_GQ"], sum(signs * expected), places=11
                )

    def test_dense_u_with_fixed_physical_k(self):
        """Mix every physical mode and independently diagonalize the Gaussian theory."""
        inputs = self.inputs(4)
        inputs.update(
            k_matrix=np.diag([1.0, -1.0, 3.0, -3.0]),
            u_matrix=25000
            * np.array(
                [
                    [1.3, 0.15, 0.12, -0.1],
                    [0.15, 1.1, -0.18, 0.2],
                    [0.12, -0.18, 3.8, 0.7],
                    [-0.1, 0.2, 0.7, 3.4],
                ]
            ),
            cosine_vectors=np.array([[1, 1, 0, 0], [0, 0, 3, -3], [1, 0, 0, 3.0]]),
            cosine_coefficients=[0.01, 0.02j, 0.03],
            bath_vectors=np.diag([1.0, -1.0, 3.0, -3.0]),
            bath_strengths=[0.001] * 4,
            boundary_start=np.array(
                [[1.0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            ),
            boundary_end=np.array(
                [[0.0, 0, 0, 0], [0, 0, 0, 0], [1, -1, 0, 0], [0, 0, 1, -1]]
            ),
            boundary_values=[1.0, 1.0, 0.0, 0.0],
            current_weights=[0.0, -1.0, 0.0, -1.0],
        )
        model = pipeline.build_model(**inputs)
        # Independent whitening uses ordinary symmetric eigensystems instead
        # of the builder's generalized K v=lambda U v problem.
        eigenvalues, eigenvectors = np.linalg.eigh(inputs["u_matrix"])
        root_u = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        signed_speeds, modes = np.linalg.eigh(
            root_u @ np.linalg.solve(inputs["k_matrix"], root_u)
        )
        reference_basis = np.linalg.solve(root_u, modes) * np.sqrt(abs(signed_speeds))
        normal_vertices = inputs["cosine_vectors"] @ reference_basis
        dimensions = np.sum(normal_vertices**2, axis=1) / 2
        spins = normal_vertices**2 @ np.sign(signed_speeds) / 2
        np.testing.assert_allclose(model.cosines.dimensions, dimensions, rtol=1e-12)
        np.testing.assert_allclose(model.cosines.spins, spins, atol=1e-13)
        np.testing.assert_allclose(
            np.sort(model.velocities), np.sort(abs(signed_speeds)), rtol=1e-12
        )
        # Express the full transport problem in the independently found normal
        # basis. Vertex, contact, and measured-current transformations must all
        # reproduce the physical-basis generator and conductance.
        normal = dict(inputs)
        normal.update(
            k_matrix=np.diag(np.sign(signed_speeds)),
            u_matrix=np.diag(abs(signed_speeds)),
            cosine_vectors=normal_vertices,
            bath_vectors=inputs["bath_vectors"] @ reference_basis,
            boundary_start=inputs["boundary_start"] @ np.linalg.inv(reference_basis.T),
            boundary_end=inputs["boundary_end"] @ np.linalg.inv(reference_basis.T),
            current_weights=reference_basis.T @ inputs["current_weights"],
        )
        reference = pipeline.build_model(**normal)
        transformed_generator = (
            reference_basis.T @ model.generator(0.2) @ np.linalg.inv(reference_basis.T)
        )
        np.testing.assert_allclose(
            reference.generator(0.2), transformed_generator, rtol=1e-10, atol=1e-8
        )
        self.assertAlmostEqual(
            model.evaluate(0.2)["G_over_GQ"],
            reference.evaluate(0.2)["G_over_GQ"],
            places=10,
        )

    def test_jain_distinct_charge_and_neutral_interactions(self):
        """The Jain preset builds U correctly even when g_C differs from g_N."""
        gc, gn = 1.3, 0.7
        model = pipeline.preset_model(
            "2over5", gc, gn, P, CONFIG["scenarioParameters"]["sc_stronger"]
        )
        dt, da = (5 * gc + gn) / 2, (5 / gc + 1 / gn) / 2
        np.testing.assert_allclose(
            model.cosines.dimensions, [dt, da, dt, da, 2 * gn, 2 / gn], rtol=1e-13
        )
        np.testing.assert_allclose(
            model.bath.dimensions, [(dt + da) / 4] * 4, rtol=1e-13
        )
        np.testing.assert_allclose(model.bath.spins, [1.5, -1.5, 1.5, -1.5], rtol=1e-13)

    def test_partial_degeneracy(self):
        """Repeated U eigenvalues and repeated generalized modes both remain valid."""
        inputs = self.inputs(4)
        inputs.update(
            k_matrix=np.diag([1.0, 1.0, -1.0, -1.0]),
            cosine_vectors=[
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            cosine_coefficients=[0.005, 0.004, 0.003],
            bath_vectors=np.eye(4),
            bath_strengths=[0.0001] * 4,
        )
        speeds = np.array([2.0, 2.0, 3.0, 5.0]) * 25000
        mixing = np.array(
            [
                [0.0, 0.0, 0.2, 0.1],
                [0.0, 0.0, -0.15, 0.25],
                [0.2, -0.15, 0.0, 0.0],
                [0.1, 0.25, 0.0, 0.0],
            ]
        )
        # A dense orthogonal rotation retains the repeated ordinary U eigenvalue.
        rotation = expm(np.triu(mixing) - np.triu(mixing).T)
        inputs["u_matrix"] = rotation @ np.diag(speeds) @ rotation.T
        model = pipeline.build_model(**inputs)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(model.u_matrix), speeds, rtol=1e-13
        )
        for matrix, diagonal in (
            (model.k_matrix, model.chiralities),
            (model.u_matrix, model.velocities),
        ):
            np.testing.assert_allclose(
                model.basis.T @ matrix @ model.basis,
                np.diag(diagonal),
                rtol=1e-12,
                atol=1e-9,
            )
        self.assertTrue(np.isfinite(model.evaluate(0.2)["G_over_GQ"]))

        # A K-preserving boost instead retains two equal signed mode velocities,
        # while producing dense U with generally distinct ordinary eigenvalues.
        inverse = expm(-mixing)
        inputs["u_matrix"] = inverse.T @ np.diag(speeds) @ inverse
        base = pipeline.build_model(**inputs)
        eigenvalues, eigenvectors = pipeline.eigh(
            inputs["k_matrix"], inputs["u_matrix"]
        )
        repeated = np.flatnonzero(
            np.isclose(eigenvalues, 1 / speeds[0], rtol=1e-12, atol=0)
        )
        self.assertEqual(len(repeated), 2)
        angle = 0.63
        rotated_vectors = eigenvectors.copy()
        rotated_vectors[:, repeated] = eigenvectors[:, repeated] @ np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        with patch.object(
            pipeline, "eigh", return_value=(eigenvalues, rotated_vectors)
        ):
            rotated = pipeline.build_model(**inputs)
        self.assertFalse(np.allclose(base.cosines.weights, rotated.cosines.weights))
        np.testing.assert_allclose(
            base.cosines.dimensions, rotated.cosines.dimensions, rtol=1e-13
        )
        np.testing.assert_allclose(
            base.cosines.spins, rotated.cosines.spins, atol=1e-13
        )
        np.testing.assert_allclose(
            base.generator(0.2), rotated.generator(0.2), rtol=1e-12, atol=1e-9
        )
        self.assertAlmostEqual(
            base.evaluate(0.2)["G_over_GQ"],
            rotated.evaluate(0.2)["G_over_GQ"],
            places=11,
        )

    def test_independent_bath_velocity(self):
        """At fixed microscopic w, doubling v_b halves the dissipative rate."""
        inputs = self.inputs()
        amplitudes = np.array([300.0, 400.0])
        bath_velocities = np.array([8000.0, 12000.0])
        alpha = abs(amplitudes) ** 2 / (
            2 * np.pi * inputs["reference_velocity"] * bath_velocities
        )
        inputs.update(bath_vectors=np.eye(2), bath_strengths=alpha)
        model = pipeline.build_model(**inputs)
        faster_bath = pipeline.build_model(**(inputs | {"bath_strengths": alpha / 2}))
        np.testing.assert_allclose(
            faster_bath.generator(0.2), model.generator(0.2) / 2, rtol=1e-14
        )

    def test_invalid_inputs(self):
        """Reject inconsistent physics inputs before diagonalization or transport."""
        for overrides in (
            dict(k_matrix=[[1, 0], [0, 0]]),
            dict(k_matrix=[[1, 0.5], [0.5, -1]]),
            dict(k_matrix=[[1, 1], [0, -1]]),
            dict(u_matrix=[[1, 0], [0, -1]]),
            dict(u_matrix=np.eye(3)),
            dict(cosine_vectors=[[1, 0]], cosine_coefficients=[]),
            dict(cosine_vectors=[[0, 0]], cosine_coefficients=[1]),
            dict(bath_vectors=[[1, 0]], bath_strengths=[-1]),
            dict(bath_vectors=[[1, 0]], bath_strengths=[1j]),
            dict(cosine_vectors=[[1, 0]], cosine_coefficients=[1]),
            dict(boundary_values=[1]),
            dict(boundary_start=np.zeros((2, 2)), boundary_end=np.zeros((2, 2))),
            dict(current_weights=[1]),
            dict(cutoff_length=0),
            dict(reference_velocity=np.nan),
            dict(segment_length=-1),
            dict(quadrature_order=1),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                pipeline.build_model(**(self.inputs() | overrides))
        model = pipeline.build_model(**self.inputs())
        for temperature in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                model.evaluate(temperature)
        for length in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                model.generator(0.2, length)


if __name__ == "__main__":
    unittest.main(verbosity=2)
