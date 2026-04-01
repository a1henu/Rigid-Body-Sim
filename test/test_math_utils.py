from pathlib import Path
import sys

import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.math_utils import (
    integrate_quat,
    quat_conjugate,
    quat_from_axis_angle,
    quat_identity,
    quat_inverse,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_mat3,
)


ti.init(arch=ti.cpu, default_fp=ti.f64)

VEC3 = ti.types.vector(3, ti.f64)
VEC4 = ti.types.vector(4, ti.f64)

quat_out = ti.Vector.field(4, dtype=ti.f64, shape=1)
vec_out = ti.Vector.field(3, dtype=ti.f64, shape=1)
mat_out = ti.Matrix.field(3, 3, dtype=ti.f64, shape=1)


@ti.kernel
def _quat_identity_kernel():
    quat_out[0] = quat_identity()


@ti.kernel
def _quat_normalize_kernel(q: VEC4):
    quat_out[0] = quat_normalize(q)


@ti.kernel
def _quat_conjugate_kernel(q: VEC4):
    quat_out[0] = quat_conjugate(q)


@ti.kernel
def _quat_inverse_kernel(q: VEC4):
    quat_out[0] = quat_inverse(q)


@ti.kernel
def _quat_mul_kernel(q1: VEC4, q2: VEC4):
    quat_out[0] = quat_mul(q1, q2)


@ti.kernel
def _quat_from_axis_angle_kernel(axis: VEC3, angle: ti.f64):
    quat_out[0] = quat_from_axis_angle(axis, angle)


@ti.kernel
def _quat_rotate_kernel(q: VEC4, v: VEC3):
    vec_out[0] = quat_rotate(q, v)


@ti.kernel
def _quat_to_mat3_kernel(q: VEC4):
    mat_out[0] = quat_to_mat3(q)


@ti.kernel
def _integrate_quat_kernel(q: VEC4, omega: VEC3, dt: ti.f64):
    quat_out[0] = integrate_quat(q, omega, dt)


def scipy_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def wxyz_to_scipy(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def assert_quat_allclose(actual, expected, atol=1e-8):
    if np.linalg.norm(actual - expected) > np.linalg.norm(actual + expected):
        expected = -expected
    np.testing.assert_allclose(actual, expected, atol=atol)


def quat_identity_np():
    _quat_identity_kernel()
    return quat_out.to_numpy()[0]


def quat_normalize_np(q):
    _quat_normalize_kernel(np.asarray(q, dtype=np.float64))
    return quat_out.to_numpy()[0]


def quat_conjugate_np(q):
    _quat_conjugate_kernel(np.asarray(q, dtype=np.float64))
    return quat_out.to_numpy()[0]


def quat_inverse_np(q):
    _quat_inverse_kernel(np.asarray(q, dtype=np.float64))
    return quat_out.to_numpy()[0]


def quat_mul_np(q1, q2):
    _quat_mul_kernel(np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64))
    return quat_out.to_numpy()[0]


def quat_from_axis_angle_np(axis, angle):
    _quat_from_axis_angle_kernel(np.asarray(axis, dtype=np.float64), float(angle))
    return quat_out.to_numpy()[0]


def quat_rotate_np(q, v):
    _quat_rotate_kernel(np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64))
    return vec_out.to_numpy()[0]


def quat_to_mat3_np(q):
    _quat_to_mat3_kernel(np.asarray(q, dtype=np.float64))
    return mat_out.to_numpy()[0]


def integrate_quat_np(q, omega, dt):
    _integrate_quat_kernel(
        np.asarray(q, dtype=np.float64),
        np.asarray(omega, dtype=np.float64),
        float(dt),
    )
    return quat_out.to_numpy()[0]


def test_quat_identity():
    np.testing.assert_allclose(quat_identity_np(), np.array([1.0, 0.0, 0.0, 0.0]))


def test_quat_normalize_random_inputs():
    rng = np.random.default_rng(1234)
    raw_quats = rng.normal(size=(64, 4))
    for raw_q in raw_quats:
        actual = quat_normalize_np(raw_q)
        expected = raw_q / np.linalg.norm(raw_q)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(actual), 1.0, atol=1e-8)


def test_quat_conjugate_and_inverse_random_inputs():
    rng = np.random.default_rng(5678)
    raw_quats = rng.normal(size=(64, 4))
    for raw_q in raw_quats:
        expected_conj = np.array([raw_q[0], -raw_q[1], -raw_q[2], -raw_q[3]])
        actual_conj = quat_conjugate_np(raw_q)
        np.testing.assert_allclose(actual_conj, expected_conj, atol=1e-8)

        expected_inv = expected_conj / np.dot(raw_q, raw_q)
        actual_inv = quat_inverse_np(raw_q)
        np.testing.assert_allclose(actual_inv, expected_inv, atol=1e-8)


def test_quat_to_mat3_matches_scipy_random_quaternions():
    rotations = Rotation.random(64, random_state=2024)
    for q_xyzw in rotations.as_quat():
        q_wxyz = scipy_to_wxyz(q_xyzw)
        actual = quat_to_mat3_np(q_wxyz)
        expected = Rotation.from_quat(q_xyzw).as_matrix()
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_quat_rotate_matches_scipy_random_cases():
    rng = np.random.default_rng(2025)
    rotations = Rotation.random(64, random_state=2026)
    vectors = rng.normal(size=(64, 3))
    for q_xyzw, v in zip(rotations.as_quat(), vectors):
        q_wxyz = scipy_to_wxyz(q_xyzw)
        actual = quat_rotate_np(q_wxyz, v)
        expected = Rotation.from_quat(q_xyzw).apply(v)
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_quat_mul_matches_rotation_composition():
    rotations_1 = Rotation.random(32, random_state=2027)
    rotations_2 = Rotation.random(32, random_state=2028)
    for q1_xyzw, q2_xyzw in zip(rotations_1.as_quat(), rotations_2.as_quat()):
        q1_wxyz = scipy_to_wxyz(q1_xyzw)
        q2_wxyz = scipy_to_wxyz(q2_xyzw)
        actual_q = quat_mul_np(q1_wxyz, q2_wxyz)
        actual_mat = quat_to_mat3_np(actual_q)
        expected_mat = (
            Rotation.from_quat(q1_xyzw).as_matrix()
            @ Rotation.from_quat(q2_xyzw).as_matrix()
        )
        np.testing.assert_allclose(actual_mat, expected_mat, atol=1e-8)


def test_quat_from_axis_angle_matches_scipy():
    rng = np.random.default_rng(2029)
    axes = rng.normal(size=(64, 3))
    angles = rng.uniform(-np.pi, np.pi, size=64)
    for axis, angle in zip(axes, angles):
        axis = axis / np.linalg.norm(axis)
        actual = quat_from_axis_angle_np(axis, angle)
        expected = scipy_to_wxyz(Rotation.from_rotvec(axis * angle).as_quat())
        assert_quat_allclose(actual, expected, atol=1e-8)


def test_quat_inverse_undoes_rotation():
    rng = np.random.default_rng(2030)
    rotations = Rotation.random(32, random_state=2031)
    vectors = rng.normal(size=(32, 3))
    for q_xyzw, v in zip(rotations.as_quat(), vectors):
        q_wxyz = scipy_to_wxyz(q_xyzw)
        q_inv = quat_inverse_np(q_wxyz)
        rotated = quat_rotate_np(q_wxyz, v)
        recovered = quat_rotate_np(q_inv, rotated)
        np.testing.assert_allclose(recovered, v, atol=1e-8)


def test_integrate_quat_zero_omega_keeps_orientation():
    rotations = Rotation.random(32, random_state=2032)
    zero_omega = np.zeros(3, dtype=np.float64)
    for q_xyzw in rotations.as_quat():
        q_wxyz = scipy_to_wxyz(q_xyzw)
        actual = integrate_quat_np(q_wxyz, zero_omega, 0.01)
        assert_quat_allclose(actual, q_wxyz, atol=1e-8)


def test_integrate_quat_small_step_matches_exact_world_space_delta():
    rng = np.random.default_rng(2033)
    rotations = Rotation.random(32, random_state=2034)
    omegas = rng.normal(size=(32, 3))
    dt = 1e-6
    for q_xyzw, omega in zip(rotations.as_quat(), omegas):
        q_wxyz = scipy_to_wxyz(q_xyzw)
        delta_xyzw = Rotation.from_rotvec(omega * dt).as_quat()
        expected = quat_mul_np(scipy_to_wxyz(delta_xyzw), q_wxyz)
        actual = integrate_quat_np(q_wxyz, omega, dt)
        assert_quat_allclose(actual, expected, atol=1e-8)


def test_integrate_quat_multi_step_stays_normalized_and_tracks_exact_rotation():
    rng = np.random.default_rng(2035)
    rotations = Rotation.random(8, random_state=2036)
    omegas = rng.normal(size=(8, 3))
    dt = 1e-4
    steps = 200
    total_time = dt * steps

    for q_xyzw, omega in zip(rotations.as_quat(), omegas):
        q_wxyz = scipy_to_wxyz(q_xyzw)
        integrated = q_wxyz.copy()
        for _ in range(steps):
            integrated = integrate_quat_np(integrated, omega, dt)

        expected = quat_mul_np(
            scipy_to_wxyz(Rotation.from_rotvec(omega * total_time).as_quat()),
            q_wxyz,
        )
        assert_quat_allclose(integrated, expected, atol=2e-5)
        np.testing.assert_allclose(np.linalg.norm(integrated), 1.0, atol=1e-10)
