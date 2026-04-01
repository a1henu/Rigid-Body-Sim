"""
Math utilities for the rigid body simulation.

Quaternion convention:
    q = (w, x, y, z)
where ``w`` is the scalar part and ``(x, y, z)`` is the vector part.
All rotation helpers assume the quaternion represents an active rotation.
"""

import taichi as ti


@ti.func
def safe_normalize(v):
    eps = 1e-12
    return v / ti.max(v.norm(), eps)


@ti.func
def quat_identity():
    return ti.Vector([1.0, 0.0, 0.0, 0.0])


@ti.func
def quat_normalize(q):
    eps = 1e-12
    return q / ti.max(q.norm(), eps)


@ti.func
def quat_conjugate(q):
    return ti.Vector([q[0], -q[1], -q[2], -q[3]])


@ti.func
def quat_inverse(q):
    eps = 1e-12
    return quat_conjugate(q) / ti.max(q.dot(q), eps)


@ti.func
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return ti.Vector([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


@ti.func
def quat_from_axis_angle(axis, angle):
    eps = 1e-12
    half_angle = 0.5 * angle
    axis_norm = axis.norm()
    axis_unit = axis / ti.max(axis_norm, eps)
    sin_half = ti.sin(half_angle)
    q = ti.Vector([
        ti.cos(half_angle),
        axis_unit[0] * sin_half,
        axis_unit[1] * sin_half,
        axis_unit[2] * sin_half,
    ])
    if axis_norm < eps:
        q = quat_identity()
    return quat_normalize(q)


@ti.func
def quat_to_mat3(q):
    qn = quat_normalize(q)
    w, x, y, z = qn[0], qn[1], qn[2], qn[3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return ti.Matrix([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])


@ti.func
def quat_rotate(q, v):
    return quat_to_mat3(q) @ v


@ti.func
def integrate_quat(q, omega, dt):
    # q represents an active body-to-world rotation.
    # omega is interpreted in world coordinates, so dq/dt = 0.5 * [0, omega] * q.
    omega_quat = ti.Vector([0.0, omega[0], omega[1], omega[2]])
    q_dot = 0.5 * quat_mul(omega_quat, q)
    return quat_normalize(q + dt * q_dot)
