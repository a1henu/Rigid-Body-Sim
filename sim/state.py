from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def vec2(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


def quat_identity_wxyz() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def mat3_identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def safe_normalize(v: Any, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / norm


def quat_normalize_wxyz(q: Any, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < eps:
        return quat_identity_wxyz()
    return arr / norm


def quat_conjugate_wxyz(q: Any) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    return np.array([arr[0], -arr[1], -arr[2], -arr[3]], dtype=np.float64)


def quat_mul_wxyz(q1: Any, q2: Any) -> np.ndarray:
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_mat3_wxyz(q: Any) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(q)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_rotate_wxyz(q: Any, v: Any) -> np.ndarray:
    return quat_to_mat3_wxyz(q) @ np.asarray(v, dtype=np.float64)


def integrate_quat_wxyz(q: Any, omega_world: Any, dt: float) -> np.ndarray:
    omega = np.asarray(omega_world, dtype=np.float64)
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
    q_dot = 0.5 * quat_mul_wxyz(omega_quat, np.asarray(q, dtype=np.float64))
    return quat_normalize_wxyz(np.asarray(q, dtype=np.float64) + float(dt) * q_dot)


def box_local_corners(half_extents: Any) -> np.ndarray:
    hx, hy, hz = _as_vec3(half_extents)
    return np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )


def signed_box_support(half_extents: Any, direction_local: Any) -> np.ndarray:
    half = _as_vec3(half_extents)
    direction = np.asarray(direction_local, dtype=np.float64)
    return np.where(direction >= 0.0, half, -half)


def _as_vec3(value: Any, *, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        value = default if default is not None else vec3()
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3D vector, got shape {arr.shape}")
    return arr.copy()


def _as_quat(value: Any, *, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        value = default if default is not None else quat_identity_wxyz()
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"expected a quaternion in wxyz order, got shape {arr.shape}")
    return arr.copy()


class MotionType(str, Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"


class CommandType(str, Enum):
    APPLY_FORCE = "apply_force"
    APPLY_TORQUE = "apply_torque"
    APPLY_IMPULSE = "apply_impulse"
    DRAG_BODY = "drag_body"


@dataclass(slots=True)
class SimulationConfig:
    time_step: float = 1.0 / 60.0
    substeps: int = 1
    solver_iterations: int = 4
    gravity: np.ndarray = field(default_factory=lambda: vec3(0.0, -9.8, 0.0))
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    default_restitution: float = 0.3
    default_friction: float = 0.5
    enable_gravity: bool = False
    enable_collisions: bool = True
    max_contacts_per_pair: int = 8
    use_taichi_step: bool = False

    def clone(self) -> "SimulationConfig":
        return copy.deepcopy(self)


@dataclass(slots=True)
class RigidBodySnapshot:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


@dataclass(slots=True)
class RigidBodyState:
    body_id: int = -1
    name: str = "body"
    motion_type: MotionType = MotionType.DYNAMIC
    half_extents: np.ndarray = field(default_factory=lambda: vec3(0.5, 0.5, 0.5))
    mass: float = 1.0
    inverse_mass: float = 1.0
    inertia_body: np.ndarray = field(default_factory=mat3_identity)
    inverse_inertia_body: np.ndarray = field(default_factory=mat3_identity)
    position: np.ndarray = field(default_factory=vec3)
    orientation: np.ndarray = field(default_factory=quat_identity_wxyz)
    linear_velocity: np.ndarray = field(default_factory=vec3)
    angular_velocity: np.ndarray = field(default_factory=vec3)
    force_accumulator: np.ndarray = field(default_factory=vec3)
    torque_accumulator: np.ndarray = field(default_factory=vec3)
    restitution: float = 0.3
    friction: float = 0.5
    color: np.ndarray = field(default_factory=lambda: vec3(0.7, 0.7, 0.8))
    is_sleeping: bool = False
    sleep_counter: int = 0
    user_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_dynamic(self) -> bool:
        return self.motion_type == MotionType.DYNAMIC

    def clear_accumulators(self) -> None:
        self.force_accumulator.fill(0.0)
        self.torque_accumulator.fill(0.0)

    def wake(self) -> None:
        self.is_sleeping = False
        self.sleep_counter = 0

    def sleep(self) -> None:
        self.is_sleeping = True
        self.sleep_counter = 0
        self.linear_velocity.fill(0.0)
        self.angular_velocity.fill(0.0)

    def rotation_matrix(self) -> np.ndarray:
        return quat_to_mat3_wxyz(self.orientation)

    def inverse_inertia_world(self) -> np.ndarray:
        if not self.is_dynamic:
            return np.zeros((3, 3), dtype=np.float64)
        rotation = self.rotation_matrix()
        return rotation @ self.inverse_inertia_body @ rotation.T

    def world_corners(self) -> np.ndarray:
        local_corners = box_local_corners(self.half_extents)
        return (self.rotation_matrix() @ local_corners.T).T + self.position

    def support_point_world(self, direction_world: Any) -> np.ndarray:
        direction_world = np.asarray(direction_world, dtype=np.float64)
        rotation = self.rotation_matrix()
        direction_local = rotation.T @ direction_world
        local_point = signed_box_support(self.half_extents, direction_local)
        return self.position + rotation @ local_point

    def capture_snapshot(self) -> RigidBodySnapshot:
        return RigidBodySnapshot(
            position=self.position.copy(),
            orientation=self.orientation.copy(),
            linear_velocity=self.linear_velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )

    def restore_snapshot(self, snapshot: RigidBodySnapshot) -> None:
        self.position = snapshot.position.copy()
        self.orientation = snapshot.orientation.copy()
        self.linear_velocity = snapshot.linear_velocity.copy()
        self.angular_velocity = snapshot.angular_velocity.copy()
        self.wake()
        self.clear_accumulators()


@dataclass(slots=True)
class Contact:
    body_a: int
    body_b: int
    position: np.ndarray
    normal: np.ndarray
    penetration_depth: float
    accumulated_normal_impulse: float = 0.0
    feature_id: str = ""


@dataclass(slots=True)
class InteractionCommand:
    command_type: CommandType
    body_id: int
    value: np.ndarray
    world_point: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorldState:
    bodies: list[RigidBodyState] = field(default_factory=list)
    contacts: list[Contact] = field(default_factory=list)
    pending_commands: list[InteractionCommand] = field(default_factory=list)
    active_demo: str = "single_body"
    demo_description: str = ""
    time: float = 0.0
    frame: int = 0
    paused: bool = False

    def get_body(self, body_id: int) -> RigidBodyState:
        for body in self.bodies:
            if body.body_id == body_id:
                return body
        raise KeyError(f"unknown body id {body_id}")


def box_inertia_tensor(mass: float, half_extents: Any) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = _as_vec3(half_extents)
    coeff = mass / 3.0
    inertia_diag = np.array(
        [
            coeff * (hy * hy + hz * hz),
            coeff * (hx * hx + hz * hz),
            coeff * (hx * hx + hy * hy),
        ],
        dtype=np.float64,
    )
    inertia = np.diag(inertia_diag)
    inverse_diag = np.divide(
        1.0,
        inertia_diag,
        out=np.zeros_like(inertia_diag),
        where=np.abs(inertia_diag) > 1e-12,
    )
    inverse_inertia = np.diag(inverse_diag)
    return inertia, inverse_inertia


def create_box_body(
    *,
    name: str,
    half_extents: Any,
    position: Any = None,
    orientation: Any = None,
    linear_velocity: Any = None,
    angular_velocity: Any = None,
    mass: float = 1.0,
    motion_type: MotionType = MotionType.DYNAMIC,
    color: Any = None,
    restitution: float = 0.3,
    friction: float = 0.5,
    user_data: dict[str, Any] | None = None,
) -> RigidBodyState:
    motion_type = MotionType(motion_type)
    if motion_type == MotionType.STATIC:
        resolved_mass = 0.0
        inverse_mass = 0.0
        inertia_body = np.zeros((3, 3), dtype=np.float64)
        inverse_inertia_body = np.zeros((3, 3), dtype=np.float64)
    else:
        resolved_mass = float(mass)
        inverse_mass = 1.0 / max(resolved_mass, 1e-8)
        inertia_body, inverse_inertia_body = box_inertia_tensor(resolved_mass, half_extents)

    return RigidBodyState(
        name=name,
        motion_type=motion_type,
        half_extents=_as_vec3(half_extents),
        mass=resolved_mass,
        inverse_mass=inverse_mass,
        inertia_body=inertia_body,
        inverse_inertia_body=inverse_inertia_body,
        position=_as_vec3(position),
        orientation=_as_quat(orientation),
        linear_velocity=_as_vec3(linear_velocity),
        angular_velocity=_as_vec3(angular_velocity),
        restitution=float(restitution),
        friction=float(friction),
        color=_as_vec3(color, default=vec3(0.7, 0.7, 0.8)),
        user_data=dict(user_data or {}),
    )
