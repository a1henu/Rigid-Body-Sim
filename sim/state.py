from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def quat_identity_wxyz() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def mat3_identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


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
    user_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_dynamic(self) -> bool:
        return self.motion_type == MotionType.DYNAMIC

    def clear_accumulators(self) -> None:
        self.force_accumulator.fill(0.0)
        self.torque_accumulator.fill(0.0)

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


def placeholder_box_inertia(mass: float) -> tuple[np.ndarray, np.ndarray]:
    # TODO: replace this isotropic placeholder with the exact box inertia tensor.
    inertia = np.eye(3, dtype=np.float64) * max(mass, 1e-8)
    inverse_inertia = np.eye(3, dtype=np.float64) / max(mass, 1e-8)
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
        inertia_body, inverse_inertia_body = placeholder_box_inertia(resolved_mass)

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
