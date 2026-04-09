from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim.solver import RigidBodySolver
from sim.state import (
    CommandType,
    InteractionCommand,
    MotionType,
    RigidBodySnapshot,
    RigidBodyState,
    SimulationConfig,
    WorldState,
    create_box_body,
    vec3,
)


def _quat_from_euler_xyz(angles_deg) -> np.ndarray:
    ax, ay, az = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    cx, sx = np.cos(ax * 0.5), np.sin(ax * 0.5)
    cy, sy = np.cos(ay * 0.5), np.sin(ay * 0.5)
    cz, sz = np.cos(az * 0.5), np.sin(az * 0.5)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


@dataclass(slots=True)
class DemoDefinition:
    name: str
    description: str
    builder: Callable[[], None]


@dataclass(slots=True)
class TwoBodyCaseDefinition:
    name: str
    description: str
    builder: Callable[[], None]


class RigidBodyWorld:
    def __init__(self, config: SimulationConfig | None = None, demo_name: str = "single_body"):
        self.base_config = config.clone() if config is not None else SimulationConfig()
        self.config = self.base_config.clone()
        self.solver = RigidBodySolver(self.config)
        self.state = WorldState()
        self._demos: dict[str, DemoDefinition] = {}
        self._demo_order: list[str] = []
        self._two_body_cases: dict[str, TwoBodyCaseDefinition] = {}
        self._two_body_case_order: list[str] = []
        self._active_two_body_case_name = "face_face"
        self._two_body_half_extents = np.array([0.35, 0.35, 0.35], dtype=np.float64)
        self._selected_body_id: int | None = None
        self._initial_config = self.config.clone()
        self._initial_snapshots: dict[int, RigidBodySnapshot] = {}
        self._random_case_seed = 20260408
        self._register_two_body_cases()
        self._register_default_demos()
        self.load_demo(demo_name)

    def _register_two_body_cases(self) -> None:
        self.register_two_body_case(
            TwoBodyCaseDefinition(
                name="point_face",
                description="A rotated box hits the broad face of another same-sized axis-aligned box.",
                builder=self._build_two_body_point_face_case,
            )
        )
        self.register_two_body_case(
            TwoBodyCaseDefinition(
                name="edge_edge",
                description="Two rotated boxes approach with skewed edges to emphasize edge-edge contact.",
                builder=self._build_two_body_edge_edge_case,
            )
        )
        self.register_two_body_case(
            TwoBodyCaseDefinition(
                name="face_face",
                description="Two same-sized axis-aligned boxes collide head-on with broad opposing faces.",
                builder=self._build_two_body_face_face_case,
            )
        )
        self.register_two_body_case(
            TwoBodyCaseDefinition(
                name="random_pose",
                description="A reproducible pseudo-random pair of poses and velocities that still guarantees collision.",
                builder=self._build_two_body_random_pose_case,
            )
        )

    def _register_default_demos(self) -> None:
        self.register_demo(
            DemoDefinition(
                name="single_body",
                description="One rigid box with prescribed linear/angular velocity and user force hooks.",
                builder=self._build_single_body_demo,
            )
        )
        self.register_demo(
            DemoDefinition(
                name="two_body_collision",
                description="Two boxes moving toward each other for collision detection and impulse response.",
                builder=self._build_two_body_collision_demo,
            )
        )
        self.register_demo(
            DemoDefinition(
                name="complex_scene",
                description="A multi-body scene with gravity, static boundaries, and several simultaneous contacts.",
                builder=self._build_complex_scene_demo,
            )
        )

    def register_demo(self, definition: DemoDefinition) -> None:
        self._demos[definition.name] = definition
        if definition.name not in self._demo_order:
            self._demo_order.append(definition.name)

    def register_two_body_case(self, definition: TwoBodyCaseDefinition) -> None:
        self._two_body_cases[definition.name] = definition
        if definition.name not in self._two_body_case_order:
            self._two_body_case_order.append(definition.name)

    def list_demos(self) -> tuple[str, ...]:
        return tuple(self._demo_order)

    def list_two_body_cases(self) -> tuple[str, ...]:
        return tuple(self._two_body_case_order)

    def two_body_case_key_map(self) -> dict[str, str]:
        return {
            "4": "point_face",
            "5": "edge_edge",
            "6": "face_face",
            "7": "random_pose",
        }

    def load_demo(self, name: str) -> None:
        if name not in self._demos:
            available = ", ".join(self._demo_order)
            raise KeyError(f"unknown demo '{name}', available demos: {available}")

        definition = self._demos[name]
        self.config = self.base_config.clone()
        self.solver = RigidBodySolver(self.config)
        self.state = WorldState(active_demo=definition.name, demo_description=definition.description)
        definition.builder()
        if name == "two_body_collision":
            case = self.active_two_body_case()
            self.state.demo_description = (
                f"{definition.description} Case={case.name}: {case.description}"
            )
        self._select_default_dynamic_body()
        self.capture_initial_state()
        self._initial_config = self.config.clone()

    def capture_initial_state(self) -> None:
        self._initial_snapshots = {
            body.body_id: body.capture_snapshot()
            for body in self.state.bodies
        }

    def reset_active_demo(self) -> None:
        self.config = self._initial_config.clone()
        self.solver = RigidBodySolver(self.config)
        for body in self.state.bodies:
            snapshot = self._initial_snapshots[body.body_id]
            body.restore_snapshot(snapshot)
        self.state.contacts.clear()
        self.state.pending_commands.clear()
        self.state.time = 0.0
        self.state.frame = 0
        self.state.paused = False
        self._select_default_dynamic_body()

    def next_demo(self) -> None:
        current_index = self._demo_order.index(self.state.active_demo)
        next_index = (current_index + 1) % len(self._demo_order)
        self.load_demo(self._demo_order[next_index])

    def previous_demo(self) -> None:
        current_index = self._demo_order.index(self.state.active_demo)
        prev_index = (current_index - 1) % len(self._demo_order)
        self.load_demo(self._demo_order[prev_index])

    def add_body(self, body: RigidBodyState) -> int:
        body.body_id = len(self.state.bodies)
        self.state.bodies.append(body)
        return body.body_id

    def _dynamic_body_ids(self) -> list[int]:
        return [body.body_id for body in self.state.bodies if body.is_dynamic]

    def _select_default_dynamic_body(self) -> None:
        dynamic_ids = self._dynamic_body_ids()
        self._selected_body_id = dynamic_ids[0] if dynamic_ids else None

    def get_selected_body_id(self) -> int | None:
        if self._selected_body_id is None:
            self._select_default_dynamic_body()
        return self._selected_body_id

    def get_selected_body(self) -> RigidBodyState | None:
        body_id = self.get_selected_body_id()
        if body_id is None:
            return None
        return self.state.get_body(body_id)

    def select_next_dynamic_body(self) -> None:
        dynamic_ids = self._dynamic_body_ids()
        if not dynamic_ids:
            self._selected_body_id = None
            return
        current = self.get_selected_body_id()
        if current not in dynamic_ids:
            self._selected_body_id = dynamic_ids[0]
            return
        index = dynamic_ids.index(current)
        self._selected_body_id = dynamic_ids[(index + 1) % len(dynamic_ids)]

    def select_previous_dynamic_body(self) -> None:
        dynamic_ids = self._dynamic_body_ids()
        if not dynamic_ids:
            self._selected_body_id = None
            return
        current = self.get_selected_body_id()
        if current not in dynamic_ids:
            self._selected_body_id = dynamic_ids[0]
            return
        index = dynamic_ids.index(current)
        self._selected_body_id = dynamic_ids[(index - 1) % len(dynamic_ids)]

    def active_two_body_case(self) -> TwoBodyCaseDefinition:
        return self._two_body_cases[self._active_two_body_case_name]

    def select_two_body_case(self, name: str) -> None:
        if name not in self._two_body_cases:
            available = ", ".join(self._two_body_case_order)
            raise KeyError(f"unknown two-body case '{name}', available cases: {available}")
        self._active_two_body_case_name = name
        if self.state.active_demo == "two_body_collision":
            self.load_demo("two_body_collision")

    def select_two_body_case_by_key(self, key: str) -> bool:
        case_name = self.two_body_case_key_map().get(key)
        if case_name is None:
            return False
        self.select_two_body_case(case_name)
        return True

    def next_two_body_case(self) -> None:
        index = self._two_body_case_order.index(self._active_two_body_case_name)
        next_index = (index + 1) % len(self._two_body_case_order)
        self.select_two_body_case(self._two_body_case_order[next_index])

    def previous_two_body_case(self) -> None:
        index = self._two_body_case_order.index(self._active_two_body_case_name)
        prev_index = (index - 1) % len(self._two_body_case_order)
        self.select_two_body_case(self._two_body_case_order[prev_index])

    def get_primary_body_id(self) -> int | None:
        return self.get_selected_body_id()

    def queue_command(self, command: InteractionCommand) -> None:
        self.state.pending_commands.append(command)

    def apply_force_to_body(self, body_id: int, force, world_point=None) -> None:
        scaled_force = max(1, self.config.substeps) * np.asarray(force, dtype=np.float64)
        self.queue_command(
            InteractionCommand(
                command_type=CommandType.APPLY_FORCE,
                body_id=body_id,
                value=scaled_force,
                world_point=None if world_point is None else np.asarray(world_point, dtype=np.float64),
            )
        )

    def apply_impulse_to_body(self, body_id: int, impulse, world_point=None) -> None:
        self.queue_command(
            InteractionCommand(
                command_type=CommandType.APPLY_IMPULSE,
                body_id=body_id,
                value=np.asarray(impulse, dtype=np.float64),
                world_point=None if world_point is None else np.asarray(world_point, dtype=np.float64),
            )
        )

    def apply_torque_to_body(self, body_id: int, torque) -> None:
        self.queue_command(
            InteractionCommand(
                command_type=CommandType.APPLY_TORQUE,
                body_id=body_id,
                value=np.asarray(torque, dtype=np.float64),
            )
        )

    def toggle_pause(self) -> None:
        self.state.paused = not self.state.paused

    def step(self) -> None:
        if self.state.paused:
            return
        substeps = max(1, self.config.substeps)
        dt = self.config.time_step / substeps
        for _ in range(substeps):
            self.solver.step(self.state, dt)

    def get_body_positions(self) -> np.ndarray:
        if not self.state.bodies:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([body.position for body in self.state.bodies], axis=0)

    def get_body_colors(self) -> np.ndarray:
        if not self.state.bodies:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([body.color for body in self.state.bodies], axis=0)

    def describe(self) -> str:
        lines = [
            f"demo={self.state.active_demo}",
            f"description={self.state.demo_description}",
            f"frame={self.state.frame}",
            f"time={self.state.time:.4f}",
            f"bodies={len(self.state.bodies)}",
            f"contacts={len(self.state.contacts)}",
        ]
        if self.state.active_demo == "two_body_collision":
            lines.append(f"collision_case={self._active_two_body_case_name}")
        for body in self.state.bodies:
            pos = np.array2string(body.position, precision=3, suppress_small=True)
            vel = np.array2string(body.linear_velocity, precision=3, suppress_small=True)
            lines.append(f"body[{body.body_id}] {body.name}: pos={pos}, vel={vel}")
        return "\n".join(lines)

    def _build_single_body_demo(self) -> None:
        self.config.enable_gravity = False
        self.config.solver_iterations = 1
        self.add_body(
            create_box_body(
                name="single_box",
                half_extents=[0.45, 0.25, 0.2],
                position=[0.0, 0.0, 0.0],
                linear_velocity=[1.2, 0.0, 0.0],
                angular_velocity=[0.0, 1.5, 0.75],
                mass=1.0,
                color=[0.85, 0.45, 0.3],
            )
        )

    def _build_two_body_collision_demo(self) -> None:
        self.config.enable_gravity = False
        self.config.solver_iterations = 4
        self.active_two_body_case().builder()

    def _build_two_body_face_face_case(self) -> None:
        self.add_body(
            create_box_body(
                name="left_box",
                half_extents=self._two_body_half_extents,
                position=[-1.5, 0.0, 0.0],
                linear_velocity=[1.5, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
                mass=1.0,
                color=[0.2, 0.6, 0.95],
            )
        )
        self.add_body(
            create_box_body(
                name="right_box",
                half_extents=self._two_body_half_extents,
                position=[1.5, 0.0, 0.0],
                linear_velocity=[-1.5, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
                mass=1.0,
                color=[0.95, 0.6, 0.2],
            )
        )

    def _build_two_body_point_face_case(self) -> None:
        self.add_body(
            create_box_body(
                name="point_box",
                half_extents=self._two_body_half_extents,
                position=[-1.35, 0.38, 0.26],
                orientation=_quat_from_euler_xyz([18.0, 32.0, 21.0]),
                linear_velocity=[1.9, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
                mass=1.0,
                color=[0.2, 0.6, 0.95],
            )
        )
        self.add_body(
            create_box_body(
                name="face_box",
                half_extents=self._two_body_half_extents,
                position=[0.55, 0.0, 0.0],
                linear_velocity=[-0.25, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
                mass=1.0,
                color=[0.95, 0.6, 0.2],
            )
        )

    def _build_two_body_edge_edge_case(self) -> None:
        self.add_body(
            create_box_body(
                name="edge_box_a",
                half_extents=self._two_body_half_extents,
                position=[-1.0, 0.24, -0.05],
                orientation=_quat_from_euler_xyz([0.0, 42.0, 32.0]),
                linear_velocity=[1.55, -0.05, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
                mass=1.0,
                color=[0.2, 0.6, 0.95],
            )
        )
        self.add_body(
            create_box_body(
                name="edge_box_b",
                half_extents=self._two_body_half_extents,
                position=[1.0, -0.24, 0.05],
                orientation=_quat_from_euler_xyz([0.0, -42.0, -32.0]),
                linear_velocity=[-1.55, 0.05, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
                mass=1.0,
                color=[0.95, 0.6, 0.2],
            )
        )

    def _build_two_body_random_pose_case(self) -> None:
        rng = np.random.default_rng(self._random_case_seed)
        self._random_case_seed += 1

        left_angles = rng.uniform(-45.0, 45.0, size=3)
        right_angles = rng.uniform(-45.0, 45.0, size=3)
        left_offset = rng.uniform(-0.25, 0.25, size=2)
        right_offset = rng.uniform(-0.25, 0.25, size=2)
        left_speed = float(rng.uniform(1.2, 1.8))
        right_speed = float(rng.uniform(1.2, 1.8))

        self.add_body(
            create_box_body(
                name="random_box_a",
                half_extents=self._two_body_half_extents,
                position=[-1.15, left_offset[0], left_offset[1]],
                orientation=_quat_from_euler_xyz(left_angles),
                linear_velocity=[left_speed, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
                mass=1.0,
                color=[0.2, 0.6, 0.95],
            )
        )
        self.add_body(
            create_box_body(
                name="random_box_b",
                half_extents=self._two_body_half_extents,
                position=[1.15, right_offset[0], right_offset[1]],
                orientation=_quat_from_euler_xyz(right_angles),
                linear_velocity=[-right_speed, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
                mass=1.0,
                color=[0.95, 0.6, 0.2],
            )
        )

    def _build_complex_scene_demo(self) -> None:
        self.config.enable_gravity = True
        self.config.substeps = 6
        self.config.solver_iterations = 10
        self.config.linear_damping = 0.12
        self.config.angular_damping = 0.18
        self.add_body(
            create_box_body(
                name="floor",
                half_extents=[4.0, 0.2, 4.0],
                position=[0.0, -1.25, 0.0],
                motion_type=MotionType.STATIC,
                restitution=0.0,
                friction=0.9,
                color=[0.45, 0.45, 0.45],
                user_data={"environment_boundary": True},
            )
        )
        self.add_body(
            create_box_body(
                name="left_wall",
                half_extents=[0.2, 2.0, 4.0],
                position=[-3.5, 0.5, 0.0],
                motion_type=MotionType.STATIC,
                restitution=0.0,
                friction=0.8,
                color=[0.55, 0.55, 0.6],
                user_data={"environment_boundary": True},
            )
        )
        self.add_body(
            create_box_body(
                name="right_wall",
                half_extents=[0.2, 2.0, 4.0],
                position=[3.5, 0.5, 0.0],
                motion_type=MotionType.STATIC,
                restitution=0.0,
                friction=0.8,
                color=[0.55, 0.55, 0.6],
                user_data={"environment_boundary": True},
            )
        )

        dynamic_boxes = [
            ("striker", [-1.8, -0.75, 0.0], [1.4, 0.0, 0.0], [0.9, 0.35, 0.25], [0.0, 0.0, 0.0]),
            ("base_left", [-0.35, -0.75, 0.0], [0.0, 0.0, 0.0], [0.25, 0.72, 0.44], [0.0, 0.0, 8.0]),
            ("base_right", [0.45, -0.75, 0.0], [0.0, 0.0, 0.0], [0.3, 0.82, 0.54], [0.0, 0.0, -6.0]),
            ("dropper", [0.05, 0.10, 0.0], [0.0, -0.1, 0.0], [0.92, 0.56, 0.34], [0.0, 0.0, 0.0]),
        ]
        for name, position, linear_velocity, color, orientation_deg in dynamic_boxes:
            self.add_body(
                create_box_body(
                    name=name,
                    half_extents=[0.3, 0.3, 0.3],
                    position=position,
                    orientation=_quat_from_euler_xyz(orientation_deg),
                    linear_velocity=linear_velocity,
                    angular_velocity=[0.0, 0.0, 0.0],
                    mass=1.0,
                    restitution=0.0,
                    friction=0.45,
                    color=color,
                )
            )
