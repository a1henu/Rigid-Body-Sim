from __future__ import annotations

from typing import Iterable
from dataclasses import dataclass

import numpy as np

from sim.state import (
    CommandType,
    Contact,
    RigidBodyState,
    SimulationConfig,
    WorldState,
    integrate_quat_wxyz,
    safe_normalize,
)


@dataclass(slots=True)
class _AxisTestResult:
    overlap: float
    axis_world: np.ndarray
    axis_kind: str
    axis_i: int
    axis_j: int = -1


class RigidBodySolver:
    """
    High-level simulation pipeline for one rigid-body time step.

    The orchestration is intentionally complete, but the numerically important
    pieces are still TODO so the lab implementation can be filled in gradually.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def step(self, state: WorldState, dt: float | None = None) -> None:
        step_dt = self.config.time_step if dt is None else float(dt)
        self._begin_step(state, step_dt)
        self._clear_transient_state(state)
        self._accumulate_global_forces(state, step_dt)
        self._consume_interaction_commands(state, step_dt)
        self._integrate_velocities(state, step_dt)
        if self.config.enable_collisions:
            state.contacts = self._detect_collisions(state)
            self._resolve_contacts(state, step_dt)
        else:
            state.contacts.clear()
        self._integrate_positions(state, step_dt)
        self._end_step(state, step_dt)

    def _begin_step(self, state: WorldState, dt: float) -> None:
        _ = state, dt

    def _clear_transient_state(self, state: WorldState) -> None:
        state.contacts.clear()
        for body in state.bodies:
            body.clear_accumulators()

    def _accumulate_global_forces(self, state: WorldState, dt: float) -> None:
        _ = dt
        if not self.config.enable_gravity:
            return
        for body in state.bodies:
            if body.is_dynamic:
                body.force_accumulator += body.mass * self.config.gravity

    def _consume_interaction_commands(self, state: WorldState, dt: float) -> None:
        _ = dt
        for command in state.pending_commands:
            body = state.get_body(command.body_id)
            if command.command_type == CommandType.APPLY_FORCE:
                body.force_accumulator += command.value
                if command.world_point is not None:
                    lever_arm = command.world_point - body.position
                    body.torque_accumulator += np.cross(lever_arm, command.value)
            elif command.command_type == CommandType.APPLY_TORQUE:
                body.torque_accumulator += command.value
            elif command.command_type == CommandType.APPLY_IMPULSE:
                self._apply_impulse_placeholder(body, command)
            elif command.command_type == CommandType.DRAG_BODY:
                self._apply_drag_placeholder(body, command)
        state.pending_commands.clear()

    def _apply_impulse_placeholder(self, body: RigidBodyState, command) -> None:
        # TODO: apply instantaneous linear/angular velocity updates here.
        _ = body, command

    def _apply_drag_placeholder(self, body: RigidBodyState, command) -> None:
        # TODO: convert drag interaction into forces or kinematic targets here.
        _ = body, command

    def _integrate_velocities(self, state: WorldState, dt: float) -> None:
        for body in state.bodies:
            if body.is_dynamic:
                self._integrate_body_velocity(body, dt)

    def _integrate_body_velocity(self, body: RigidBodyState, dt: float) -> None:
        linear_acceleration = body.force_accumulator * body.inverse_mass
        angular_acceleration = body.inverse_inertia_world() @ body.torque_accumulator

        body.linear_velocity = body.linear_velocity + dt * linear_acceleration
        body.angular_velocity = body.angular_velocity + dt * angular_acceleration

        linear_damping = max(0.0, 1.0 - self.config.linear_damping * dt)
        angular_damping = max(0.0, 1.0 - self.config.angular_damping * dt)
        body.linear_velocity *= linear_damping
        body.angular_velocity *= angular_damping

    def _detect_collisions(self, state: WorldState) -> list[Contact]:
        contacts: list[Contact] = []
        bodies = state.bodies
        for i, body_a in enumerate(bodies):
            for body_b in bodies[i + 1 :]:
                if not self._should_test_pair(body_a, body_b):
                    continue
                contacts.extend(self._detect_body_pair(body_a, body_b))
        return contacts

    def _should_test_pair(self, body_a: RigidBodyState, body_b: RigidBodyState) -> bool:
        return body_a.is_dynamic or body_b.is_dynamic

    def _detect_body_pair(
        self,
        body_a: RigidBodyState,
        body_b: RigidBodyState,
    ) -> Iterable[Contact]:
        contact = self._detect_box_box_sat(body_a, body_b)
        if contact is None:
            return []
        return [contact]

    def _detect_box_box_sat(
        self,
        body_a: RigidBodyState,
        body_b: RigidBodyState,
    ) -> Contact | None:
        rotation_a = body_a.rotation_matrix()
        rotation_b = body_b.rotation_matrix()
        half_a = body_a.half_extents
        half_b = body_b.half_extents

        to_b_world = body_b.position - body_a.position
        relative_rotation = rotation_a.T @ rotation_b
        to_b_in_a = rotation_a.T @ to_b_world

        epsilon = 1e-8
        abs_relative_rotation = np.abs(relative_rotation) + epsilon
        best_axis: _AxisTestResult | None = None

        def consider_axis(
            overlap: float,
            axis_world: np.ndarray,
            axis_kind: str,
            axis_i: int,
            axis_j: int = -1,
        ) -> bool:
            nonlocal best_axis
            if overlap < 0.0:
                return False
            axis_world = safe_normalize(axis_world)
            if np.linalg.norm(axis_world) < 1e-8:
                return True
            if best_axis is None or overlap < best_axis.overlap:
                best_axis = _AxisTestResult(
                    overlap=float(overlap),
                    axis_world=axis_world,
                    axis_kind=axis_kind,
                    axis_i=axis_i,
                    axis_j=axis_j,
                )
            return True

        for i in range(3):
            ra = half_a[i]
            rb = np.dot(half_b, abs_relative_rotation[i, :])
            overlap = ra + rb - abs(to_b_in_a[i])
            if not consider_axis(overlap, rotation_a[:, i], "face_a", i):
                return None

        for j in range(3):
            axis_b_in_a = relative_rotation[:, j]
            ra = np.dot(half_a, abs_relative_rotation[:, j])
            rb = half_b[j]
            distance = abs(np.dot(to_b_in_a, axis_b_in_a))
            overlap = ra + rb - distance
            if not consider_axis(overlap, rotation_b[:, j], "face_b", j):
                return None

        for i in range(3):
            i1 = (i + 1) % 3
            i2 = (i + 2) % 3
            for j in range(3):
                j1 = (j + 1) % 3
                j2 = (j + 2) % 3
                ra = half_a[i1] * abs_relative_rotation[i2, j] + half_a[i2] * abs_relative_rotation[i1, j]
                rb = half_b[j1] * abs_relative_rotation[i, j2] + half_b[j2] * abs_relative_rotation[i, j1]
                distance = abs(
                    to_b_in_a[i2] * relative_rotation[i1, j]
                    - to_b_in_a[i1] * relative_rotation[i2, j]
                )
                overlap = ra + rb - distance
                if not consider_axis(
                    overlap,
                    np.cross(rotation_a[:, i], rotation_b[:, j]),
                    "edge_cross",
                    i,
                    j,
                ):
                    return None

        if best_axis is None:
            return None

        normal_world = best_axis.axis_world
        if np.dot(normal_world, to_b_world) < 0.0:
            normal_world = -normal_world

        point_a = body_a.support_point_world(normal_world)
        point_b = body_b.support_point_world(-normal_world)
        contact_position = 0.5 * (point_a + point_b)

        return Contact(
            body_a=body_a.body_id,
            body_b=body_b.body_id,
            position=contact_position,
            normal=normal_world,
            penetration_depth=best_axis.overlap,
            feature_id=f"{best_axis.axis_kind}:{best_axis.axis_i}:{best_axis.axis_j}",
        )

    def _resolve_contacts(self, state: WorldState, dt: float) -> None:
        iterations = max(1, self.config.solver_iterations)
        for _ in range(iterations):
            for contact in state.contacts:
                self._solve_contact(contact, state, dt)

    def _solve_contact(self, contact: Contact, state: WorldState, dt: float) -> None:
        # TODO: impulse-based collision response goes here.
        _ = contact, state, dt

    def _integrate_positions(self, state: WorldState, dt: float) -> None:
        for body in state.bodies:
            if body.is_dynamic:
                self._integrate_body_pose(body, dt)

    def _integrate_body_pose(self, body: RigidBodyState, dt: float) -> None:
        body.position = body.position + dt * body.linear_velocity
        body.orientation = integrate_quat_wxyz(body.orientation, body.angular_velocity, dt)

    def _end_step(self, state: WorldState, dt: float) -> None:
        state.time += dt
        state.frame += 1
