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
            self._stabilize_supported_bodies(state, step_dt)
        else:
            state.contacts.clear()
        self._integrate_positions(state, step_dt)
        self._update_sleep_states(state)
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
            body.wake()
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
        if not body.is_dynamic:
            return
        impulse = np.asarray(command.value, dtype=np.float64)
        if np.linalg.norm(impulse) > 1e-10:
            body.wake()
        body.linear_velocity = body.linear_velocity + impulse * body.inverse_mass
        if command.world_point is not None:
            lever_arm = np.asarray(command.world_point, dtype=np.float64) - body.position
            body.angular_velocity = (
                body.angular_velocity
                + body.inverse_inertia_world() @ np.cross(lever_arm, impulse)
            )

    def _apply_drag_placeholder(self, body: RigidBodyState, command) -> None:
        # TODO: convert drag interaction into forces or kinematic targets here.
        _ = body, command

    def _integrate_velocities(self, state: WorldState, dt: float) -> None:
        for body in state.bodies:
            if body.is_dynamic and not body.is_sleeping:
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
        if body_a.is_dynamic and self._is_environment_boundary(body_b):
            return self._detect_dynamic_boundary_contacts(body_a, body_b)
        if body_b.is_dynamic and self._is_environment_boundary(body_a):
            return self._detect_dynamic_boundary_contacts(body_b, body_a)
        contact = self._detect_box_box_sat(body_a, body_b)
        if contact is None:
            return []
        return [contact]

    def _detect_dynamic_boundary_contacts(
        self,
        dynamic_body: RigidBodyState,
        boundary_body: RigidBodyState,
    ) -> list[Contact]:
        boundary_rotation = boundary_body.rotation_matrix()
        boundary_rotation_t = boundary_rotation.T
        world_corners = dynamic_body.world_corners()
        local_corners = (
            boundary_rotation_t @ (world_corners - boundary_body.position).T
        ).T

        contacts: list[Contact] = []
        contact_skin = 2e-3
        for corner_id, local_corner in enumerate(local_corners):
            margins = boundary_body.half_extents - np.abs(local_corner)
            if np.any(margins < -contact_skin):
                continue

            axis = int(np.argmin(margins))
            face_sign = 1.0 if local_corner[axis] >= 0.0 else -1.0
            face_normal_local = np.zeros(3, dtype=np.float64)
            face_normal_local[axis] = face_sign
            boundary_surface_local = local_corner.copy()
            boundary_surface_local[axis] = face_sign * boundary_body.half_extents[axis]

            contacts.append(
                Contact(
                    body_a=dynamic_body.body_id,
                    body_b=boundary_body.body_id,
                    position=boundary_body.position + boundary_rotation @ boundary_surface_local,
                    normal=-(boundary_rotation @ face_normal_local),
                    penetration_depth=float(max(margins[axis], 0.0)),
                    feature_id=f"boundary:{axis}:{corner_id}",
                )
            )

        contacts.sort(key=lambda contact: contact.penetration_depth, reverse=True)
        max_contacts = max(1, self.config.max_contacts_per_pair)
        return contacts[:max_contacts]

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
        skip_edge_cross_axes = self._should_skip_edge_cross_axes(body_a, body_b)

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

        if not skip_edge_cross_axes:
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

        contact_position = self._estimate_contact_position(
            body_a,
            body_b,
            normal_world,
            best_axis,
        )

        return Contact(
            body_a=body_a.body_id,
            body_b=body_b.body_id,
            position=contact_position,
            normal=normal_world,
            penetration_depth=best_axis.overlap,
            feature_id=f"{best_axis.axis_kind}:{best_axis.axis_i}:{best_axis.axis_j}",
        )

    def _should_skip_edge_cross_axes(
        self,
        body_a: RigidBodyState,
        body_b: RigidBodyState,
    ) -> bool:
        return self._is_environment_boundary(body_a) or self._is_environment_boundary(body_b)

    def _is_environment_boundary(self, body: RigidBodyState) -> bool:
        return (not body.is_dynamic) and bool(body.user_data.get("environment_boundary", False))

    def _estimate_contact_position(
        self,
        body_a: RigidBodyState,
        body_b: RigidBodyState,
        normal_world: np.ndarray,
        axis_result: _AxisTestResult,
    ) -> np.ndarray:
        if axis_result.axis_kind in {"face_a", "face_b"}:
            face_center_a = (
                body_a.position
                + normal_world * self._support_extent_along_normal(body_a, normal_world)
            )
            face_center_b = (
                body_b.position
                - normal_world * self._support_extent_along_normal(body_b, normal_world)
            )
            return 0.5 * (face_center_a + face_center_b)

        point_a = body_a.support_point_world(normal_world)
        point_b = body_b.support_point_world(-normal_world)
        return 0.5 * (point_a + point_b)

    def _support_extent_along_normal(
        self,
        body: RigidBodyState,
        normal_world: np.ndarray,
    ) -> float:
        local_normal = body.rotation_matrix().T @ normal_world
        return float(np.dot(np.abs(local_normal), body.half_extents))

    def _resolve_contacts(self, state: WorldState, dt: float) -> None:
        iterations = max(1, self.config.solver_iterations)
        for _ in range(iterations):
            for contact in state.contacts:
                self._solve_contact(contact, state, dt)

    def _solve_contact(self, contact: Contact, state: WorldState, dt: float) -> None:
        _ = dt
        body_a = state.get_body(contact.body_a)
        body_b = state.get_body(contact.body_b)

        inverse_mass_sum = body_a.inverse_mass + body_b.inverse_mass
        if inverse_mass_sum <= 0.0:
            return

        normal = safe_normalize(contact.normal)
        ra = contact.position - body_a.position
        rb = contact.position - body_b.position

        velocity_a = self._world_point_velocity(body_a, ra)
        velocity_b = self._world_point_velocity(body_b, rb)
        relative_velocity = velocity_b - velocity_a
        normal_speed = np.dot(relative_velocity, normal)

        restitution = min(body_a.restitution, body_b.restitution)
        if abs(normal_speed) < 0.5:
            restitution = 0.0
        if normal_speed < 0.0:
            impulse_magnitude = self._normal_impulse_magnitude(
                body_a,
                body_b,
                ra,
                rb,
                normal,
                normal_speed,
                restitution,
            )
            if impulse_magnitude > 0.0:
                impulse = impulse_magnitude * normal
                self._apply_contact_impulse(body_a, -impulse, ra)
                self._apply_contact_impulse(body_b, impulse, rb)
                contact.accumulated_normal_impulse += impulse_magnitude

                velocity_a = self._world_point_velocity(body_a, ra)
                velocity_b = self._world_point_velocity(body_b, rb)
                relative_velocity = velocity_b - velocity_a
                self._apply_friction_impulse(
                    body_a,
                    body_b,
                    ra,
                    rb,
                    relative_velocity,
                    normal,
                    impulse_magnitude,
                )

        self._apply_positional_correction(body_a, body_b, normal, contact.penetration_depth)

    def _world_point_velocity(self, body: RigidBodyState, relative_point: np.ndarray) -> np.ndarray:
        return body.linear_velocity + np.cross(body.angular_velocity, relative_point)

    def _normal_impulse_magnitude(
        self,
        body_a: RigidBodyState,
        body_b: RigidBodyState,
        ra: np.ndarray,
        rb: np.ndarray,
        normal: np.ndarray,
        normal_speed: float,
        restitution: float,
    ) -> float:
        angular_term_a = np.cross(body_a.inverse_inertia_world() @ np.cross(ra, normal), ra)
        angular_term_b = np.cross(body_b.inverse_inertia_world() @ np.cross(rb, normal), rb)
        effective_mass = (
            body_a.inverse_mass
            + body_b.inverse_mass
            + np.dot(normal, angular_term_a + angular_term_b)
        )
        if effective_mass <= 1e-8:
            return 0.0
        return max(0.0, -(1.0 + restitution) * normal_speed / effective_mass)

    def _apply_contact_impulse(
        self,
        body: RigidBodyState,
        impulse: np.ndarray,
        relative_point: np.ndarray,
    ) -> None:
        if not body.is_dynamic:
            return
        if body.is_sleeping and np.linalg.norm(impulse) > 1e-4:
            body.wake()
        body.linear_velocity = body.linear_velocity + impulse * body.inverse_mass
        body.angular_velocity = (
            body.angular_velocity
            + body.inverse_inertia_world() @ np.cross(relative_point, impulse)
        )

    def _apply_friction_impulse(
        self,
        body_a: RigidBodyState,
        body_b: RigidBodyState,
        ra: np.ndarray,
        rb: np.ndarray,
        relative_velocity: np.ndarray,
        normal: np.ndarray,
        normal_impulse_magnitude: float,
    ) -> None:
        tangential_velocity = relative_velocity - np.dot(relative_velocity, normal) * normal
        tangent_norm = np.linalg.norm(tangential_velocity)
        if tangent_norm < 1e-8:
            return

        tangent = tangential_velocity / tangent_norm
        angular_term_a = np.cross(body_a.inverse_inertia_world() @ np.cross(ra, tangent), ra)
        angular_term_b = np.cross(body_b.inverse_inertia_world() @ np.cross(rb, tangent), rb)
        effective_mass = (
            body_a.inverse_mass
            + body_b.inverse_mass
            + np.dot(tangent, angular_term_a + angular_term_b)
        )
        if effective_mass <= 1e-8:
            return

        jt = -np.dot(relative_velocity, tangent) / effective_mass
        friction_coefficient = min(body_a.friction, body_b.friction)
        max_friction = friction_coefficient * normal_impulse_magnitude
        jt = float(np.clip(jt, -max_friction, max_friction))
        if abs(jt) < 1e-10:
            return

        friction_impulse = jt * tangent
        self._apply_contact_impulse(body_a, -friction_impulse, ra)
        self._apply_contact_impulse(body_b, friction_impulse, rb)

    def _apply_positional_correction(
        self,
        body_a: RigidBodyState,
        body_b: RigidBodyState,
        normal: np.ndarray,
        penetration_depth: float,
    ) -> None:
        inverse_mass_sum = body_a.inverse_mass + body_b.inverse_mass
        if inverse_mass_sum <= 0.0:
            return

        slop = 1e-4
        percent = 0.8
        correction_magnitude = percent * max(penetration_depth - slop, 0.0) / inverse_mass_sum
        correction = correction_magnitude * normal

        if body_a.is_dynamic:
            body_a.position = body_a.position - correction * body_a.inverse_mass
        if body_b.is_dynamic:
            body_b.position = body_b.position + correction * body_b.inverse_mass

    def _integrate_positions(self, state: WorldState, dt: float) -> None:
        for body in state.bodies:
            if body.is_dynamic and not body.is_sleeping:
                self._integrate_body_pose(body, dt)

    def _stabilize_supported_bodies(self, state: WorldState, dt: float) -> None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        for body in state.bodies:
            if not body.is_dynamic or body.is_sleeping:
                continue

            support_normals: list[np.ndarray] = []
            for contact in state.contacts:
                if contact.body_a == body.body_id:
                    other_body = state.get_body(contact.body_b)
                    support_normal = -contact.normal
                elif contact.body_b == body.body_id:
                    other_body = state.get_body(contact.body_a)
                    support_normal = contact.normal
                else:
                    continue

                if not self._is_environment_boundary(other_body):
                    continue
                if np.dot(support_normal, up) > 0.5:
                    support_normals.append(support_normal)

            if len(support_normals) < 2:
                continue

            support_normal = safe_normalize(np.sum(support_normals, axis=0))
            normal_speed = float(np.dot(body.linear_velocity, support_normal))
            if normal_speed < 0.0:
                body.linear_velocity = body.linear_velocity - normal_speed * support_normal

            tangential_velocity = (
                body.linear_velocity
                - np.dot(body.linear_velocity, support_normal) * support_normal
            )
            tangential_speed = float(np.linalg.norm(tangential_velocity))
            if tangential_speed < 0.25:
                body.linear_velocity = body.linear_velocity - 0.35 * tangential_velocity

            angular_speed = float(np.linalg.norm(body.angular_velocity))
            if angular_speed < 1.5:
                body.angular_velocity *= max(0.0, 1.0 - 18.0 * dt)

    def _update_sleep_states(self, state: WorldState) -> None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        for body in state.bodies:
            if not body.is_dynamic:
                continue

            support_contact_count = 0
            for contact in state.contacts:
                if contact.body_a == body.body_id:
                    support_normal = -contact.normal
                elif contact.body_b == body.body_id:
                    support_normal = contact.normal
                else:
                    continue
                if np.dot(support_normal, up) > 0.5:
                    support_contact_count += 1

            linear_speed = np.linalg.norm(body.linear_velocity)
            angular_speed = np.linalg.norm(body.angular_velocity)
            low_energy = linear_speed < 0.35 and angular_speed < 0.35
            enough_support = support_contact_count >= 2

            if enough_support and low_energy:
                body.sleep_counter += 1
                if body.sleep_counter >= 15:
                    body.sleep()
            else:
                body.sleep_counter = 0
                if body.is_sleeping and (
                    not enough_support
                    or linear_speed > 0.05
                    or angular_speed > 0.05
                ):
                    body.wake()

    def _integrate_body_pose(self, body: RigidBodyState, dt: float) -> None:
        body.position = body.position + dt * body.linear_velocity
        body.orientation = integrate_quat_wxyz(body.orientation, body.angular_velocity, dt)

    def _end_step(self, state: WorldState, dt: float) -> None:
        state.time += dt
        state.frame += 1
