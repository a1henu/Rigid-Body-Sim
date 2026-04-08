from __future__ import annotations

from typing import Iterable

from sim.state import CommandType, Contact, RigidBodyState, SimulationConfig, WorldState


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
                # TODO: if world_point is provided, convert off-center force to torque.
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
        # TODO: explicit Euler update for linear/angular velocity.
        _ = body, dt

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
        # TODO: SAT/FCL-based box-box collision detection goes here.
        _ = body_a, body_b
        return []

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
        # TODO: explicit Euler update for position and quaternion orientation.
        _ = body, dt

    def _end_step(self, state: WorldState, dt: float) -> None:
        state.time += dt
        state.frame += 1
