import numpy as np

try:
    import taichi as ti
    import taichi.lang.impl as ti_impl
except ImportError:  # pragma: no cover
    ti = None
    ti_impl = None


def taichi_step_available() -> bool:
    return ti is not None and ti_impl is not None


def ensure_taichi_step_runtime() -> bool:
    if not taichi_step_available():
        return False
    if ti_impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
    return True


if ti is not None:

    @ti.func
    def _quat_normalize(q):
        return q / ti.max(q.norm(), 1e-8)


    @ti.func
    def _quat_mul(q1, q2):
        return ti.Vector(
            [
                q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
                q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
            ]
        )


    @ti.func
    def _quat_to_mat3(q):
        qn = _quat_normalize(q)
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
        return ti.Matrix(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ]
        )


    @ti.func
    def _inverse_inertia_world(orientation, inverse_inertia_diag):
        rotation = _quat_to_mat3(orientation)
        inverse_inertia_body = ti.Matrix(
            [
                [inverse_inertia_diag[0], 0.0, 0.0],
                [0.0, inverse_inertia_diag[1], 0.0],
                [0.0, 0.0, inverse_inertia_diag[2]],
            ]
        )
        return rotation @ inverse_inertia_body @ rotation.transpose()


    @ti.kernel
    def integrate_velocities_kernel(
        body_count: int,
        dt: float,
        gravity_x: float,
        gravity_y: float,
        gravity_z: float,
        linear_damping: float,
        angular_damping: float,
        dynamic_mask: ti.types.ndarray(dtype=ti.i32, ndim=1),
        sleep_mask: ti.types.ndarray(dtype=ti.i32, ndim=1),
        inverse_mass: ti.types.ndarray(dtype=ti.f32, ndim=1),
        inverse_inertia_diag: ti.types.ndarray(dtype=ti.f32, ndim=2),
        orientation: ti.types.ndarray(dtype=ti.f32, ndim=2),
        linear_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        angular_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        force: ti.types.ndarray(dtype=ti.f32, ndim=2),
        torque: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        gravity = ti.Vector([gravity_x, gravity_y, gravity_z])
        for i in range(body_count):
            if dynamic_mask[i] == 1:
                force[i, 0] = force[i, 0] + gravity[0] / ti.max(inverse_mass[i], 1e-8)
                force[i, 1] = force[i, 1] + gravity[1] / ti.max(inverse_mass[i], 1e-8)
                force[i, 2] = force[i, 2] + gravity[2] / ti.max(inverse_mass[i], 1e-8)

            if dynamic_mask[i] == 1 and sleep_mask[i] == 0:
                linear_acceleration = ti.Vector(
                    [force[i, 0], force[i, 1], force[i, 2]]
                ) * inverse_mass[i]
                inverse_inertia = _inverse_inertia_world(
                    ti.Vector(
                        [
                            orientation[i, 0],
                            orientation[i, 1],
                            orientation[i, 2],
                            orientation[i, 3],
                        ]
                    ),
                    ti.Vector(
                        [
                            inverse_inertia_diag[i, 0],
                            inverse_inertia_diag[i, 1],
                            inverse_inertia_diag[i, 2],
                        ]
                    ),
                )
                angular_acceleration = inverse_inertia @ ti.Vector(
                    [torque[i, 0], torque[i, 1], torque[i, 2]]
                )

                linear_velocity[i, 0] = (
                    linear_velocity[i, 0] + dt * linear_acceleration[0]
                ) * ti.max(0.0, 1.0 - linear_damping * dt)
                linear_velocity[i, 1] = (
                    linear_velocity[i, 1] + dt * linear_acceleration[1]
                ) * ti.max(0.0, 1.0 - linear_damping * dt)
                linear_velocity[i, 2] = (
                    linear_velocity[i, 2] + dt * linear_acceleration[2]
                ) * ti.max(0.0, 1.0 - linear_damping * dt)

                angular_velocity[i, 0] = (
                    angular_velocity[i, 0] + dt * angular_acceleration[0]
                ) * ti.max(0.0, 1.0 - angular_damping * dt)
                angular_velocity[i, 1] = (
                    angular_velocity[i, 1] + dt * angular_acceleration[1]
                ) * ti.max(0.0, 1.0 - angular_damping * dt)
                angular_velocity[i, 2] = (
                    angular_velocity[i, 2] + dt * angular_acceleration[2]
                ) * ti.max(0.0, 1.0 - angular_damping * dt)


    @ti.kernel
    def integrate_positions_kernel(
        body_count: int,
        dt: float,
        dynamic_mask: ti.types.ndarray(dtype=ti.i32, ndim=1),
        sleep_mask: ti.types.ndarray(dtype=ti.i32, ndim=1),
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        orientation: ti.types.ndarray(dtype=ti.f32, ndim=2),
        linear_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        angular_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i in range(body_count):
            if dynamic_mask[i] == 1 and sleep_mask[i] == 0:
                position[i, 0] = position[i, 0] + dt * linear_velocity[i, 0]
                position[i, 1] = position[i, 1] + dt * linear_velocity[i, 1]
                position[i, 2] = position[i, 2] + dt * linear_velocity[i, 2]

                omega_quat = ti.Vector(
                    [0.0, angular_velocity[i, 0], angular_velocity[i, 1], angular_velocity[i, 2]]
                )
                q = ti.Vector([orientation[i, 0], orientation[i, 1], orientation[i, 2], orientation[i, 3]])
                q_dot = 0.5 * _quat_mul(omega_quat, q)
                q_next = _quat_normalize(q + dt * q_dot)
                orientation[i, 0] = q_next[0]
                orientation[i, 1] = q_next[1]
                orientation[i, 2] = q_next[2]
                orientation[i, 3] = q_next[3]


class TaichiStepBuffers:
    def __init__(self, body_capacity: int):
        self.body_capacity = max(1, int(body_capacity))
        self.dynamic_mask = np.zeros(self.body_capacity, dtype=np.int32)
        self.sleep_mask = np.zeros(self.body_capacity, dtype=np.int32)
        self.inverse_mass = np.zeros(self.body_capacity, dtype=np.float32)
        self.inverse_inertia_diag = np.zeros((self.body_capacity, 3), dtype=np.float32)
        self.position = np.zeros((self.body_capacity, 3), dtype=np.float32)
        self.orientation = np.zeros((self.body_capacity, 4), dtype=np.float32)
        self.linear_velocity = np.zeros((self.body_capacity, 3), dtype=np.float32)
        self.angular_velocity = np.zeros((self.body_capacity, 3), dtype=np.float32)
        self.force = np.zeros((self.body_capacity, 3), dtype=np.float32)
        self.torque = np.zeros((self.body_capacity, 3), dtype=np.float32)

    def load_from_state(self, state) -> int:
        body_count = len(state.bodies)
        self.dynamic_mask.fill(0)
        self.sleep_mask.fill(0)
        self.inverse_mass.fill(0.0)
        self.inverse_inertia_diag.fill(0.0)
        self.position.fill(0.0)
        self.orientation.fill(0.0)
        self.linear_velocity.fill(0.0)
        self.angular_velocity.fill(0.0)
        self.force.fill(0.0)
        self.torque.fill(0.0)

        for body_id, body in enumerate(state.bodies):
            self.dynamic_mask[body_id] = int(body.is_dynamic)
            self.sleep_mask[body_id] = int(body.is_sleeping)
            self.inverse_mass[body_id] = np.float32(body.inverse_mass)
            self.inverse_inertia_diag[body_id] = np.diag(body.inverse_inertia_body).astype(np.float32)
            self.position[body_id] = body.position.astype(np.float32)
            self.orientation[body_id] = body.orientation.astype(np.float32)
            self.linear_velocity[body_id] = body.linear_velocity.astype(np.float32)
            self.angular_velocity[body_id] = body.angular_velocity.astype(np.float32)
            self.force[body_id] = body.force_accumulator.astype(np.float32)
            self.torque[body_id] = body.torque_accumulator.astype(np.float32)
        return body_count

    def store_motion_to_state(self, state) -> None:
        for body_id, body in enumerate(state.bodies):
            body.position = self.position[body_id].astype(np.float64)
            body.orientation = self.orientation[body_id].astype(np.float64)
            body.linear_velocity = self.linear_velocity[body_id].astype(np.float64)
            body.angular_velocity = self.angular_velocity[body_id].astype(np.float64)
