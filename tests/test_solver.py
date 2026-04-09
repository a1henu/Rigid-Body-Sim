import numpy as np

from sim.solver import RigidBodySolver
from sim.state import Contact, MotionType, SimulationConfig, WorldState, create_box_body
from sim.world import RigidBodyWorld


def test_integrate_body_velocity_and_pose_updates_single_body_state():
    config = SimulationConfig(enable_collisions=False, linear_damping=0.0, angular_damping=0.0)
    solver = RigidBodySolver(config)
    body = create_box_body(
        name="body",
        half_extents=[0.5, 0.5, 0.5],
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        mass=2.0,
    )
    body.force_accumulator[:] = np.array([4.0, 0.0, 0.0])

    solver._integrate_body_velocity(body, 0.5)
    solver._integrate_body_pose(body, 0.5)

    np.testing.assert_allclose(body.linear_velocity, np.array([1.0, 0.0, 0.0]), atol=1e-10)
    np.testing.assert_allclose(body.position, np.array([0.5, 0.0, 0.0]), atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(body.orientation), 1.0, atol=1e-10)


def test_detect_box_box_sat_returns_contact_for_overlapping_boxes():
    solver = RigidBodySolver(SimulationConfig())
    body_a = create_box_body(
        name="a",
        half_extents=[0.5, 0.5, 0.5],
        position=[0.0, 0.0, 0.0],
    )
    body_b = create_box_body(
        name="b",
        half_extents=[0.5, 0.5, 0.5],
        position=[0.8, 0.0, 0.0],
    )
    body_a.body_id = 0
    body_b.body_id = 1

    contact = solver._detect_box_box_sat(body_a, body_b)

    assert contact is not None
    assert isinstance(contact, Contact)
    assert contact.body_a == 0
    assert contact.body_b == 1
    assert contact.penetration_depth > 0.0
    np.testing.assert_allclose(contact.normal, np.array([1.0, 0.0, 0.0]), atol=1e-8)


def test_face_face_collision_reverses_velocities_without_spurious_spin():
    world = RigidBodyWorld(demo_name="two_body_collision")
    world.select_two_body_case("face_face")

    for _ in range(47):
        world.step()

    left_body, right_body = world.state.bodies

    assert left_body.linear_velocity[0] < 0.0
    assert right_body.linear_velocity[0] > 0.0
    np.testing.assert_allclose(left_body.angular_velocity, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(right_body.angular_velocity, np.zeros(3), atol=1e-10)


def test_step_populates_contacts_for_overlapping_world_state():
    config = SimulationConfig(time_step=1.0 / 60.0, enable_collisions=True)
    solver = RigidBodySolver(config)
    state = WorldState(
        bodies=[
            create_box_body(name="a", half_extents=[0.5, 0.5, 0.5], position=[0.0, 0.0, 0.0]),
            create_box_body(name="b", half_extents=[0.5, 0.5, 0.5], position=[0.9, 0.0, 0.0]),
        ]
    )
    state.bodies[0].body_id = 0
    state.bodies[1].body_id = 1

    solver.step(state, config.time_step)

    assert len(state.contacts) == 1


def test_boundary_collision_generates_multiple_support_contacts():
    solver = RigidBodySolver(SimulationConfig(enable_gravity=False, enable_collisions=True))
    floor = create_box_body(
        name="floor",
        half_extents=[4.0, 0.2, 4.0],
        position=[0.0, -1.25, 0.0],
        motion_type=MotionType.STATIC,
        user_data={"environment_boundary": True},
    )
    box = create_box_body(
        name="box",
        half_extents=[0.3, 0.3, 0.3],
        position=[0.0, -0.73, 0.0],
        orientation=[0.9961947, 0.0, 0.0, 0.0871557],
    )
    floor.body_id = 0
    box.body_id = 1
    state = WorldState(bodies=[floor, box])

    contacts = solver._detect_collisions(state)

    assert len(contacts) >= 2
    assert all(contact.body_a == 1 and contact.body_b == 0 for contact in contacts)
    assert all(contact.feature_id.startswith("boundary:") for contact in contacts)


def test_box_resting_on_floor_enters_sleep_state():
    config = SimulationConfig(
        time_step=1.0 / 240.0,
        enable_gravity=True,
        enable_collisions=True,
        solver_iterations=10,
    )
    solver = RigidBodySolver(config)
    floor = create_box_body(
        name="floor",
        half_extents=[3.0, 0.2, 3.0],
        position=[0.0, -1.0, 0.0],
        motion_type=MotionType.STATIC,
        restitution=0.0,
        friction=0.9,
        user_data={"environment_boundary": True},
    )
    box = create_box_body(
        name="box",
        half_extents=[0.3, 0.3, 0.3],
        position=[0.0, 0.5, 0.0],
        mass=1.0,
        restitution=0.0,
        friction=0.8,
    )
    floor.body_id = 0
    box.body_id = 1
    state = WorldState(bodies=[floor, box])

    for _ in range(600):
        solver.step(state, config.time_step)

    assert box.is_sleeping
    np.testing.assert_allclose(box.linear_velocity, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(box.angular_velocity, np.zeros(3), atol=1e-10)
