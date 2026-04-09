import numpy as np

from sim.world import RigidBodyWorld


def test_list_two_body_cases_contains_expected_presets():
    world = RigidBodyWorld(demo_name="single_body")

    assert world.list_two_body_cases() == (
        "point_face",
        "edge_edge",
        "face_face",
        "random_pose",
    )


def test_select_two_body_case_reloads_demo_with_same_box_sizes():
    world = RigidBodyWorld(demo_name="two_body_collision")

    world.select_two_body_case("edge_edge")

    assert world.state.active_demo == "two_body_collision"
    assert "Case=edge_edge" in world.state.demo_description
    assert len(world.state.bodies) == 2
    np.testing.assert_allclose(world.state.bodies[0].half_extents, [0.35, 0.35, 0.35])
    np.testing.assert_allclose(world.state.bodies[1].half_extents, [0.35, 0.35, 0.35])


def test_reset_active_demo_restores_initial_body_snapshot():
    world = RigidBodyWorld(demo_name="single_body")
    initial_position = world.state.bodies[0].position.copy()
    initial_velocity = world.state.bodies[0].linear_velocity.copy()

    for _ in range(10):
        world.step()
    world.apply_force_to_body(0, [10.0, 0.0, 0.0])
    world.step()

    world.reset_active_demo()

    np.testing.assert_allclose(world.state.bodies[0].position, initial_position, atol=1e-10)
    np.testing.assert_allclose(world.state.bodies[0].linear_velocity, initial_velocity, atol=1e-10)
    assert world.state.frame == 0
    assert world.state.time == 0.0


def test_next_demo_cycles_through_registered_demos():
    world = RigidBodyWorld(demo_name="single_body")

    world.next_demo()
    assert world.state.active_demo == "two_body_collision"

    world.next_demo()
    assert world.state.active_demo == "complex_scene"

    world.next_demo()
    assert world.state.active_demo == "single_body"


def test_complex_scene_configures_gravity_boundaries_and_four_dynamic_boxes():
    world = RigidBodyWorld(demo_name="complex_scene")

    assert world.config.enable_gravity is True
    assert world.config.substeps == 4
    assert world.config.solver_iterations == 8
    assert len(world.state.bodies) == 7

    static_bodies = world.state.bodies[:3]
    dynamic_bodies = world.state.bodies[3:]

    assert all(not body.is_dynamic for body in static_bodies)
    assert all(body.is_dynamic for body in dynamic_bodies)
    assert all(body.user_data.get("environment_boundary", False) for body in static_bodies)
    assert world.get_selected_body().name == "ring_a"


def test_complex_scene_selection_cycles_over_dynamic_bodies_only():
    world = RigidBodyWorld(demo_name="complex_scene")

    names = []
    for _ in range(5):
        names.append(world.get_selected_body().name)
        world.select_next_dynamic_body()

    assert names == ["ring_a", "ring_b", "ring_c", "ring_d", "ring_a"]


def test_complex_scene_initializes_boxes_around_circle_with_inward_velocity():
    world = RigidBodyWorld(demo_name="complex_scene")

    dynamic_bodies = world.state.bodies[3:]
    radii = []
    inward_speeds = []
    for body in dynamic_bodies:
        horizontal_position = body.position[[0, 2]]
        horizontal_velocity = body.linear_velocity[[0, 2]]
        radii.append(float(np.linalg.norm(horizontal_position)))
        inward_speeds.append(float(np.dot(horizontal_velocity, -horizontal_position)))

    assert all(1.0 < radius < 1.35 for radius in radii)
    assert all(speed > 1.3 for speed in inward_speeds)


def test_complex_scene_produces_contacts_early_without_immediate_fallthrough():
    world = RigidBodyWorld(demo_name="complex_scene")
    contact_frames = 0

    for _ in range(60):
        world.step()
        if world.state.contacts:
            contact_frames += 1

    assert contact_frames > 0
    for body in world.state.bodies[3:]:
        bottom = body.position[1] - body.half_extents[1]
        assert bottom > -1.2


def test_complex_scene_produces_dynamic_body_collisions_near_center():
    world = RigidBodyWorld(demo_name="complex_scene")
    dynamic_collision_frames = 0

    for _ in range(60):
        world.step()
        for contact in world.state.contacts:
            body_a = world.state.get_body(contact.body_a)
            body_b = world.state.get_body(contact.body_b)
            if body_a.is_dynamic and body_b.is_dynamic:
                dynamic_collision_frames += 1
                break

    assert dynamic_collision_frames > 0


def test_complex_scene_settles_without_explosive_jitter():
    world = RigidBodyWorld(demo_name="complex_scene")

    for _ in range(240):
        world.step()

    dynamic_bodies = world.state.bodies[3:]
    assert any(body.is_sleeping for body in dynamic_bodies)
    for body in dynamic_bodies:
        assert np.linalg.norm(body.linear_velocity) < 1.0
        assert np.linalg.norm(body.angular_velocity) < 2.5


def test_complex_scene_impulse_interaction_changes_selected_body_velocity():
    world = RigidBodyWorld(demo_name="complex_scene")
    selected_body = world.get_selected_body()
    initial_velocity_x = float(selected_body.linear_velocity[0])

    world.apply_impulse_to_body(selected_body.body_id, [0.08, 0.0, 0.0])
    world.step()

    updated_body = world.state.get_body(selected_body.body_id)
    assert float(updated_body.linear_velocity[0]) > initial_velocity_x
