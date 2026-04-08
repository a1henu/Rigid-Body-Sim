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
