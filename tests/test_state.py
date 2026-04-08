import numpy as np

from sim.state import (
    MotionType,
    box_inertia_tensor,
    create_box_body,
    quat_identity_wxyz,
)


def test_box_inertia_tensor_matches_closed_form_diagonal():
    mass = 2.0
    half_extents = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    inertia, inverse_inertia = box_inertia_tensor(mass, half_extents)

    expected_diag = np.array(
        [
            mass / 3.0 * (2.0**2 + 3.0**2),
            mass / 3.0 * (1.0**2 + 3.0**2),
            mass / 3.0 * (1.0**2 + 2.0**2),
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(np.diag(inertia), expected_diag, atol=1e-10)
    np.testing.assert_allclose(np.diag(inverse_inertia), 1.0 / expected_diag, atol=1e-10)


def test_create_box_body_static_has_zero_inverse_mass_and_inertia():
    body = create_box_body(
        name="floor",
        half_extents=[1.0, 0.2, 1.0],
        motion_type=MotionType.STATIC,
    )

    assert body.mass == 0.0
    assert body.inverse_mass == 0.0
    np.testing.assert_allclose(body.inverse_inertia_body, np.zeros((3, 3)))


def test_support_point_world_matches_axis_aligned_box_extreme_corner():
    body = create_box_body(
        name="box",
        half_extents=[0.5, 0.25, 0.75],
        position=[1.0, 2.0, 3.0],
        orientation=quat_identity_wxyz(),
    )

    support = body.support_point_world([1.0, -1.0, 1.0])

    np.testing.assert_allclose(support, np.array([1.5, 1.75, 3.75]), atol=1e-10)
