from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim.world import RigidBodyWorld


FACE_INDICES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (1, 2, 6, 5),
    (0, 3, 7, 4),
)


def _project(points: np.ndarray, scale: float, center: np.ndarray) -> np.ndarray:
    projected = np.empty((points.shape[0], 2), dtype=np.float64)
    projected[:, 0] = points[:, 0] - 0.55 * points[:, 2]
    projected[:, 1] = 1.05 * points[:, 1] + 0.28 * points[:, 2]
    projected[:, 0] = center[0] + scale * projected[:, 0]
    projected[:, 1] = center[1] - scale * projected[:, 1]
    return projected


def _shade(color: np.ndarray, factor: float) -> tuple[int, int, int]:
    rgb = np.clip(np.asarray(color, dtype=np.float64) * factor * 255.0, 0.0, 255.0)
    return tuple(int(x) for x in rgb)


def _draw_body(draw: ImageDraw.ImageDraw, body, scale: float, center: np.ndarray) -> None:
    corners = body.world_corners()
    projected = _project(corners, scale=scale, center=center)
    face_depths = []
    for face in FACE_INDICES:
        face_points = corners[list(face)]
        face_depths.append((float(np.mean(face_points[:, 1] + 0.35 * face_points[:, 2])), face))
    face_depths.sort()

    shades = (0.78, 0.95, 0.88, 1.02, 0.92, 0.84)
    for (_, face), shade in zip(face_depths, shades, strict=True):
        polygon = [tuple(projected[idx]) for idx in face]
        draw.polygon(polygon, fill=_shade(body.color, shade), outline=(24, 26, 30))


def _draw_frame(world: RigidBodyWorld, size: tuple[int, int] = (720, 480)) -> Image.Image:
    image = Image.new("RGB", size, color=(243, 245, 248))
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = size
    center = np.array([width * 0.5, height * 0.62], dtype=np.float64)

    dynamic_bodies = [body for body in world.state.bodies if body.is_dynamic]
    positions = np.stack([body.position for body in world.state.bodies], axis=0)
    span = np.max(positions, axis=0) - np.min(positions, axis=0)
    scale = 175.0 / max(2.2, float(max(span[0], span[1], span[2], 1.0)))
    if world.state.active_demo == "complex_scene":
        scale *= 0.92

    draw.rectangle([(0, height * 0.72), (width, height)], fill=(228, 232, 238))
    draw.text((24, 20), f"Demo: {world.state.active_demo}", fill=(30, 34, 42))
    draw.text((24, 44), f"Frame: {world.state.frame}", fill=(30, 34, 42))
    draw.text((24, 68), f"Contacts: {len(world.state.contacts)}", fill=(30, 34, 42))

    ordered_bodies = sorted(
        world.state.bodies,
        key=lambda body: float(body.position[1] + 0.25 * body.position[2]),
    )
    for body in ordered_bodies:
        _draw_body(draw, body, scale=scale, center=center)

    if dynamic_bodies:
        body = dynamic_bodies[0]
        draw.text((24, 92), f"Body: {body.name}", fill=(30, 34, 42))
    return image


def _generate_demo(
    *,
    demo_name: str,
    output_path: Path,
    steps: int,
    step_stride: int = 1,
    collision_case: str | None = None,
) -> None:
    world = RigidBodyWorld(demo_name=demo_name)
    if collision_case is not None:
        world.select_two_body_case(collision_case)

    frames: list[Image.Image] = []
    for step in range(steps):
        if step % step_stride == 0:
            frames.append(_draw_frame(world))
        world.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        disposal=2,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "assets" / "gifs"
    _generate_demo(
        demo_name="single_body",
        output_path=output_dir / "demo1_single_body.gif",
        steps=96,
        step_stride=2,
    )
    _generate_demo(
        demo_name="two_body_collision",
        collision_case="face_face",
        output_path=output_dir / "demo2_two_body_collision.gif",
        steps=96,
        step_stride=2,
    )
    _generate_demo(
        demo_name="complex_scene",
        output_path=output_dir / "demo3_complex_scene.gif",
        steps=120,
        step_stride=2,
    )


if __name__ == "__main__":
    main()
