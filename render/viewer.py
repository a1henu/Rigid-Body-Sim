from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from sim.world import RigidBodyWorld

try:
    import taichi as ti
except ImportError:
    ti = None


BOX_EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)


@dataclass(slots=True)
class OrbitCamera:
    position: np.ndarray = field(
        default_factory=lambda: np.array([3.6, 2.4, 4.8], dtype=np.float64)
    )
    look_at: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )
    up_hint: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    fov_y_degrees: float = 50.0


class RigidBodyViewer:
    """
    Simple viewer for incremental lab development.

    When Taichi is available, the viewer opens a 2D GUI window and renders a
    perspective-projected wireframe of each box. When Taichi is unavailable, it
    falls back to periodic console logging so the rest of the framework remains
    runnable in headless environments.
    """

    def __init__(
        self,
        world: RigidBodyWorld,
        *,
        realtime: bool = False,
        resolution: tuple[int, int] = (1280, 720),
        force_strength: float = 12.0,
    ):
        self.world = world
        self.realtime = realtime
        self.resolution = resolution
        self.force_strength = force_strength
        self.camera = OrbitCamera()
        self.gui = None

        if ti is not None:
            ti.init(arch=ti.cpu, default_fp=ti.f64)
            self.gui = ti.GUI(
                name="Rigid Body Lab",
                res=resolution,
                background_color=0x11151C,
            )

    def run(self, max_frames: int | None = None, print_every: int = 30) -> None:
        if self.gui is None:
            self._run_console(max_frames=max_frames, print_every=print_every)
            return

        frame_idx = 0
        while self.gui.running and (max_frames is None or frame_idx < max_frames):
            self._handle_gui_events()
            self.world.step()
            self._draw_gui_frame()
            frame_idx += 1
            if self.realtime:
                time.sleep(self.world.config.time_step)

    def _run_console(self, max_frames: int | None, print_every: int) -> None:
        total_frames = 180 if max_frames is None else max_frames
        for frame_idx in range(total_frames):
            self.world.step()
            if frame_idx % max(1, print_every) == 0:
                self._draw_frame()
            if self.realtime:
                time.sleep(self.world.config.time_step)

    def _draw_frame(self) -> None:
        print(self.world.describe())
        print("-" * 60)

    def _handle_gui_events(self) -> None:
        assert self.gui is not None
        self.gui.get_event()
        if self.gui.event is not None and self.gui.event.type == ti.GUI.PRESS:
            if self.gui.event.key in (ti.GUI.ESCAPE, ti.GUI.EXIT):
                self.gui.running = False
                return
            if self.gui.event.key == "r":
                self.world.reset_active_demo()
            elif self.gui.event.key == "p":
                self.world.toggle_pause()
            elif self.gui.event.key == "1":
                self.world.load_demo("single_body")
            elif self.gui.event.key == "2":
                self.world.load_demo("two_body_collision")
            elif self.gui.event.key == "3":
                self.world.load_demo("complex_scene")

        primary_body_id = self.world.get_primary_body_id()
        if primary_body_id is None:
            return

        force = np.zeros(3, dtype=np.float64)
        if self.gui.is_pressed("a", ti.GUI.LEFT):
            force[0] -= self.force_strength
        if self.gui.is_pressed("d", ti.GUI.RIGHT):
            force[0] += self.force_strength
        if self.gui.is_pressed("w", ti.GUI.UP):
            force[1] += self.force_strength
        if self.gui.is_pressed("s", ti.GUI.DOWN):
            force[1] -= self.force_strength
        if self.gui.is_pressed("q"):
            force[2] += self.force_strength
        if self.gui.is_pressed("e"):
            force[2] -= self.force_strength

        if np.any(force):
            self.world.apply_force_to_body(primary_body_id, force)

    def _draw_gui_frame(self) -> None:
        assert self.gui is not None
        self.gui.clear(0x11151C)

        line_begins: list[np.ndarray] = []
        line_ends: list[np.ndarray] = []
        point_positions: list[np.ndarray] = []
        point_palette_indices: list[int] = []

        for body in self.world.state.bodies:
            corners = body.world_corners()
            projected = self._project_points(corners)
            visible_mask = np.isfinite(projected).all(axis=1)
            for edge in BOX_EDGES:
                if visible_mask[edge[0]] and visible_mask[edge[1]]:
                    line_begins.append(projected[edge[0]])
                    line_ends.append(projected[edge[1]])

            center_proj = self._project_points(body.position[None, :])[0]
            if np.isfinite(center_proj).all():
                point_positions.append(center_proj)
                point_palette_indices.append(0 if body.is_dynamic else 1)

        if line_begins:
            self.gui.lines(
                begin=np.asarray(line_begins, dtype=np.float64),
                end=np.asarray(line_ends, dtype=np.float64),
                radius=1.8,
                color=0xD7E3F4,
            )

        if point_positions:
            self.gui.circles(
                pos=np.asarray(point_positions, dtype=np.float64),
                radius=5,
                palette=[0xF29E4C, 0x7D8597],
                palette_indices=np.asarray(point_palette_indices, dtype=np.int32),
            )

        self._draw_overlay()
        self.gui.show()

    def _draw_overlay(self) -> None:
        assert self.gui is not None
        lines = [
            f"Demo: {self.world.state.active_demo}",
            f"Frame: {self.world.state.frame}",
            f"Time: {self.world.state.time:.3f}",
            "[WASD/QE] apply force to primary body",
            "[P] pause  [R] reset  [1/2/3] switch demo  [Esc] quit",
        ]

        primary_body_id = self.world.get_primary_body_id()
        if primary_body_id is not None:
            body = self.world.state.get_body(primary_body_id)
            lines.append(
                "Body vel: "
                + np.array2string(body.linear_velocity, precision=2, suppress_small=True)
            )
            lines.append(
                "Body ang vel: "
                + np.array2string(body.angular_velocity, precision=2, suppress_small=True)
            )

        y = 0.97
        for line in lines:
            self.gui.text(line, pos=(0.02, y), font_size=18, color=0xF8F9FA)
            y -= 0.04

    def _project_points(self, points_world: np.ndarray) -> np.ndarray:
        width, height = self.resolution
        aspect = width / height
        fov_y = np.deg2rad(self.camera.fov_y_degrees)

        forward = self.camera.look_at - self.camera.position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.camera.up_hint)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        relative = np.asarray(points_world, dtype=np.float64) - self.camera.position
        camera_x = relative @ right
        camera_y = relative @ up
        camera_z = relative @ forward

        projected = np.full((relative.shape[0], 2), np.nan, dtype=np.float64)
        valid = camera_z > 1e-5
        if not np.any(valid):
            return projected

        tan_half_fov_y = np.tan(0.5 * fov_y)
        tan_half_fov_x = tan_half_fov_y * aspect

        projected[valid, 0] = 0.5 + 0.5 * (camera_x[valid] / (camera_z[valid] * tan_half_fov_x))
        projected[valid, 1] = 0.5 + 0.5 * (camera_y[valid] / (camera_z[valid] * tan_half_fov_y))
        return projected
