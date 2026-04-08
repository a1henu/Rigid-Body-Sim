from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from sim.world import RigidBodyWorld

try:
    import taichi as ti
except ImportError:
    ti = None


_TAICHI_RUNTIME_READY = False
_TAICHI_ARCH_NAME = "unavailable"

LOCAL_BOX_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

LOCAL_BOX_NORMALS = LOCAL_BOX_CORNERS / np.linalg.norm(LOCAL_BOX_CORNERS, axis=1, keepdims=True)

BOX_TRIANGLES = np.array(
    [
        0,
        1,
        2,
        0,
        2,
        3,
        4,
        6,
        5,
        4,
        7,
        6,
        0,
        4,
        5,
        0,
        5,
        1,
        1,
        5,
        6,
        1,
        6,
        2,
        2,
        6,
        7,
        2,
        7,
        3,
        3,
        7,
        4,
        3,
        4,
        0,
    ],
    dtype=np.int32,
)


def _ensure_taichi_runtime() -> None:
    global _TAICHI_RUNTIME_READY, _TAICHI_ARCH_NAME
    if ti is None or _TAICHI_RUNTIME_READY:
        return

    init_attempts = (
        ("gpu", getattr(ti, "gpu", None)),
        ("cpu", ti.cpu),
    )
    last_error = None
    for arch_name, arch in init_attempts:
        if arch is None:
            continue
        try:
            ti.init(arch=arch, default_fp=ti.f32)
            _TAICHI_RUNTIME_READY = True
            _TAICHI_ARCH_NAME = arch_name
            return
        except Exception as exc:  # pragma: no cover - depends on local runtime setup.
            last_error = exc

    raise RuntimeError("failed to initialize Taichi runtime") from last_error


if ti is not None:

    @ti.data_oriented
    class TaichiSceneBuffers:
        vertices_per_body = 8
        triangle_index_count_per_body = 36

        def __init__(self, body_capacity: int):
            self.body_capacity = max(1, int(body_capacity))
            self.active_body_count = ti.field(dtype=ti.i32, shape=())
            self.body_positions = ti.Vector.field(3, dtype=ti.f32, shape=self.body_capacity)
            self.body_orientations = ti.Vector.field(4, dtype=ti.f32, shape=self.body_capacity)
            self.body_half_extents = ti.Vector.field(3, dtype=ti.f32, shape=self.body_capacity)
            self.body_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.body_capacity)
            self.center_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.body_capacity)

            vertex_count = self.body_capacity * self.vertices_per_body
            index_count = self.body_capacity * self.triangle_index_count_per_body

            self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
            self.normals = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
            self.vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
            self.centers = ti.Vector.field(3, dtype=ti.f32, shape=self.body_capacity)
            self.indices = ti.field(dtype=ti.i32, shape=index_count)

            self.local_corners = ti.Vector.field(3, dtype=ti.f32, shape=self.vertices_per_body)
            self.local_normals = ti.Vector.field(3, dtype=ti.f32, shape=self.vertices_per_body)

            self.local_corners.from_numpy(LOCAL_BOX_CORNERS)
            self.local_normals.from_numpy(LOCAL_BOX_NORMALS.astype(np.float32))
            self.indices.from_numpy(self._build_index_buffer(self.body_capacity))

        @staticmethod
        def _build_index_buffer(body_capacity: int) -> np.ndarray:
            indices = np.empty(
                body_capacity * TaichiSceneBuffers.triangle_index_count_per_body,
                dtype=np.int32,
            )
            for body_id in range(body_capacity):
                vertex_offset = body_id * TaichiSceneBuffers.vertices_per_body
                dst_begin = body_id * TaichiSceneBuffers.triangle_index_count_per_body
                indices[dst_begin : dst_begin + TaichiSceneBuffers.triangle_index_count_per_body] = (
                    BOX_TRIANGLES + vertex_offset
                )
            return indices

        @ti.func
        def _quat_to_mat3(self, q):
            qn = q / ti.max(q.norm(), 1e-8)
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
        def _safe_normalize(self, v):
            return v / ti.max(v.norm(), 1e-8)

        @ti.kernel
        def update_transforms(self):
            for body_id in range(self.active_body_count[None]):
                position = self.body_positions[body_id]
                half_extents = self.body_half_extents[body_id]
                rotation = self._quat_to_mat3(self.body_orientations[body_id])
                color = self.body_colors[body_id]
                self.centers[body_id] = position
                for corner_id in ti.static(range(8)):
                    vertex_id = body_id * self.vertices_per_body + corner_id
                    local_corner = self.local_corners[corner_id] * half_extents
                    self.vertices[vertex_id] = position + rotation @ local_corner
                    self.normals[vertex_id] = self._safe_normalize(
                        rotation @ self.local_normals[corner_id]
                    )
                    self.vertex_colors[vertex_id] = color


@dataclass(slots=True)
class ViewerStats:
    fps_smoothed: float = 0.0

    def update(self, frame_dt: float) -> None:
        fps = 1.0 / max(frame_dt, 1e-8)
        if self.fps_smoothed <= 0.0:
            self.fps_smoothed = fps
        else:
            self.fps_smoothed = 0.9 * self.fps_smoothed + 0.1 * fps


class RigidBodyViewer:
    """
    Taichi-based viewer with a console fallback.

    The GGUI path renders 3D box meshes using Taichi fields so per-frame
    geometry updates can run in parallel inside kernels. The fallback keeps the
    project runnable on machines where Taichi is unavailable.
    """

    def __init__(
        self,
        world: RigidBodyWorld,
        *,
        realtime: bool = False,
        resolution: tuple[int, int] = (1280, 720),
        force_strength: float = 4.0,
    ):
        self.world = world
        self.realtime = realtime
        self.resolution = resolution
        self.force_strength = force_strength
        self.stats = ViewerStats()

        self.window = None
        self.canvas = None
        self.scene = None
        self.camera = None
        self.gui = None
        self.buffers = None
        self._buffer_capacity = 0
        self._hotkey_latches: dict[str, bool] = {}

        if ti is not None:
            _ensure_taichi_runtime()
            self.window = ti.ui.Window(
                name="Rigid Body Lab",
                res=resolution,
                vsync=False,
                fps_limit=1000,
            )
            self.canvas = self.window.get_canvas()
            self.scene = ti.ui.Scene()
            self.camera = ti.ui.Camera()
            self.gui = self.window.get_gui()
            self._configure_camera()
            self._ensure_buffers()

    def run(self, max_frames: int | None = None, print_every: int = 30) -> None:
        if self.window is None:
            self._run_console(max_frames=max_frames, print_every=print_every)
            return

        frame_idx = 0
        while self.window.running and (max_frames is None or frame_idx < max_frames):
            frame_start = time.perf_counter()
            self._handle_window_events()
            self.world.step()
            self._draw_taichi_frame()
            self.stats.update(time.perf_counter() - frame_start)
            frame_idx += 1
            if self.realtime:
                time.sleep(self.world.config.time_step)

    def _run_console(self, max_frames: int | None, print_every: int) -> None:
        total_frames = 180 if max_frames is None else max_frames
        for frame_idx in range(total_frames):
            self.world.step()
            if frame_idx % max(1, print_every) == 0:
                print(self.world.describe())
                print("-" * 60)
            if self.realtime:
                time.sleep(self.world.config.time_step)

    def _configure_camera(self) -> None:
        assert self.camera is not None
        self.camera.position(3.8, 2.6, 5.2)
        self.camera.lookat(0.0, 0.8, 0.0)
        self.camera.up(0.0, 1.0, 0.0)

    def _ensure_buffers(self) -> None:
        body_count = max(1, len(self.world.state.bodies))
        if self.buffers is not None and self._buffer_capacity == body_count:
            return
        if ti is None:
            return
        self.buffers = TaichiSceneBuffers(body_count)
        self._buffer_capacity = body_count

    def _handle_window_events(self) -> None:
        assert self.window is not None
        assert self.camera is not None

        self.camera.track_user_inputs(
            self.window,
            movement_speed=0.03,
            hold_key=ti.ui.RMB,
        )

        if self._edge_pressed("escape", ti.ui.ESCAPE):
            self.window.running = False
            return
        if self._edge_pressed("reset", "r"):
            self.world.reset_active_demo()
        elif self._edge_pressed("pause", "p"):
            self.world.toggle_pause()
        elif self._edge_pressed("demo_next", ti.ui.TAB):
            self.world.next_demo()

        if self.world.state.active_demo == "two_body_collision":
            if self._edge_pressed("case_next", "n"):
                self.world.next_two_body_case()
            elif self._edge_pressed("case_prev", "b"):
                self.world.previous_two_body_case()

        primary_body_id = self.world.get_primary_body_id()
        if primary_body_id is None:
            return

        force = np.zeros(3, dtype=np.float64)
        if self.window.is_pressed(ti.ui.LEFT, "a"):
            force[0] -= self.force_strength
        if self.window.is_pressed(ti.ui.RIGHT, "d"):
            force[0] += self.force_strength
        if self.window.is_pressed(ti.ui.UP, "w"):
            force[1] += self.force_strength
        if self.window.is_pressed(ti.ui.DOWN, "s"):
            force[1] -= self.force_strength
        if self.window.is_pressed("q"):
            force[2] += self.force_strength
        if self.window.is_pressed("e"):
            force[2] -= self.force_strength
        if np.any(force):
            self.world.apply_force_to_body(primary_body_id, force)

    def _edge_pressed(self, name: str, *keys) -> bool:
        assert self.window is not None
        pressed = self.window.is_pressed(*keys)
        was_pressed = self._hotkey_latches.get(name, False)
        self._hotkey_latches[name] = pressed
        return pressed and not was_pressed

    def _draw_taichi_frame(self) -> None:
        assert self.canvas is not None
        assert self.scene is not None
        assert self.camera is not None
        assert self.gui is not None

        body_count = len(self.world.state.bodies)
        self._ensure_buffers()
        self._sync_world_to_fields()

        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.55, 0.55, 0.58))
        self.scene.point_light(pos=(2.5, 4.0, 2.5), color=(1.0, 1.0, 1.0))
        self.scene.point_light(pos=(-2.5, 3.0, 1.5), color=(0.7, 0.75, 0.9))

        if body_count > 0:
            vertex_count = body_count * self.buffers.vertices_per_body
            index_count = body_count * self.buffers.triangle_index_count_per_body
            self.scene.mesh(
                self.buffers.vertices,
                self.buffers.indices,
                normals=self.buffers.normals,
                per_vertex_color=self.buffers.vertex_colors,
                vertex_count=vertex_count,
                index_count=index_count,
            )
            self.scene.mesh(
                self.buffers.vertices,
                self.buffers.indices,
                color=(0.08, 0.09, 0.12),
                vertex_count=vertex_count,
                index_count=index_count,
                show_wireframe=True,
            )
            self.scene.particles(
                self.buffers.centers,
                radius=0.04,
                per_vertex_color=self.buffers.center_colors,
                index_count=body_count,
            )

        self.canvas.set_background_color((0.08, 0.09, 0.11))
        self.canvas.scene(self.scene)
        self._draw_overlay()
        self.window.show()

    def _sync_world_to_fields(self) -> None:
        assert self.buffers is not None

        body_count = len(self.world.state.bodies)
        positions = np.zeros((self._buffer_capacity, 3), dtype=np.float32)
        orientations = np.zeros((self._buffer_capacity, 4), dtype=np.float32)
        half_extents = np.zeros((self._buffer_capacity, 3), dtype=np.float32)
        colors = np.zeros((self._buffer_capacity, 3), dtype=np.float32)
        center_colors = np.zeros((self._buffer_capacity, 3), dtype=np.float32)

        for body_id, body in enumerate(self.world.state.bodies):
            positions[body_id] = body.position.astype(np.float32)
            orientations[body_id] = body.orientation.astype(np.float32)
            half_extents[body_id] = body.half_extents.astype(np.float32)
            colors[body_id] = body.color.astype(np.float32)
            center_colors[body_id] = (
                np.array([1.0, 0.62, 0.26], dtype=np.float32)
                if body.is_dynamic
                else np.array([0.62, 0.66, 0.74], dtype=np.float32)
            )

        self.buffers.body_positions.from_numpy(positions)
        self.buffers.body_orientations.from_numpy(orientations)
        self.buffers.body_half_extents.from_numpy(half_extents)
        self.buffers.body_colors.from_numpy(colors)
        self.buffers.center_colors.from_numpy(center_colors)
        self.buffers.active_body_count[None] = body_count
        self.buffers.update_transforms()

    def _draw_overlay(self) -> None:
        assert self.gui is not None

        primary_body_id = self.world.get_primary_body_id()
        with self.gui.sub_window("Simulation", 0.02, 0.02, 0.28, 0.25):
            self.gui.text(f"Demo: {self.world.state.active_demo}")
            self.gui.text(f"Frame: {self.world.state.frame}")
            self.gui.text(f"Time: {self.world.state.time:.3f}")
            self.gui.text(f"FPS: {self.stats.fps_smoothed:.1f}")
            self.gui.text(f"Taichi arch: {_TAICHI_ARCH_NAME}")
            if self.world.state.active_demo == "two_body_collision":
                self.gui.text(f"Case: {self.world.active_two_body_case().name}")

            if primary_body_id is not None:
                body = self.world.state.get_body(primary_body_id)
                self.gui.text(
                    "Linear vel: "
                    + np.array2string(body.linear_velocity, precision=2, suppress_small=True)
                )
                self.gui.text(
                    "Angular vel: "
                    + np.array2string(body.angular_velocity, precision=2, suppress_small=True)
                )

            if self.gui.button("Reset [R]"):
                self.world.reset_active_demo()
            if self.gui.button("Pause / Resume [P]"):
                self.world.toggle_pause()

        with self.gui.sub_window("Controls", 0.02, 0.30, 0.28, 0.22):
            self.gui.text("Move camera: RMB + WASD/EQ")
            self.gui.text("Apply force: WASD/QE")
            self.gui.text("Switch demo: Tab")
            if self.world.state.active_demo == "two_body_collision":
                self.gui.text("Collision case: B / N")
            self.gui.text("Close: Esc")
