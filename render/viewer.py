from __future__ import annotations

import time

from sim.world import RigidBodyWorld


class RigidBodyViewer:
    """
    Minimal placeholder viewer.

    The lab ultimately needs a proper Taichi/GGUI renderer, but for the current
    framework pass this class focuses on wiring the simulation loop, demo
    switching, and status reporting together.
    """

    def __init__(self, world: RigidBodyWorld, *, realtime: bool = False):
        self.world = world
        self.realtime = realtime

    def run(self, max_frames: int = 180, print_every: int = 30) -> None:
        for frame_idx in range(max_frames):
            self.world.step()
            if frame_idx % max(1, print_every) == 0:
                self._draw_frame()
            if self.realtime:
                time.sleep(self.world.config.time_step)

    def _draw_frame(self) -> None:
        print(self.world.describe())
        print("-" * 60)
