from __future__ import annotations

import argparse

from render.viewer import RigidBodyViewer
from sim.state import SimulationConfig
from sim.world import RigidBodyWorld


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rigid body lab 1 framework entrypoint.")
    parser.add_argument(
        "--demo",
        default="single_body",
        choices=("single_body", "two_body_collision", "complex_scene"),
        help="Select which handout demo to load.",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Simulation time step.")
    parser.add_argument("--substeps", type=int, default=1, help="Number of substeps per frame.")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="How many frames to advance. Use 0 to run until the GUI window is closed.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the simulation loop and print a final state summary.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep for one simulation step between rendered frames.",
    )
    parser.add_argument(
        "--collision-case",
        choices=("point_face", "edge_edge", "face_face", "random_pose"),
        default=None,
        help="Select a preset setup for the two-body collision demo.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = SimulationConfig(time_step=args.dt, substeps=args.substeps)
    world = RigidBodyWorld(config=config, demo_name=args.demo)
    if args.collision_case is not None:
        world.select_two_body_case(args.collision_case)

    if args.headless:
        headless_steps = args.steps if args.steps > 0 else 180
        for _ in range(headless_steps):
            world.step()
        print(world.describe())
        return

    viewer = RigidBodyViewer(world, realtime=args.realtime)
    viewer.run(max_frames=None if args.steps <= 0 else args.steps)


if __name__ == "__main__":
    main()
