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
    parser.add_argument("--steps", type=int, default=180, help="How many frames to advance.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the simulation loop and print a final state summary.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = SimulationConfig(time_step=args.dt, substeps=args.substeps)
    world = RigidBodyWorld(config=config, demo_name=args.demo)

    if args.headless:
        for _ in range(args.steps):
            world.step()
        print(world.describe())
        return

    viewer = RigidBodyViewer(world)
    viewer.run(max_frames=args.steps)


if __name__ == "__main__":
    main()
