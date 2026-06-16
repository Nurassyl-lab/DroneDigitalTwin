"""
Visualize Project AirSim NED coordinates for PX4 waypoint selection.

This helper loads a scene, samples a voxel grid, saves a top-down occupancy
slice, and optionally draws persistent debug markers in Unreal.

python px4_map_viewer.py `
   --start "30,0,-6" `
   --goal "30,-48,-10" `
   --slice-z-ned -8 `
   --resolution-m 1 `
   --grid-step-m 10 `
   --label-step-m 20 `
   --output astar_map_view.png `
   --map-size "120,120,40"
"""

import argparse
import asyncio
import math
from pathlib import Path
from typing import List, Sequence

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.planners import AStarPlanner
from projectairsim.types import Pose, Quaternion, Vector3
from projectairsim.utils import calculate_path_length, projectairsim_log


def parse_vector3(value: str) -> List[float]:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected three coordinates, got {len(parts)} from '{value}'"
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Coordinates must be numeric: '{value}'"
        ) from exc


def parse_size3(value: str) -> List[int]:
    size = parse_vector3(value)
    if any(component <= 0 for component in size):
        raise argparse.ArgumentTypeError("Map size values must be positive")
    return [int(math.ceil(component)) for component in size]


def make_pose_ned(position_ned: Sequence[float]) -> Pose:
    return Pose(
        {
            "translation": Vector3(
                {
                    "x": position_ned[0],
                    "y": position_ned[1],
                    "z": position_ned[2],
                }
            ),
            "rotation": Quaternion({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}),
            "frame_id": "DEFAULT_ID",
        }
    )


def infer_map_center(start: Sequence[float], goal: Sequence[float]) -> List[float]:
    return [(start[idx] + goal[idx]) / 2.0 for idx in range(3)]


def infer_map_size(
    start: Sequence[float],
    goal: Sequence[float],
    margin_m: float,
    minimum_size_m: float,
) -> List[int]:
    return [
        int(max(minimum_size_m, math.ceil(abs(goal[idx] - start[idx]) + 2 * margin_m)))
        for idx in range(3)
    ]


def make_range(min_value: float, max_value: float, step: float) -> List[float]:
    values = []
    current = math.ceil(min_value / step) * step
    while current <= max_value + 1e-6:
        values.append(round(current, 6))
        current += step
    return values


def sample_occupancy_slice(
    planner: AStarPlanner,
    map_center: Sequence[float],
    map_size: Sequence[int],
    resolution_m: float,
    slice_z_ned: float,
):
    x_cells = int(map_size[0] / resolution_m)
    y_cells = int(map_size[1] / resolution_m)
    x_min = map_center[0] - map_size[0] / 2.0
    y_min = map_center[1] - map_size[1] / 2.0

    occupancy = []
    for y_idx in range(y_cells):
        row = []
        y = y_min + (y_idx + 0.5) * resolution_m
        for x_idx in range(x_cells):
            x = x_min + (x_idx + 0.5) * resolution_m
            try:
                idx = planner.get_grid_idx([x, y, slice_z_ned], is_NED=True)
                occupied = bool(planner.occupancy_map[idx])
            except Exception:
                occupied = True
            row.append(occupied)
        occupancy.append(row)

    return occupancy


def save_top_down_map(
    occupancy,
    output_path: Path,
    map_center: Sequence[float],
    map_size: Sequence[int],
    slice_z_ned: float,
    start: Sequence[float] = None,
    goal: Sequence[float] = None,
    path: List[List[float]] = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    x_min = map_center[0] - map_size[0] / 2.0
    x_max = map_center[0] + map_size[0] / 2.0
    y_min = map_center[1] - map_size[1] / 2.0
    y_max = map_center[1] + map_size[1] / 2.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        occupancy,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap=ListedColormap(["white", "black"]),
        interpolation="nearest",
        alpha=0.85,
    )
    ax.set_title(f"Project AirSim occupancy slice, NED z={slice_z_ned:g} m")
    ax.set_xlabel("NED x / north (m)")
    ax.set_ylabel("NED y / east (m)")
    ax.grid(True, color="0.75", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")

    if path:
        ax.plot([p[0] for p in path], [p[1] for p in path], color="limegreen", linewidth=2)
    if start:
        ax.scatter([start[0]], [start[1]], marker="o", s=90, color="dodgerblue", label="start")
        ax.text(start[0], start[1], f" start {start}", color="dodgerblue")
    if goal:
        ax.scatter([goal[0]], [goal[1]], marker="x", s=120, color="crimson", label="goal")
        ax.text(goal[0], goal[1], f" goal {goal}", color="crimson")
    if start or goal:
        ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_debug_grid(
    world: World,
    map_center: Sequence[float],
    map_size: Sequence[int],
    slice_z_ned: float,
    grid_step_m: float,
    label_step_m: float,
    duration_sec: float,
    persistent: bool,
):
    x_min = map_center[0] - map_size[0] / 2.0
    x_max = map_center[0] + map_size[0] / 2.0
    y_min = map_center[1] - map_size[1] / 2.0
    y_max = map_center[1] + map_size[1] / 2.0

    grid_segments = []
    for x in make_range(x_min, x_max, grid_step_m):
        grid_segments.extend([[x, y_min, slice_z_ned], [x, y_max, slice_z_ned]])
    for y in make_range(y_min, y_max, grid_step_m):
        grid_segments.extend([[x_min, y, slice_z_ned], [x_max, y, slice_z_ned]])

    if grid_segments:
        world.plot_debug_dashed_line(
            grid_segments,
            [0.55, 0.55, 0.55, 1.0],
            1.0,
            duration_sec,
            persistent,
        )

    world.plot_debug_solid_line(
        [[x_min, 0.0, slice_z_ned], [x_max, 0.0, slice_z_ned]],
        [1.0, 0.05, 0.05, 1.0],
        4.0,
        duration_sec,
        persistent,
    )
    world.plot_debug_solid_line(
        [[0.0, y_min, slice_z_ned], [0.0, y_max, slice_z_ned]],
        [0.05, 0.9, 0.05, 1.0],
        4.0,
        duration_sec,
        persistent,
    )

    strings = []
    positions = []
    for x in make_range(x_min, x_max, label_step_m):
        strings.append(f"x={x:g}")
        positions.append([x, y_min, slice_z_ned])
    for y in make_range(y_min, y_max, label_step_m):
        strings.append(f"y={y:g}")
        positions.append([x_min, y, slice_z_ned])

    if strings:
        world.plot_debug_strings(
            strings,
            positions,
            0.8,
            [1.0, 1.0, 1.0, 1.0],
            duration_sec,
        )


def plot_debug_points_and_path(
    world: World,
    start: Sequence[float],
    goal: Sequence[float],
    path: List[List[float]],
    duration_sec: float,
    persistent: bool,
):
    points = []
    labels = []
    positions = []

    if start:
        points.append(start)
        labels.append(f"START {start}")
        positions.append(start)
    if goal:
        points.append(goal)
        labels.append(f"GOAL {goal}")
        positions.append(goal)

    if points:
        world.plot_debug_points(points, [0.0, 0.45, 1.0, 1.0], 18.0, duration_sec, persistent)
        world.plot_debug_strings(labels, positions, 1.0, [1.0, 1.0, 1.0, 1.0], duration_sec)

    if path and len(path) >= 2:
        world.plot_debug_solid_line(path, [0.0, 1.0, 0.15, 1.0], 5.0, duration_sec, persistent)


async def main(args):
    client = ProjectAirSimClient(
        address=args.server_ip,
        port_topics=args.topics_port,
        port_services=args.services_port,
    )

    try:
        client.connect()
        world = World(
            client,
            args.scene,
            delay_after_load_sec=args.load_delay_sec,
            sim_config_path=args.sim_config_path,
        )

        if args.clear_only:
            world.flush_persistent_markers()
            projectairsim_log().info("Unreal debug markers cleared")
            return

        map_center = args.map_center
        map_size = args.map_size
        if args.start and args.goal:
            map_center = map_center or infer_map_center(args.start, args.goal)
            map_size = map_size or infer_map_size(
                args.start,
                args.goal,
                args.map_margin_m,
                args.min_map_size_m,
            )
        else:
            map_center = map_center or [0.0, 0.0, args.slice_z_ned]
            map_size = map_size or [100, 100, 40]

        projectairsim_log().info(
            "Map bounds NED: x=[%.1f, %.1f], y=[%.1f, %.1f], z=[%.1f, %.1f]",
            map_center[0] - map_size[0] / 2.0,
            map_center[0] + map_size[0] / 2.0,
            map_center[1] - map_size[1] / 2.0,
            map_center[1] + map_size[1] / 2.0,
            map_center[2] - map_size[2] / 2.0,
            map_center[2] + map_size[2] / 2.0,
        )

        if args.print_drone_pose:
            drone = Drone(client, world, args.drone_name)
            position = drone.get_ground_truth_kinematics()["pose"]["position"]
            projectairsim_log().info(
                "Current %s pose NED: [%.2f, %.2f, %.2f]",
                args.drone_name,
                position["x"],
                position["y"],
                position["z"],
            )

        actors_to_ignore = args.ignore_actor or [args.drone_name]
        occupancy_grid = world.create_voxel_grid(
            make_pose_ned(map_center),
            map_size[0],
            map_size[1],
            map_size[2],
            args.resolution_m,
            actors_to_ignore=actors_to_ignore,
        )
        planner = AStarPlanner(
            occupancy_grid,
            map_center,
            map_size,
            args.resolution_m,
            ground_z_ned=args.ground_z_ned,
        )

        path = None
        if args.start:
            projectairsim_log().info(
                "Start valid at planning altitude: %s",
                planner.check_coordinate_validity(args.start, is_NED=True),
            )
        if args.goal:
            projectairsim_log().info(
                "Goal valid at planning altitude: %s",
                planner.check_coordinate_validity(args.goal, is_NED=True),
            )
        if args.start and args.goal and not args.no_plan:
            path = planner.generate_plan(args.start, args.goal)
            if path:
                projectairsim_log().info(
                    "A* path has %d points, length %.2f m",
                    len(path),
                    calculate_path_length(path),
                )
            else:
                projectairsim_log().info("A* did not find a path")

        if not args.no_unreal_markers:
            if not args.keep_existing_markers:
                world.flush_persistent_markers()
            plot_debug_grid(
                world,
                map_center,
                map_size,
                args.slice_z_ned,
                args.grid_step_m,
                args.label_step_m,
                args.duration_sec,
                not args.non_persistent,
            )
            plot_debug_points_and_path(
                world,
                args.start,
                args.goal,
                path,
                args.duration_sec,
                not args.non_persistent,
            )
            projectairsim_log().info("Unreal debug markers plotted")

        if args.output:
            occupancy = sample_occupancy_slice(
                planner,
                map_center,
                map_size,
                args.resolution_m,
                args.slice_z_ned,
            )
            output_path = Path(args.output)
            save_top_down_map(
                occupancy,
                output_path,
                map_center,
                map_size,
                args.slice_z_ned,
                args.start,
                args.goal,
                path,
            )
            projectairsim_log().info("Top-down map written to %s", output_path)

    finally:
        if client.state:
            client.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw and save a Project AirSim NED coordinate map."
    )
    parser.add_argument("--scene", default="scene_px4_sitl.jsonc")
    parser.add_argument("--sim-config-path", default="sim_config/")
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--topics-port", type=int, default=8989)
    parser.add_argument("--services-port", type=int, default=8990)
    parser.add_argument("--load-delay-sec", type=float, default=2.0)
    parser.add_argument("--start", type=parse_vector3)
    parser.add_argument("--goal", type=parse_vector3)
    parser.add_argument("--map-center", type=parse_vector3)
    parser.add_argument("--map-size", type=parse_size3)
    parser.add_argument("--map-margin-m", type=float, default=20.0)
    parser.add_argument("--min-map-size-m", type=float, default=50.0)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--slice-z-ned", type=float, default=-8.0)
    parser.add_argument("--ground-z-ned", type=float, default=0.0)
    parser.add_argument("--grid-step-m", type=float, default=10.0)
    parser.add_argument("--label-step-m", type=float, default=20.0)
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--output", default="astar_map_view.png")
    parser.add_argument("--no-unreal-markers", action="store_true")
    parser.add_argument("--keep-existing-markers", action="store_true")
    parser.add_argument("--clear-only", action="store_true")
    parser.add_argument("--non-persistent", action="store_true")
    parser.add_argument("--no-plan", action="store_true")
    parser.add_argument("--print-drone-pose", action="store_true")
    parser.add_argument(
        "--ignore-actor",
        action="append",
        default=None,
        help="Actor to ignore in the occupancy grid. Repeatable.",
    )
    return parser


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
