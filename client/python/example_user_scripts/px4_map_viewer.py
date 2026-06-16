"""
Visualize Project AirSim NED coordinates for PX4 waypoint selection.

This helper loads a scene, samples a voxel grid, saves a top-down occupancy
map, and optionally draws persistent debug markers in Unreal.

python px4_map_viewer.py `
   --start "30,0,-6" `
   --goal "30,-48,-10" `
   --slice-z-ned -8 `
   --resolution-m 1 `
   --grid-step-m 10 `
   --label-step-m 20 `
   --output astar_map_view.png `
   --output-3d astar_map_view_3d.png `
   --map-size "120,120,40"
"""

import argparse
import asyncio
import math
import sys
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


def normalize_vector_args(argv: Sequence[str]) -> List[str]:
    normalized = []
    vector_options = {"--start", "--goal", "--map-center", "--map-size"}
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in vector_options and idx + 1 < len(argv):
            normalized.append(f"{arg}={argv[idx + 1]}")
            idx += 2
            continue
        normalized.append(arg)
        idx += 1
    return normalized


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


def infer_vertical_map_center(
    start: Sequence[float],
    goal: Sequence[float],
    slice_z_ned: float,
    ground_z_ned: float,
    map_height_m: float,
) -> float:
    z_values = [start[2], goal[2], slice_z_ned, ground_z_ned]
    z_min = min(z_values)
    z_max = max(z_values)
    if z_max - z_min <= map_height_m:
        return (z_min + z_max) / 2.0
    return (start[2] + goal[2]) / 2.0


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


def sample_occupancy_projection(
    planner: AStarPlanner,
    map_center: Sequence[float],
    map_size: Sequence[int],
    resolution_m: float,
):
    x_cells = int(map_size[0] / resolution_m)
    y_cells = int(map_size[1] / resolution_m)
    z_cells = int(map_size[2] / resolution_m)
    x_min = map_center[0] - map_size[0] / 2.0
    y_min = map_center[1] - map_size[1] / 2.0
    z_min = map_center[2] - map_size[2] / 2.0

    occupancy = []
    for y_idx in range(y_cells):
        row = []
        y = y_min + y_idx * resolution_m
        for x_idx in range(x_cells):
            x = x_min + x_idx * resolution_m
            occupied = False
            for z_idx in range(z_cells):
                z = z_min + z_idx * resolution_m
                try:
                    idx = planner.get_grid_idx([x, y, z], is_NED=True)
                    if bool(planner.occupancy_map[idx]):
                        occupied = True
                        break
                except Exception:
                    occupied = True
                    break
            row.append(occupied)
        occupancy.append(row)

    return occupancy


def occupied_voxel_points(
    occupancy_grid,
    map_center: Sequence[float],
    map_size: Sequence[int],
    resolution_m: float,
    max_points: int,
):
    x_cells = int(map_size[0] / resolution_m)
    y_cells = int(map_size[1] / resolution_m)
    z_cells = int(map_size[2] / resolution_m)
    occupied_count = sum(1 for occupied in occupancy_grid if occupied)
    if occupied_count == 0:
        return [], 0, 1

    stride = max(1, math.ceil(occupied_count / max_points)) if max_points > 0 else 1
    points = []
    seen = 0
    x_origin = map_center[0] - (x_cells / 2.0) * resolution_m
    y_origin = map_center[1] - (y_cells / 2.0) * resolution_m
    neu_z_origin = -map_center[2] - (z_cells / 2.0) * resolution_m

    for y_idx in range(y_cells):
        for z_idx in range(z_cells):
            for x_idx in range(x_cells):
                idx = x_idx + x_cells * (z_idx + z_cells * y_idx)
                if not occupancy_grid[idx]:
                    continue
                seen += 1
                if (seen - 1) % stride != 0:
                    continue

                x = x_origin + x_idx * resolution_m
                y = y_origin + y_idx * resolution_m
                z_ned = -(neu_z_origin + z_idx * resolution_m)
                points.append((x, y, z_ned))

    return points, occupied_count, stride


def save_top_down_map(
    occupancy,
    output_path: Path,
    map_center: Sequence[float],
    map_size: Sequence[int],
    title: str,
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
    ax.set_title(title)
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


def save_3d_map(
    occupancy_grid,
    output_path: Path,
    map_center: Sequence[float],
    map_size: Sequence[int],
    resolution_m: float,
    start: Sequence[float] = None,
    goal: Sequence[float] = None,
    path: List[List[float]] = None,
    max_points: int = 100000,
    elev: float = 25.0,
    azim: float = -55.0,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points, occupied_count, stride = occupied_voxel_points(
        occupancy_grid,
        map_center,
        map_size,
        resolution_m,
        max_points,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    x_min = map_center[0] - map_size[0] / 2.0
    x_max = map_center[0] + map_size[0] / 2.0
    y_min = map_center[1] - map_size[1] / 2.0
    y_max = map_center[1] + map_size[1] / 2.0
    z_min = map_center[2] - map_size[2] / 2.0
    z_max = map_center[2] + map_size[2] / 2.0

    if points:
        xs, ys, zs = zip(*points)
        ax.scatter(
            xs,
            ys,
            zs,
            c=zs,
            cmap="viridis",
            marker="s",
            s=3,
            alpha=0.55,
            linewidths=0,
        )

    if path:
        ax.plot(
            [p[0] for p in path],
            [p[1] for p in path],
            [p[2] for p in path],
            color="limegreen",
            linewidth=2,
            label="path",
        )
    if start:
        ax.scatter(
            [start[0]],
            [start[1]],
            [start[2]],
            marker="o",
            s=80,
            color="dodgerblue",
            label="start",
        )
    if goal:
        ax.scatter(
            [goal[0]],
            [goal[1]],
            [goal[2]],
            marker="x",
            s=100,
            color="crimson",
            label="goal",
        )
    if start or goal or path:
        ax.legend(loc="upper right")

    ax.set_title(
        "Project AirSim 3D occupancy "
        f"({len(points):,}/{occupied_count:,} occupied voxels shown, stride={stride})"
    )
    ax.set_xlabel("NED x / north (m)")
    ax.set_ylabel("NED y / east (m)")
    ax.set_zlabel("NED z / down (m)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((map_size[0], map_size[1], map_size[2]))
    except AttributeError:
        pass
    ax.grid(True, color="0.8")

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
            if args.map_center is None:
                map_center[2] = infer_vertical_map_center(
                    args.start,
                    args.goal,
                    args.slice_z_ned,
                    args.ground_z_ned,
                    map_size[2],
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
            if args.render_mode == "slice":
                occupancy = sample_occupancy_slice(
                    planner,
                    map_center,
                    map_size,
                    args.resolution_m,
                    args.slice_z_ned,
                )
                title = f"Project AirSim occupancy slice, NED z={args.slice_z_ned:g} m"
            else:
                occupancy = sample_occupancy_projection(
                    planner,
                    map_center,
                    map_size,
                    args.resolution_m,
                )
                z_min = map_center[2] - map_size[2] / 2.0
                z_max = map_center[2] + map_size[2] / 2.0
                title = (
                    "Project AirSim occupancy projection, "
                    f"NED z=[{z_min:g}, {z_max:g}] m"
                )
            output_path = Path(args.output)
            save_top_down_map(
                occupancy,
                output_path,
                map_center,
                map_size,
                title,
                args.start,
                args.goal,
                path,
            )
            projectairsim_log().info("Top-down map written to %s", output_path)

        if args.output_3d:
            output_3d_path = Path(args.output_3d)
            save_3d_map(
                occupancy_grid,
                output_3d_path,
                map_center,
                map_size,
                args.resolution_m,
                args.start,
                args.goal,
                path,
                args.max_3d_points,
                args.view_elev,
                args.view_azim,
            )
            projectairsim_log().info("3D occupancy map written to %s", output_3d_path)

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
    parser.add_argument(
        "--render-mode",
        choices=("projection", "slice"),
        default="projection",
        help=(
            "projection marks a top-down cell occupied if any voxel in its "
            "vertical column is occupied; slice only samples --slice-z-ned."
        ),
    )
    parser.add_argument("--slice-z-ned", type=float, default=-8.0)
    parser.add_argument("--ground-z-ned", type=float, default=0.0)
    parser.add_argument("--grid-step-m", type=float, default=10.0)
    parser.add_argument("--label-step-m", type=float, default=20.0)
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--output", default="astar_map_view.png")
    parser.add_argument(
        "--output-3d",
        default=None,
        help="Optional path for a 3D occupancy PNG covering the full voxel grid.",
    )
    parser.add_argument(
        "--max-3d-points",
        type=int,
        default=100000,
        help="Maximum occupied voxels to draw in the 3D PNG before downsampling.",
    )
    parser.add_argument("--view-elev", type=float, default=25.0)
    parser.add_argument("--view-azim", type=float, default=-55.0)
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
    asyncio.run(main(build_parser().parse_args(normalize_vector_args(sys.argv[1:]))))
