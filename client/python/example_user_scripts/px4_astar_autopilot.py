"""
PX4-backed A* waypoint autopilot example.

Run this with the Project AirSim server and PX4 SITL already running. The script
loads a PX4 scene, builds an occupancy grid, plans a NED path with A*, and sends
that path through Project AirSim's PX4 offboard movement API.
"""

import argparse
import asyncio
import math
import time
import sys
from pathlib import Path
from typing import List, Sequence

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.drone import YawControlMode
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


def sparsify_path(path: List[List[float]], min_spacing_m: float) -> List[List[float]]:
    if len(path) <= 2 or min_spacing_m <= 0:
        return path

    sparse_path = [path[0]]
    last_kept = path[0]
    for point in path[1:-1]:
        if calculate_path_length([last_kept, point]) >= min_spacing_m:
            sparse_path.append(point)
            last_kept = point

    if sparse_path[-1] != path[-1]:
        sparse_path.append(path[-1])

    return sparse_path


def validate_grid_coordinate(planner: AStarPlanner, point: Sequence[float], name: str):
    if not planner.check_coordinate_validity(point, is_NED=True):
        raise RuntimeError(f"{name} point is outside the free planning space: {point}")


def distance_between(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((a[idx] - b[idx]) ** 2 for idx in range(3)))


def format_vector3(values: Sequence[float]) -> str:
    return f"[{values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}]"


def get_pose_position_ned(drone: Drone) -> List[float]:
    position = drone.get_ground_truth_kinematics()["pose"]["position"]
    return [float(position["x"]), float(position["y"]), float(position["z"])]


def log_pose(drone: Drone, label: str) -> List[float]:
    position = get_pose_position_ned(drone)
    projectairsim_log().info(f"{label} pose NED {format_vector3(position)}")
    return position


class FlightTrace:
    def __init__(self, sample_interval_sec: float):
        self.sample_interval_sec = max(0.0, sample_interval_sec)
        self.started_at = time.time()
        self.last_sample_at = 0.0
        self.samples = []

    def record(self, label: str, position: Sequence[float], force: bool = False):
        now = time.time()
        if (
            not force
            and self.samples
            and now - self.last_sample_at < self.sample_interval_sec
        ):
            return

        self.samples.append(
            {
                "time_sec": now - self.started_at,
                "label": label,
                "position": [float(position[0]), float(position[1]), float(position[2])],
            }
        )
        self.last_sample_at = now


class LiveNedDisplay:
    def __init__(self, enabled: bool, interval_sec: float):
        self.enabled = enabled
        self.interval_sec = max(0.0, interval_sec)
        self.started_at = time.time()
        self.last_print_at = 0.0

    def show(
        self,
        label: str,
        position: Sequence[float],
        target: Sequence[float] = None,
        remaining_m: float = None,
        force: bool = False,
    ):
        if not self.enabled:
            return

        now = time.time()
        if not force and now - self.last_print_at < self.interval_sec:
            return

        elapsed = now - self.started_at
        message = (
            f"[LIVE NED {elapsed:7.1f}s] {label}: "
            f"x={position[0]:8.2f}  y={position[1]:8.2f}  z={position[2]:8.2f}"
        )
        if target is not None:
            message += (
                f"  target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
            )
        if remaining_m is not None:
            message += f"  remaining={remaining_m:.2f}m"

        print(message, flush=True)
        self.last_print_at = now


def save_flight_trace_plot(
    trace: FlightTrace,
    output_path: Path,
    planned_path: List[List[float]] = None,
    start: Sequence[float] = None,
    goal: Sequence[float] = None,
):
    if not trace.samples:
        projectairsim_log().info("No flight trace samples to plot")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [sample["time_sec"] for sample in trace.samples]
    xs = [sample["position"][0] for sample in trace.samples]
    ys = [sample["position"][1] for sample in trace.samples]
    zs = [sample["position"][2] for sample in trace.samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (xy_ax, z_ax) = plt.subplots(1, 2, figsize=(14, 6))

    if planned_path:
        xy_ax.plot(
            [point[0] for point in planned_path],
            [point[1] for point in planned_path],
            color="0.55",
            linestyle="--",
            linewidth=1.5,
            label="planned path",
        )
    xy_ax.plot(xs, ys, color="dodgerblue", linewidth=2.0, label="actual drone")
    xy_ax.scatter([xs[0]], [ys[0]], color="dodgerblue", s=60, marker="o", label="trace start")
    xy_ax.scatter([xs[-1]], [ys[-1]], color="crimson", s=70, marker="x", label="trace end")
    if start:
        xy_ax.scatter([start[0]], [start[1]], color="navy", s=45, marker="o", label="requested start")
    if goal:
        xy_ax.scatter([goal[0]], [goal[1]], color="darkred", s=60, marker="x", label="requested goal")
    xy_ax.set_title("Actual Drone NED Track")
    xy_ax.set_xlabel("NED x / north (m)")
    xy_ax.set_ylabel("NED y / east (m)")
    xy_ax.set_aspect("equal", adjustable="box")
    xy_ax.grid(True, color="0.82")
    xy_ax.legend(loc="best")

    z_ax.plot(times, zs, color="darkorange", linewidth=2.0)
    z_ax.set_title("NED z During Flight")
    z_ax.set_xlabel("time (s)")
    z_ax.set_ylabel("NED z / down (m)")
    z_ax.grid(True, color="0.82")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    projectairsim_log().info("Flight trace plot written to %s", output_path)


async def wait_for_px4_ready(drone: Drone, timeout_sec: float):
    timeout_at = time.time() + timeout_sec
    last_message = ""

    while time.time() < timeout_at:
        state = drone.get_ready_state()
        if state.get("ready_val") and drone.can_arm():
            projectairsim_log().info("PX4 controller is connected and can arm")
            return

        message = state.get("ready_message") or "Waiting for PX4 controller"
        if state.get("ready_val"):
            message = (
                "Waiting for PX4 MAVLink/GPS readiness. If PX4 says it is "
                "waiting for simulator TCP port 4560, restart PX4 after this "
                "scene is loaded or check PX4_SIM_HOST_ADDR/local-host-ip."
            )
        if message != last_message:
            projectairsim_log().info(message)
            last_message = message
        await asyncio.sleep(1.0)

    raise RuntimeError(
        "PX4 did not become armable. Check that PX4 is running, that it has "
        "connected to Project AirSim on TCP port 4560, that GPS fusion/home "
        "position completed, and that PX4_SIM_HOST_ADDR matches the robot "
        "config local-host-ip value."
    )


async def arm_with_retry(drone: Drone, timeout_sec: float):
    timeout_at = time.time() + timeout_sec
    attempt = 0

    while time.time() < timeout_at:
        attempt += 1
        if not drone.can_arm():
            projectairsim_log().info("PX4 is not armable yet")
            await asyncio.sleep(1.0)
            continue

        projectairsim_log().info(f"Invoking drone.arm() attempt {attempt}")
        if drone.arm():
            projectairsim_log().info("PX4 armed")
            return

        projectairsim_log().info("PX4 rejected arm request; retrying")
        await asyncio.sleep(1.0)

    raise RuntimeError(
        "PX4 did not arm before timeout. Check the PX4 terminal for health "
        "failures such as EKF, power, or GPS readiness."
    )


async def await_drone_task(
    drone: Drone,
    task: asyncio.Task,
    label: str,
    timeout_sec: float,
    report_interval_sec: float,
    flight_trace: FlightTrace = None,
    live_ned: LiveNedDisplay = None,
):
    started_at = time.time()
    last_report_at = started_at

    while not task.done():
        now = time.time()
        elapsed = now - started_at
        if timeout_sec > 0 and elapsed > timeout_sec:
            drone.cancel_last_task()
            position = get_pose_position_ned(drone)
            if flight_trace:
                flight_trace.record(label, position, force=True)
            if live_ned:
                live_ned.show(label, position, force=True)
            raise RuntimeError(
                f"{label} timed out after {timeout_sec:.1f}s at pose "
                f"{format_vector3(position)}"
            )

        position = get_pose_position_ned(drone)
        if flight_trace:
            flight_trace.record(label, position)
        if live_ned:
            live_ned.show(label, position)

        if now - last_report_at >= report_interval_sec:
            projectairsim_log().info(
                f"{label} still running after {elapsed:.1f}s; pose NED "
                f"{format_vector3(position)}"
            )
            last_report_at = now

        await asyncio.sleep(0.25)

    result = await task
    position = get_pose_position_ned(drone)
    if flight_trace:
        flight_trace.record(label, position, force=True)
    if live_ned:
        live_ned.show(label, position, force=True)
    projectairsim_log().info(
        f"{label} completed with result={result}; pose NED {format_vector3(position)}"
    )
    if result is False:
        raise RuntimeError(f"{label} returned False")
    return result


async def request_px4_control(drone: Drone):
    projectairsim_log().info("Requesting PX4 control for direct movement")
    request_control_task = await drone.request_control_async()
    await request_control_task


async def fly_to_point_by_velocity(
    drone: Drone,
    target: Sequence[float],
    velocity_mps: float,
    acceptance_m: float,
    timeout_sec: float,
    report_interval_sec: float,
    face_travel_direction: bool,
    label: str,
    flight_trace: FlightTrace = None,
    live_ned: LiveNedDisplay = None,
):
    await request_px4_control(drone)
    started_at = time.time()
    last_report_at = 0.0

    while True:
        current = get_pose_position_ned(drone)
        if flight_trace:
            flight_trace.record(label, current)
        delta = [target[idx] - current[idx] for idx in range(3)]
        distance = distance_between(current, target)
        if live_ned:
            live_ned.show(label, current, target, distance)
        elapsed = time.time() - started_at

        if distance <= acceptance_m:
            if flight_trace:
                flight_trace.record(label, current, force=True)
            if live_ned:
                live_ned.show(label, current, target, distance, force=True)
            projectairsim_log().info(
                f"{label} reached target {format_vector3(target)}; pose NED "
                f"{format_vector3(current)}; error {distance:.2f} m"
            )
            return

        if timeout_sec > 0 and elapsed > timeout_sec:
            drone.cancel_last_task()
            raise RuntimeError(
                f"{label} timed out after {timeout_sec:.1f}s; target "
                f"{format_vector3(target)}, pose NED {format_vector3(current)}, "
                f"remaining {distance:.2f} m"
            )

        if elapsed - last_report_at >= report_interval_sec:
            projectairsim_log().info(
                f"{label} moving toward {format_vector3(target)}; pose NED "
                f"{format_vector3(current)}; remaining {distance:.2f} m"
            )
            last_report_at = elapsed

        duration = min(0.5, max(0.1, distance / max(velocity_mps, 0.1)))
        speed = min(velocity_mps, distance / duration)
        velocity = [(component / distance) * speed for component in delta]
        yaw_control_mode = (
            YawControlMode.ForwardOnly
            if face_travel_direction
            else YawControlMode.MaxDegreeOfFreedom
        )

        move_task = await drone.move_by_velocity_async(
            v_north=velocity[0],
            v_east=velocity[1],
            v_down=velocity[2],
            duration=duration,
            yaw_control_mode=yaw_control_mode,
            yaw_is_rate=not face_travel_direction,
            yaw=0.0,
        )
        result = await move_task
        if result is False:
            raise RuntimeError(f"{label} velocity command returned False")


async def fly_path_by_velocity(
    drone: Drone,
    path: List[List[float]],
    velocity_mps: float,
    acceptance_m: float,
    timeout_sec: float,
    report_interval_sec: float,
    face_travel_direction: bool,
    flight_trace: FlightTrace = None,
    live_ned: LiveNedDisplay = None,
):
    for index, waypoint in enumerate(path[1:], start=1):
        await fly_to_point_by_velocity(
            drone,
            waypoint,
            velocity_mps,
            acceptance_m,
            timeout_sec,
            report_interval_sec,
            face_travel_direction,
            f"Waypoint {index:03d}",
            flight_trace,
            live_ned,
        )


async def run_autopilot(args):
    client = ProjectAirSimClient(
        address=args.server_ip,
        port_topics=args.topics_port,
        port_services=args.services_port,
    )
    image_display = None
    flight_trace = (
        FlightTrace(args.flight_trace_interval_sec)
        if args.flight_trace_output
        else None
    )
    live_ned = LiveNedDisplay(
        enabled=not args.no_live_ned,
        interval_sec=args.live_ned_interval_sec,
    )
    planned_path_for_trace = None

    try:
        projectairsim_log().info("Connecting to Project AirSim")
        client.connect()

        world = World(
            client,
            args.scene,
            delay_after_load_sec=args.load_delay_sec,
            sim_config_path=args.sim_config_path,
        )
        drone = Drone(client, world, args.drone_name)

        if args.show_chase_camera:
            from projectairsim.image_utils import ImageDisplay

            chase_window = "ChaseCam"
            image_display = ImageDisplay(num_subwin=1)
            image_display.add_chase_cam(
                chase_window,
                resize_x=args.chase_camera_width,
                resize_y=args.chase_camera_height,
            )
            chase_topic = drone.sensors.get("Chase", {}).get("scene_camera")
            if not chase_topic:
                available_topics = {
                    sensor: sorted(topics.keys())
                    for sensor, topics in drone.sensors.items()
                    if topics
                }
                raise RuntimeError(
                    "Chase scene_camera topic is not available. Make sure the "
                    "Chase camera capture setting has capture-enabled=true in "
                    "the active robot config. Available sensor topics: "
                    f"{available_topics}"
                )
            client.subscribe(
                chase_topic,
                lambda _, chase: image_display.receive(chase, chase_window),
            )
            image_display.start()
            projectairsim_log().info("Chase camera window opened")

        start = args.start
        goal = args.goal
        map_center = args.map_center or infer_map_center(start, goal)
        map_size = args.map_size or infer_map_size(
            start, goal, args.map_margin_m, args.min_map_size_m
        )

        if args.teleport_start:
            projectairsim_log().info(f"Teleporting '{args.drone_name}' to {start}")
            drone.set_pose(make_pose_ned(start), reset_kinematics=True)
            await asyncio.sleep(args.after_teleport_delay_sec)
            if flight_trace:
                flight_trace.record("Teleport start", get_pose_position_ned(drone), force=True)
            live_ned.show("Teleport start", get_pose_position_ned(drone), force=True)

        actors_to_ignore = args.ignore_actor or [args.drone_name]

        projectairsim_log().info(
            "Creating voxel grid: center=%s, size=%s, resolution=%s",
            map_center,
            map_size,
            args.resolution_m,
        )
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
        validate_grid_coordinate(planner, start, "Start")
        validate_grid_coordinate(planner, goal, "Goal")

        projectairsim_log().info(f"Planning path from {start} to {goal}")
        dense_path = planner.generate_plan(start, goal)
        if not dense_path:
            raise RuntimeError("A* did not find a path")

        path = sparsify_path(dense_path, args.waypoint_spacing_m)
        planned_path_for_trace = path
        projectairsim_log().info(
            "Planned %d dense points, reduced to %d waypoints, path length %.2f m",
            len(dense_path),
            len(path),
            calculate_path_length(path),
        )

        if args.print_waypoints:
            for idx, point in enumerate(path):
                projectairsim_log().info(f"Waypoint {idx:03d}: {point}")

        await wait_for_px4_ready(drone, args.px4_ready_timeout_sec)
        before_arming_pose = log_pose(drone, "Before arming")
        if flight_trace:
            flight_trace.record("Before arming", before_arming_pose, force=True)
        live_ned.show("Before arming", before_arming_pose, force=True)

        if not drone.enable_api_control():
            raise RuntimeError("Project AirSim did not enable API control")
        await arm_with_retry(drone, args.arm_timeout_sec)

        if not args.skip_takeoff:
            projectairsim_log().info("Taking off")
            takeoff_task = await drone.takeoff_async(timeout_sec=args.takeoff_timeout_sec)
            await await_drone_task(
                drone,
                takeoff_task,
                "Takeoff",
                args.takeoff_timeout_sec + 5.0,
                args.pose_report_interval_sec,
                flight_trace,
                live_ned,
            )
        else:
            skipping_takeoff_pose = log_pose(drone, "Skipping takeoff")
            if flight_trace:
                flight_trace.record("Skipping takeoff", skipping_takeoff_pose, force=True)
            live_ned.show("Skipping takeoff", skipping_takeoff_pose, force=True)

        if not args.teleport_start:
            if args.flight_driver == "velocity":
                projectairsim_log().info(f"Velocity-driving to start point {start}")
                await fly_to_point_by_velocity(
                    drone,
                    start,
                    args.velocity_mps,
                    args.waypoint_acceptance_m,
                    args.move_timeout_sec,
                    args.pose_report_interval_sec,
                    args.face_travel_direction,
                    "Move to start",
                    flight_trace,
                    live_ned,
                )
            else:
                await request_px4_control(drone)
                projectairsim_log().info(f"Moving to requested start point {start}")
                start_task = await drone.move_to_position_async(
                    north=start[0],
                    east=start[1],
                    down=start[2],
                    velocity=args.velocity_mps,
                    timeout_sec=args.move_timeout_sec,
                    yaw_control_mode=(
                        YawControlMode.ForwardOnly
                        if args.face_travel_direction
                        else YawControlMode.MaxDegreeOfFreedom
                    ),
                    yaw_is_rate=not args.face_travel_direction,
                    yaw=0.0,
                )
                await await_drone_task(
                    drone,
                    start_task,
                    "Move to start",
                    args.move_timeout_sec + 5.0,
                    args.pose_report_interval_sec,
                    flight_trace,
                    live_ned,
                )

        if args.flight_driver == "velocity":
            projectairsim_log().info("Velocity-driving planned waypoint path")
            await fly_path_by_velocity(
                drone,
                path,
                args.velocity_mps,
                args.waypoint_acceptance_m,
                args.move_timeout_sec,
                args.pose_report_interval_sec,
                args.face_travel_direction,
                flight_trace,
                live_ned,
            )
        else:
            await request_px4_control(drone)
            projectairsim_log().info("Following planned waypoint path")
            path_timeout_sec = args.path_timeout_sec
            if path_timeout_sec <= 0:
                path_timeout_sec = max(
                    args.move_timeout_sec,
                    calculate_path_length(path) / max(args.velocity_mps, 0.1) * 3.0,
                )
            path_task = await drone.move_on_path_async(
                path,
                velocity=args.velocity_mps,
                timeout_sec=path_timeout_sec,
                yaw_control_mode=(
                    YawControlMode.ForwardOnly
                    if args.face_travel_direction
                    else YawControlMode.MaxDegreeOfFreedom
                ),
                yaw_is_rate=not args.face_travel_direction,
                yaw=0.0,
                lookahead=args.lookahead_m,
                adaptive_lookahead=args.adaptive_lookahead,
            )
            await await_drone_task(
                drone,
                path_task,
                "Follow waypoint path",
                path_timeout_sec + 5.0,
                args.pose_report_interval_sec,
                flight_trace,
                live_ned,
            )
        projectairsim_log().info(f"Reached destination {goal}")

        if args.land_at_goal:
            projectairsim_log().info("Landing at destination")
            land_task = await drone.land_async(timeout_sec=args.land_timeout_sec)
            await await_drone_task(
                drone,
                land_task,
                "Landing",
                args.land_timeout_sec + 5.0,
                args.pose_report_interval_sec,
                flight_trace,
                live_ned,
            )

        if not args.keep_armed:
            drone.disarm()
            drone.disable_api_control()

    finally:
        if flight_trace and args.flight_trace_output:
            save_flight_trace_plot(
                flight_trace,
                Path(args.flight_trace_output),
                planned_path_for_trace,
                args.start,
                args.goal,
            )
        if image_display:
            image_display.stop()
        if client.state:
            client.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan an A* path and fly it with the PX4-backed drone API."
    )
    parser.add_argument("--start", type=parse_vector3, required=True, help="NED x,y,z")
    parser.add_argument("--goal", type=parse_vector3, required=True, help="NED x,y,z")
    parser.add_argument(
        "--scene",
        default="scene_px4_sitl.jsonc",
        help="Scene config to load from sim_config/",
    )
    parser.add_argument("--sim-config-path", default="sim_config/")
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--topics-port", type=int, default=8989)
    parser.add_argument("--services-port", type=int, default=8990)
    parser.add_argument("--load-delay-sec", type=float, default=2.0)
    parser.add_argument("--velocity-mps", type=float, default=4.0)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--map-center", type=parse_vector3)
    parser.add_argument("--map-size", type=parse_size3)
    parser.add_argument("--map-margin-m", type=float, default=20.0)
    parser.add_argument("--min-map-size-m", type=float, default=50.0)
    parser.add_argument("--ground-z-ned", type=float, default=0.0)
    parser.add_argument("--waypoint-spacing-m", type=float, default=3.0)
    parser.add_argument(
        "--flight-driver",
        choices=["velocity", "path-api"],
        default="velocity",
        help=(
            "Use velocity segments for PX4-friendly waypoint following, or use "
            "Project AirSim's MoveToPosition/MoveOnPath APIs directly."
        ),
    )
    parser.add_argument("--waypoint-acceptance-m", type=float, default=1.0)
    parser.add_argument(
        "--no-face-travel-direction",
        dest="face_travel_direction",
        action="store_false",
        default=True,
        help="Keep the existing yaw instead of yawing toward each waypoint.",
    )
    parser.add_argument("--lookahead-m", type=float, default=-1.0)
    parser.add_argument("--adaptive-lookahead", type=float, default=1.0)
    parser.add_argument("--takeoff-timeout-sec", type=float, default=20.0)
    parser.add_argument("--land-timeout-sec", type=float, default=30.0)
    parser.add_argument("--arm-timeout-sec", type=float, default=60.0)
    parser.add_argument("--px4-ready-timeout-sec", type=float, default=60.0)
    parser.add_argument("--move-timeout-sec", type=float, default=45.0)
    parser.add_argument(
        "--path-timeout-sec",
        type=float,
        default=0.0,
        help="Path API timeout. Use 0 to infer from path length and velocity.",
    )
    parser.add_argument("--pose-report-interval-sec", type=float, default=2.0)
    parser.add_argument(
        "--live-ned-interval-sec",
        type=float,
        default=0.5,
        help="How often to print live actual drone NED pose in the terminal.",
    )
    parser.add_argument(
        "--no-live-ned",
        action="store_true",
        help="Disable live actual drone NED pose printing in the terminal.",
    )
    parser.add_argument(
        "--flight-trace-output",
        default="px4_flight_trace.png",
        help="PNG path for plotting actual drone NED position during flight. Use '' to disable.",
    )
    parser.add_argument(
        "--flight-trace-interval-sec",
        type=float,
        default=0.5,
        help="Minimum time between actual NED pose samples used in the flight trace plot.",
    )
    parser.add_argument("--after-teleport-delay-sec", type=float, default=2.0)
    parser.add_argument(
        "--ignore-actor",
        action="append",
        default=None,
        help="Actor to ignore in the occupancy grid. Repeatable.",
    )
    parser.add_argument("--teleport-start", action="store_true")
    parser.add_argument("--skip-takeoff", action="store_true")
    parser.add_argument("--land-at-goal", action="store_true")
    parser.add_argument("--keep-armed", action="store_true")
    parser.add_argument("--print-waypoints", action="store_true")
    parser.add_argument(
        "--show-chase-camera",
        action="store_true",
        help="Open the drone-mounted Chase camera in a separate OpenCV window.",
    )
    parser.add_argument("--chase-camera-width", type=int, default=1280)
    parser.add_argument("--chase-camera-height", type=int, default=720)
    return parser


if __name__ == "__main__":
    asyncio.run(run_autopilot(build_parser().parse_args(normalize_vector_args(sys.argv[1:]))))
