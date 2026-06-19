"""
PX4-backed A* waypoint autopilot example.

Run this with the Project AirSim server and PX4 SITL already running. The script
loads a PX4 scene, builds an occupancy grid, plans a NED path with A*, and sends
that path through Project AirSim's PX4 offboard movement API.
"""

import argparse
import asyncio
import commentjson
import math
import time
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.drone import YawControlMode
from projectairsim.planners import AStarPlanner
from projectairsim.types import Pose, Quaternion, Vector3
from projectairsim.utils import calculate_path_length, projectairsim_log

keyboard = None


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


def format_scene_origin_xyz(position_ned: Sequence[float]) -> str:
    return " ".join(f"{component:g}" for component in position_ned)


def resolve_scene_config_path(scene: str, sim_config_path: str) -> Path:
    scene_path = Path(scene)
    if scene_path.is_absolute():
        return scene_path

    config_dir = Path(sim_config_path)
    cwd_candidate = config_dir / scene_path
    if cwd_candidate.exists():
        return cwd_candidate

    script_candidate = Path(__file__).resolve().parent / config_dir / scene_path
    if script_candidate.exists():
        return script_candidate

    return cwd_candidate


def create_start_origin_scene_config(
    scene: str,
    sim_config_path: str,
    drone_name: str,
    start: Sequence[float],
) -> Tuple[str, str, Path]:
    source_path = resolve_scene_config_path(scene, sim_config_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Scene config not found: {source_path}")

    scene_config = commentjson.loads(source_path.read_text(encoding="utf-8"))
    target_actor = None
    for actor in scene_config.get("actors", []):
        if actor.get("type") == "robot" and actor.get("name") == drone_name:
            target_actor = actor
            break

    if target_actor is None:
        raise RuntimeError(
            f"Could not find robot actor '{drone_name}' in {source_path}"
        )

    origin = target_actor.setdefault("origin", {})
    origin["xyz"] = format_scene_origin_xyz(start)

    output_name = f"{source_path.stem}_runtime_start{source_path.suffix or '.jsonc'}"
    output_path = source_path.with_name(output_name)
    output_path.write_text(
        commentjson.dumps(scene_config, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_path.name, str(output_path.parent), output_path


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


def vector_magnitude(values: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in values))


def limit_vector_delta(
    current: Sequence[float],
    target: Sequence[float],
    max_delta: float,
) -> List[float]:
    if max_delta <= 0.0:
        return [float(target[0]), float(target[1]), float(target[2])]

    delta = [target[idx] - current[idx] for idx in range(3)]
    delta_magnitude = vector_magnitude(delta)
    if delta_magnitude <= max_delta or delta_magnitude == 0.0:
        return [float(target[0]), float(target[1]), float(target[2])]

    scale = max_delta / delta_magnitude
    return [current[idx] + delta[idx] * scale for idx in range(3)]


def move_scalar_toward(current: float, target: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return target

    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


async def teleport_and_verify(
    drone: Drone,
    target: Sequence[float],
    delay_sec: float,
    live_ned=None,
    flight_trace=None,
    label: str = "Teleport start",
    tolerance_m: float = 2.0,
    attempts: int = 3,
    strict: bool = False,
):
    attempts = max(1, attempts)
    actual = None
    error = math.inf

    for attempt in range(1, attempts + 1):
        projectairsim_log().info(
            "Teleporting to %s, attempt %d/%d",
            target,
            attempt,
            attempts,
        )
        drone.set_pose(make_pose_ned(target), reset_kinematics=True)
        await asyncio.sleep(delay_sec)

        actual = get_pose_position_ned(drone)
        error = distance_between(actual, target)
        if flight_trace:
            flight_trace.record(label, actual, force=True)
        if live_ned:
            live_ned.show(label, actual, target=target, remaining_m=error, force=True)

        if error <= tolerance_m:
            projectairsim_log().info(
                "Teleport verified: requested=%s actual=%s error=%.2fm",
                target,
                actual,
                error,
            )
            return actual

        projectairsim_log().info(
            "Teleport not settled yet: requested=%s actual=%s error=%.2fm",
            target,
            actual,
            error,
        )

    message = (
        f"Teleport did not reach requested NED within {tolerance_m:.2f}m: "
        f"requested={target}, actual={actual}, error={error:.2f}m. "
        "If only z differs, the vehicle may be colliding with terrain or takeoff "
        "logic may be changing altitude."
    )
    if strict:
        raise RuntimeError(message)

    projectairsim_log().info("WARNING: %s", message)
    return actual


def format_vector3(values: Sequence[float]) -> str:
    return f"[{values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}]"


def get_pose_position_ned(drone: Drone) -> List[float]:
    position = drone.get_ground_truth_kinematics()["pose"]["position"]
    return [float(position["x"]), float(position["y"]), float(position["z"])]


def log_pose(drone: Drone, label: str) -> List[float]:
    position = get_pose_position_ned(drone)
    projectairsim_log().info(f"{label} pose NED {format_vector3(position)}")
    return position


def load_keyboard_module():
    global keyboard
    if keyboard is not None:
        return keyboard

    try:
        import keyboard as keyboard_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Keyboard control mode needs the 'keyboard' Python package. "
            "Install it in DroneSimDev_ENV with: pip install keyboard"
        ) from exc

    keyboard = keyboard_module
    return keyboard


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
    command_duration_sec: float,
    acceleration_limit_mps2: float,
    slowdown_distance_m: float,
    initial_velocity: Sequence[float] = None,
    stop_at_target: bool = True,
    request_control: bool = True,
    flight_trace: FlightTrace = None,
    live_ned: LiveNedDisplay = None,
) -> List[float]:
    if request_control:
        await request_px4_control(drone)
    started_at = time.time()
    last_report_at = 0.0
    commanded_velocity = (
        [float(initial_velocity[0]), float(initial_velocity[1]), float(initial_velocity[2])]
        if initial_velocity is not None
        else [0.0, 0.0, 0.0]
    )
    command_duration_sec = max(0.05, command_duration_sec)
    max_velocity_delta = max(0.0, acceleration_limit_mps2) * command_duration_sec
    slowdown_distance_m = max(acceptance_m, slowdown_distance_m)

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
            if stop_at_target:
                stop_velocity = [0.0, 0.0, 0.0]
                commanded_velocity = limit_vector_delta(
                    commanded_velocity,
                    stop_velocity,
                    max_velocity_delta,
                )
                await drone.move_by_velocity_async(
                    v_north=commanded_velocity[0],
                    v_east=commanded_velocity[1],
                    v_down=commanded_velocity[2],
                    duration=command_duration_sec,
                    yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
                    yaw_is_rate=True,
                    yaw=0.0,
                )
                await asyncio.sleep(command_duration_sec)
            if flight_trace:
                flight_trace.record(label, current, force=True)
            if live_ned:
                live_ned.show(label, current, target, distance, force=True)
            projectairsim_log().info(
                f"{label} reached target {format_vector3(target)}; pose NED "
                f"{format_vector3(current)}; error {distance:.2f} m"
            )
            return commanded_velocity

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

        speed_scale = min(1.0, max(0.2, distance / slowdown_distance_m))
        desired_speed = min(velocity_mps, distance / command_duration_sec) * speed_scale
        desired_velocity = [(component / distance) * desired_speed for component in delta]
        commanded_velocity = limit_vector_delta(
            commanded_velocity,
            desired_velocity,
            max_velocity_delta,
        )
        yaw_control_mode = (
            YawControlMode.ForwardOnly
            if face_travel_direction
            else YawControlMode.MaxDegreeOfFreedom
        )

        await drone.move_by_velocity_async(
            v_north=commanded_velocity[0],
            v_east=commanded_velocity[1],
            v_down=commanded_velocity[2],
            duration=command_duration_sec,
            yaw_control_mode=yaw_control_mode,
            yaw_is_rate=not face_travel_direction,
            yaw=0.0,
        )
        await asyncio.sleep(command_duration_sec)


async def fly_path_by_velocity(
    drone: Drone,
    path: List[List[float]],
    velocity_mps: float,
    acceptance_m: float,
    timeout_sec: float,
    report_interval_sec: float,
    face_travel_direction: bool,
    command_duration_sec: float,
    acceleration_limit_mps2: float,
    slowdown_distance_m: float,
    flight_trace: FlightTrace = None,
    live_ned: LiveNedDisplay = None,
):
    await request_px4_control(drone)
    commanded_velocity = [0.0, 0.0, 0.0]
    final_waypoint_index = len(path) - 1
    for index, waypoint in enumerate(path[1:], start=1):
        commanded_velocity = await fly_to_point_by_velocity(
            drone,
            waypoint,
            velocity_mps,
            acceptance_m,
            timeout_sec,
            report_interval_sec,
            face_travel_direction,
            f"Waypoint {index:03d}",
            command_duration_sec,
            acceleration_limit_mps2,
            slowdown_distance_m,
            commanded_velocity,
            index == final_waypoint_index,
            False,
            flight_trace,
            live_ned,
        )


async def run_px4_keyboard_control(
    drone: Drone,
    velocity_mps: float,
    yaw_rate_dps: float,
    command_duration_sec: float,
    acceleration_limit_mps2: float,
    yaw_acceleration_dps2: float,
    live_ned: LiveNedDisplay,
    flight_trace: FlightTrace = None,
):
    keyboard_module = load_keyboard_module()
    await request_px4_control(drone)
    command_duration_sec = max(0.05, command_duration_sec)
    max_velocity_delta = max(0.0, acceleration_limit_mps2) * command_duration_sec
    max_yaw_delta = max(0.0, yaw_acceleration_dps2) * command_duration_sec
    commanded_velocity = [0.0, 0.0, 0.0]
    commanded_yaw_rate = 0.0

    print("\n--- PX4 Keyboard Control ---")
    print("W/S: forward/backward")
    print("A/D: left/right")
    print("Up/Down Arrows: up/down altitude")
    print("Left/Right Arrows: yaw left/right")
    print("L: land and exit")
    print("Q: quit without landing")
    print("----------------------------")

    while True:
        current = get_pose_position_ned(drone)
        if flight_trace:
            flight_trace.record("Keyboard control", current)
        if live_ned:
            live_ned.show("Keyboard control", current)

        target_velocity = [0.0, 0.0, 0.0]
        target_yaw_rate = 0.0

        if keyboard_module.is_pressed("w"):
            target_velocity[0] = velocity_mps
        elif keyboard_module.is_pressed("s"):
            target_velocity[0] = -velocity_mps

        if keyboard_module.is_pressed("a"):
            target_velocity[1] = -velocity_mps
        elif keyboard_module.is_pressed("d"):
            target_velocity[1] = velocity_mps

        if keyboard_module.is_pressed("up"):
            target_velocity[2] = -velocity_mps
        elif keyboard_module.is_pressed("down"):
            target_velocity[2] = velocity_mps

        if keyboard_module.is_pressed("left"):
            target_yaw_rate = -yaw_rate_dps
        elif keyboard_module.is_pressed("right"):
            target_yaw_rate = yaw_rate_dps

        if keyboard_module.is_pressed("l"):
            projectairsim_log().info("Keyboard requested landing")
            land_task = await drone.land_async()
            await await_drone_task(
                drone,
                land_task,
                "Keyboard landing",
                30.0,
                1.0,
                flight_trace,
                live_ned,
            )
            return

        if keyboard_module.is_pressed("q"):
            projectairsim_log().info("Keyboard requested quit")
            return

        commanded_velocity = limit_vector_delta(
            commanded_velocity,
            target_velocity,
            max_velocity_delta,
        )
        commanded_yaw_rate = move_scalar_toward(
            commanded_yaw_rate,
            target_yaw_rate,
            max_yaw_delta,
        )

        if vector_magnitude(commanded_velocity) > 0.05 or vector_magnitude(target_velocity) > 0.0:
            await drone.move_by_velocity_body_frame_async(
                commanded_velocity[0],
                commanded_velocity[1],
                commanded_velocity[2],
                command_duration_sec,
            )

        if abs(commanded_yaw_rate) > 0.1 or abs(target_yaw_rate) > 0.0:
            await drone.rotate_by_yaw_rate_async(
                commanded_yaw_rate,
                command_duration_sec,
            )

        await asyncio.sleep(0.02)


async def run_autopilot(args):
    scene = args.scene
    sim_config_path = args.sim_config_path
    drone = None
    cleanup_needed = False
    start_as_scene_origin = False
    if args.start_as_scene_origin:
        if args.start is None:
            raise RuntimeError("--start-as-scene-origin requires --start")

        scene, sim_config_path, generated_scene_path = create_start_origin_scene_config(
            args.scene,
            args.sim_config_path,
            args.drone_name,
            args.start,
        )
        start_as_scene_origin = True
        projectairsim_log().info(
            "Generated PX4 start scene %s with %s origin xyz=%s",
            generated_scene_path,
            args.drone_name,
            format_scene_origin_xyz(args.start),
        )

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
            scene,
            delay_after_load_sec=args.load_delay_sec,
            sim_config_path=sim_config_path,
        )
        if start_as_scene_origin:
            projectairsim_log().info(
                "Loaded %s with %s spawned at --start. If PX4 was already "
                "running before this scene loaded, restart PX4 now and let "
                "this script keep waiting for the vehicle.",
                scene,
                args.drone_name,
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
        runtime_teleport_start = args.teleport_start and not start_as_scene_origin
        if args.teleport_start and start_as_scene_origin:
            projectairsim_log().info(
                "Ignoring --teleport-start because --start-as-scene-origin "
                "already placed the actor at --start before PX4 connects."
            )

        if args.keyboard_control:
            await wait_for_px4_ready(drone, args.px4_ready_timeout_sec)

            if runtime_teleport_start:
                if not start:
                    raise RuntimeError("--teleport-start requires --start in keyboard mode")
                await teleport_and_verify(
                    drone,
                    start,
                    args.after_teleport_delay_sec,
                    live_ned,
                    flight_trace,
                    tolerance_m=args.teleport_tolerance_m,
                    attempts=args.teleport_attempts,
                    strict=args.strict_teleport,
                )

            before_arming_pose = log_pose(drone, "Before arming")
            if flight_trace:
                flight_trace.record("Before arming", before_arming_pose, force=True)
            live_ned.show("Before arming", before_arming_pose, force=True)

            if not drone.enable_api_control():
                raise RuntimeError("Project AirSim did not enable API control")
            cleanup_needed = True
            await arm_with_retry(drone, args.arm_timeout_sec)

            if not args.skip_takeoff:
                projectairsim_log().info("Taking off")
                takeoff_task = await drone.takeoff_async(
                    timeout_sec=args.takeoff_timeout_sec
                )
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
                    flight_trace.record(
                        "Skipping takeoff",
                        skipping_takeoff_pose,
                        force=True,
                    )
                live_ned.show("Skipping takeoff", skipping_takeoff_pose, force=True)

            await run_px4_keyboard_control(
                drone,
                args.keyboard_speed_mps,
                args.keyboard_yaw_rate_dps,
                args.keyboard_command_duration_sec,
                args.keyboard_acceleration_limit_mps2,
                args.keyboard_yaw_acceleration_dps2,
                live_ned,
                flight_trace,
            )

            if not args.keep_armed:
                drone.disarm()
                drone.disable_api_control()
                cleanup_needed = False
            return

        map_center = args.map_center or infer_map_center(start, goal)
        map_size = args.map_size or infer_map_size(
            start, goal, args.map_margin_m, args.min_map_size_m
        )

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

        if args.plan_only:
            projectairsim_log().info(
                "Plan-only mode requested; skipping PX4 readiness, arming, and flight"
            )
            return

        await wait_for_px4_ready(drone, args.px4_ready_timeout_sec)
        if runtime_teleport_start:
            await teleport_and_verify(
                drone,
                start,
                args.after_teleport_delay_sec,
                live_ned,
                flight_trace,
                tolerance_m=args.teleport_tolerance_m,
                attempts=args.teleport_attempts,
                strict=args.strict_teleport,
            )
        before_arming_pose = log_pose(drone, "Before arming")
        if flight_trace:
            flight_trace.record("Before arming", before_arming_pose, force=True)
        live_ned.show("Before arming", before_arming_pose, force=True)

        if not drone.enable_api_control():
            raise RuntimeError("Project AirSim did not enable API control")
        cleanup_needed = True
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

        if not runtime_teleport_start:
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
                    args.velocity_command_duration_sec,
                    args.acceleration_limit_mps2,
                    args.slowdown_distance_m,
                    None,
                    True,
                    True,
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
                args.velocity_command_duration_sec,
                args.acceleration_limit_mps2,
                args.slowdown_distance_m,
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
            cleanup_needed = False

    finally:
        if cleanup_needed and drone is not None and not args.keep_armed:
            try:
                projectairsim_log().info("Cleaning up PX4 control before exit")
                drone.cancel_last_task()
                drone.disarm()
                drone.disable_api_control()
            except Exception as cleanup_error:
                projectairsim_log().warning(
                    "PX4 cleanup failed during script exit: %s",
                    cleanup_error,
                )
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
    parser.add_argument("--start", type=parse_vector3, help="NED x,y,z")
    parser.add_argument("--goal", type=parse_vector3, help="NED x,y,z")
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
    parser.add_argument(
        "--keyboard-control",
        action="store_true",
        help="Use PX4 keyboard/manual control mode instead of A* planning.",
    )
    parser.add_argument(
        "--keyboard-speed-mps",
        type=float,
        default=4.0,
        help="Body-frame speed used by --keyboard-control.",
    )
    parser.add_argument(
        "--keyboard-yaw-rate-dps",
        type=float,
        default=30.0,
        help="Yaw rate in degrees/s used by --keyboard-control.",
    )
    parser.add_argument(
        "--keyboard-command-duration-sec",
        type=float,
        default=0.1,
        help="Duration of each keyboard velocity/yaw command.",
    )
    parser.add_argument(
        "--keyboard-acceleration-limit-mps2",
        type=float,
        default=6.0,
        help="Maximum keyboard-mode velocity change rate.",
    )
    parser.add_argument(
        "--keyboard-yaw-acceleration-dps2",
        type=float,
        default=120.0,
        help="Maximum keyboard-mode yaw-rate change rate.",
    )
    parser.add_argument(
        "--start-as-scene-origin",
        action="store_true",
        help=(
            "Generate and load a runtime scene config with the drone actor "
            "origin set to --start. This is the preferred PX4 start workflow; "
            "it avoids runtime set_pose teleporting after PX4 connects."
        ),
    )
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--map-center", type=parse_vector3)
    parser.add_argument("--map-size", type=parse_size3)
    parser.add_argument("--map-margin-m", type=float, default=20.0)
    parser.add_argument("--min-map-size-m", type=float, default=50.0)
    parser.add_argument("--ground-z-ned", type=float, default=0.0)
    parser.add_argument("--waypoint-spacing-m", type=float, default=3.0)
    parser.add_argument(
        "--velocity-command-duration-sec",
        type=float,
        default=0.1,
        help="Duration of each PX4 velocity setpoint in velocity flight mode.",
    )
    parser.add_argument(
        "--acceleration-limit-mps2",
        type=float,
        default=2.0,
        help="Maximum velocity change rate in velocity flight mode.",
    )
    parser.add_argument(
        "--slowdown-distance-m",
        type=float,
        default=4.0,
        help="Distance over which velocity mode eases down near each waypoint.",
    )
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
    parser.set_defaults(face_travel_direction=False)
    parser.add_argument(
        "--face-travel-direction",
        dest="face_travel_direction",
        action="store_true",
        help="Yaw toward each waypoint in velocity/path flight modes.",
    )
    parser.add_argument(
        "--no-face-travel-direction",
        dest="face_travel_direction",
        action="store_false",
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
        "--teleport-tolerance-m",
        type=float,
        default=2.0,
        help="Allowed distance between requested and actual NED after teleport.",
    )
    parser.add_argument(
        "--teleport-attempts",
        type=int,
        default=3,
        help="Number of times to retry teleport before warning or failing.",
    )
    parser.add_argument(
        "--strict-teleport",
        action="store_true",
        help="Fail if actual NED is not within --teleport-tolerance-m after teleport.",
    )
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
        "--plan-only",
        action="store_true",
        help="Build the occupancy grid and A* path, then exit before PX4 arming/flight.",
    )
    parser.add_argument(
        "--show-chase-camera",
        action="store_true",
        help="Open the drone-mounted Chase camera in a separate OpenCV window.",
    )
    parser.add_argument("--chase-camera-width", type=int, default=1280)
    parser.add_argument("--chase-camera-height", type=int, default=720)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args(normalize_vector_args(sys.argv[1:]))
    if not parsed_args.keyboard_control:
        if parsed_args.start is None or parsed_args.goal is None:
            parser.error("--start and --goal are required unless --keyboard-control is used")
    asyncio.run(run_autopilot(parsed_args))
