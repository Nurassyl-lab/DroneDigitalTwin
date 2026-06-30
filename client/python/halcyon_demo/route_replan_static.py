"""
Manual route following with live static-obstacle rerouting for a PX4-style drone.

The user provides a route as [START, NED1, NED2, ..., END], and the drone
follows those points in order. While flying, it scans a short lookahead window
for occupied objects on the route; if one is found, A* replaces the blocked
segment and rejoins the original downstream route.
"""

import argparse
import asyncio
import math
import queue
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Sequence, Tuple

import commentjson

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.drone import YawControlMode
from projectairsim.planners import AStarPlanner
from projectairsim.types import Pose, Quaternion, Vector3
from projectairsim.utils import (
    calculate_path_length,
    projectairsim_log,
    rpy_to_quaternion,
    unpack_image,
)

# px4_astar_autopilot is located in ../example_user_scripts/px4_astar_autopilot.py, so we add that directory to sys.path to import it
# W:\UnsyncProjects\DroneSimDev\client\python\
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "example_user_scripts")
)

from px4_astar_autopilot import (
    brake_to_stop_by_velocity,
    clamp,
    fly_path_by_velocity,
    fly_to_point_by_velocity,
    hold_waypoint_by_velocity,
    infer_map_center,
    infer_map_size,
    limit_vector_delta,
    parse_size3,
    sparsify_path,
    validate_grid_coordinate,
    wrap_angle_rad,
)


FRIENDLY_CAMERA_IDS = {
    "rgb": "FrontCamera",
    "front_rgb": "FrontCamera",
    "front": "FrontCamera",
    "down_rgb": "DownCamera",
    "down": "DownCamera",
    "chase": "Chase",
    "chase_rgb": "Chase",
}


@dataclass
class Waypoint:
    label: str
    position: List[float]
    status: str = "active"


@dataclass
class StaticRouteObstacle:
    obstacle_point: List[float]
    stop_point: List[float]
    route_distance_m: float
    stop_distance_m: float
    segment_index: int
    rejoin_index: Optional[int] = None
    rejoin_point: Optional[List[float]] = None
    skipped_points: Optional[List[List[float]]] = None


@dataclass
class StaticRouteScan:
    planner: AStarPlanner
    map_center: List[float]
    map_size: List[float]
    resolution_m: float


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


def parse_route(value: str) -> List[List[float]]:
    route = []
    for index, point_text in enumerate(value.split(";")):
        point_text = point_text.strip()
        if not point_text:
            continue
        try:
            route.append(parse_vector3(point_text))
        except argparse.ArgumentTypeError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid route point {index}: {exc}"
            ) from exc

    if len(route) < 2:
        raise argparse.ArgumentTypeError(
            "--route must contain at least START and END points separated by ';'"
        )
    return route


def normalize_vector_args(argv: Sequence[str]) -> List[str]:
    normalized = []
    vector_options = {
        "--start",
        "--goal",
        "--waypoint",
        "--map-center",
        "--map-size",
        "--route",
        "--replan-rejoin-point",
        "--replan-emergency-node",
    }
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


def format_scene_origin_xyz(position_ned: Sequence[float]) -> str:
    return " ".join(f"{component:g}" for component in position_ned)


def format_vector3(values: Sequence[float]) -> str:
    return f"[{values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}]"


def distance_between(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((a[idx] - b[idx]) ** 2 for idx in range(3)))


def interpolate_point(
    a: Sequence[float],
    b: Sequence[float],
    fraction: float,
) -> List[float]:
    fraction = max(0.0, min(1.0, fraction))
    return [float(a[idx]) + (float(b[idx]) - float(a[idx])) * fraction for idx in range(3)]


def cumulative_route_distances(path: Sequence[Sequence[float]]) -> List[float]:
    distances = [0.0]
    for index in range(1, len(path)):
        distances.append(distances[-1] + distance_between(path[index - 1], path[index]))
    return distances


def point_at_route_distance(
    path: Sequence[Sequence[float]],
    target_distance_m: float,
) -> Tuple[List[float], int]:
    if not path:
        raise ValueError("Path must contain at least one point")

    if target_distance_m <= 0.0 or len(path) == 1:
        return [float(value) for value in path[0]], 0

    traveled = 0.0
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        segment_length = distance_between(start, end)
        if segment_length <= 1e-6:
            continue
        if traveled + segment_length >= target_distance_m:
            fraction = (target_distance_m - traveled) / segment_length
            return interpolate_point(start, end, fraction), index - 1
        traveled += segment_length

    return [float(value) for value in path[-1]], max(0, len(path) - 2)


def iter_route_samples(
    path: Sequence[Sequence[float]],
    spacing_m: float,
):
    spacing_m = max(0.1, spacing_m)
    total_length = calculate_path_length(path)
    distance_m = 0.0
    while distance_m < total_length:
        point, segment_index = point_at_route_distance(path, distance_m)
        yield distance_m, point, segment_index
        distance_m += spacing_m
    point, segment_index = point_at_route_distance(path, total_length)
    yield total_length, point, segment_index


def truncate_route_at_distance(
    path: Sequence[Sequence[float]],
    target_distance_m: float,
) -> List[List[float]]:
    target_point, _ = point_at_route_distance(path, target_distance_m)
    if len(path) <= 1:
        return [target_point]

    truncated = [[float(value) for value in path[0]]]
    traveled = 0.0
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        segment_length = distance_between(start, end)
        if segment_length <= 1e-6:
            continue
        next_distance = traveled + segment_length
        if next_distance < target_distance_m:
            truncated.append([float(value) for value in end])
            traveled = next_distance
            continue
        if distance_between(truncated[-1], target_point) > 1e-3:
            truncated.append(target_point)
        return truncated

    if distance_between(truncated[-1], target_point) > 1e-3:
        truncated.append(target_point)
    return truncated


def append_unique_point(
    path: List[List[float]],
    point: Sequence[float],
    min_distance_m: float = 1e-3,
) -> None:
    candidate = [float(point[0]), float(point[1]), float(point[2])]
    if not path or distance_between(path[-1], candidate) > min_distance_m:
        path.append(candidate)


def densify_path(
    path: Sequence[Sequence[float]],
    max_spacing_m: float,
) -> List[List[float]]:
    if len(path) <= 1 or max_spacing_m <= 0.0:
        return [[float(point[0]), float(point[1]), float(point[2])] for point in path]

    dense_path = [[float(path[0][0]), float(path[0][1]), float(path[0][2])]]
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        segment_length = distance_between(start, end)
        steps = max(1, math.ceil(segment_length / max_spacing_m))
        for step in range(1, steps + 1):
            append_unique_point(dense_path, interpolate_point(start, end, step / steps))
    return dense_path


def route_waypoint_index_at_distance(
    path: Sequence[Sequence[float]],
    distance_m: float,
    minimum_index: int = 0,
) -> int:
    route_distances = cumulative_route_distances(path)
    for index, route_distance_m in enumerate(route_distances):
        if index < minimum_index:
            continue
        if route_distance_m >= distance_m:
            return index
    return len(path) - 1


def route_scan_volume(
    path: Sequence[Sequence[float]],
    margin_m: float,
    min_size_m: float,
    resolution_m: float,
) -> Tuple[List[float], List[float]]:
    margin_m = max(0.0, margin_m)
    min_size_m = max(resolution_m, min_size_m)
    resolution_m = max(0.1, resolution_m)
    center = []
    size = []
    for axis in range(3):
        values = [point[axis] for point in path]
        axis_min = min(values) - margin_m
        axis_max = max(values) + margin_m
        axis_size = max(min_size_m, axis_max - axis_min)
        axis_size = math.ceil(axis_size / resolution_m) * resolution_m
        center.append((axis_min + axis_max) * 0.5)
        size.append(axis_size)
    return center, size


def horizontal_offsets(radius_m: float, resolution_m: float) -> List[Tuple[float, float]]:
    if radius_m <= 0.0:
        return [(0.0, 0.0)]

    step = max(0.25, min(radius_m, resolution_m))
    offsets = [(0.0, 0.0)]
    cells = max(1, math.ceil(radius_m / step))
    for x_index in range(-cells, cells + 1):
        for y_index in range(-cells, cells + 1):
            dx = x_index * step
            dy = y_index * step
            if dx == 0.0 and dy == 0.0:
                continue
            if math.hypot(dx, dy) <= radius_m + 1e-6:
                offsets.append((dx, dy))
    return offsets


def is_route_corridor_clear(planner: AStarPlanner, point: Sequence[float], args) -> bool:
    for dx, dy in horizontal_offsets(
        args.object_path_clearance_m,
        args.object_scan_resolution_m,
    ):
        candidate = [point[0] + dx, point[1] + dy, point[2]]
        if not planner.check_coordinate_validity(candidate, is_NED=True):
            return False
    return True


def parse_float3(value: str) -> List[float]:
    parts = str(value).replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Expected three values, got '{value}'")
    return [float(part) for part in parts]


def resolve_config_path(config_name: str, sim_config_path: str) -> Path:
    config_path = Path(config_name)
    if config_path.is_absolute():
        return config_path

    config_dir = Path(sim_config_path)
    candidates = [
        config_dir / config_path,
        Path(__file__).resolve().parent / config_dir / config_path,
        Path(__file__).resolve().parent.parent / "example_user_scripts" / config_dir / config_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def prepare_video_output_path(video_path: str) -> Optional[Path]:
    if not video_path:
        projectairsim_log().info("FPV video recording disabled")
        return None

    requested_path = Path(video_path).expanduser()
    if requested_path.suffix.lower() == ".mp4":
        output_dir = requested_path.parent
        output_path = requested_path
    else:
        output_dir = requested_path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"fpv_route_overlay_{timestamp}.mp4"

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        projectairsim_log().info("Created FPV video directory: %s", output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f"FPV video path is not a directory: {output_dir}")

    return output_path


def load_jsonc(path: Path):
    return commentjson.loads(path.read_text(encoding="utf-8"))


def camera_arg_to_sensor_id(camera: str) -> str:
    return FRIENDLY_CAMERA_IDS.get(camera, camera)


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


def default_front_camera_sensor(args) -> Dict:
    return {
        "id": "FrontCamera",
        "type": "camera",
        "enabled": True,
        "parent-link": "Frame",
        "capture-interval": args.camera_capture_interval_sec,
        "capture-settings": [
            {
                "image-type": 0,
                "width": args.camera_capture_width,
                "height": args.camera_capture_height,
                "fov-degrees": args.camera_fov_degrees,
                "capture-enabled": True,
                "streaming-enabled": True,
                "pixels-as-float": False,
                "compress": False,
                "target-gamma": 2.5,
            }
        ],
        "origin": {
            "xyz": "0.5 0.0 0.0",
            "rpy-deg": "0 0 0",
        },
    }


def ensure_scene_camera_capture(sensor: Dict, args) -> None:
    capture_settings = sensor.setdefault("capture-settings", [])
    scene_capture = next(
        (capture for capture in capture_settings if capture.get("image-type") == 0),
        None,
    )
    if scene_capture is None:
        capture_settings.append(
            {
                "image-type": 0,
                "width": args.camera_capture_width,
                "height": args.camera_capture_height,
                "fov-degrees": args.camera_fov_degrees,
                "capture-enabled": True,
                "streaming-enabled": True,
                "pixels-as-float": False,
                "compress": False,
                "target-gamma": 2.5,
            }
        )
        return

    scene_capture["capture-enabled"] = True
    scene_capture["streaming-enabled"] = True
    scene_capture.setdefault("width", args.camera_capture_width)
    scene_capture.setdefault("height", args.camera_capture_height)
    scene_capture.setdefault("fov-degrees", args.camera_fov_degrees)
    scene_capture.setdefault("pixels-as-float", False)
    scene_capture.setdefault("compress", False)
    scene_capture.setdefault("target-gamma", 2.5)


def ensure_requested_camera(robot_config: Dict, camera_sensor_id: str, args) -> None:
    sensors = robot_config.setdefault("sensors", [])
    sensor = next(
        (
            candidate
            for candidate in sensors
            if candidate.get("id") == camera_sensor_id
        ),
        None,
    )

    if sensor is None and camera_sensor_id == "FrontCamera":
        sensors.append(default_front_camera_sensor(args))
        projectairsim_log().info(
            "Runtime config added FrontCamera for --camera %s",
            args.camera,
        )
        return

    if sensor is None:
        return

    if sensor.get("type") == "camera":
        sensor["enabled"] = True
        ensure_scene_camera_capture(sensor, args)


def make_runtime_scene_config(args, camera_sensor_id: str):
    scene_path = resolve_config_path(args.scene, args.sim_config_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene config not found: {scene_path}")

    scene_config = load_jsonc(scene_path)
    actors = scene_config.get("actors", [])
    target_actor = next(
        (
            actor
            for actor in actors
            if actor.get("type") == "robot" and actor.get("name") == args.drone_name
        ),
        None,
    )
    if target_actor is None or not target_actor.get("robot-config"):
        raise RuntimeError(
            f"Could not find robot actor '{args.drone_name}' with a robot-config"
        )

    if args.start_as_scene_origin:
        if args.start is None:
            raise RuntimeError("--start-as-scene-origin requires --start")
        target_actor.setdefault("origin", {})["xyz"] = format_scene_origin_xyz(args.start)

    temp_dir = tempfile.TemporaryDirectory(prefix="fpv_route_overlay_")
    temp_config_dir = Path(temp_dir.name)

    try:
        for actor_index, actor in enumerate(actors):
            if actor.get("type") != "robot" or not actor.get("robot-config"):
                continue

            robot_config_path = resolve_config_path(
                actor["robot-config"],
                str(scene_path.parent),
            )
            robot_config = load_jsonc(robot_config_path)
            if actor is target_actor:
                ensure_requested_camera(robot_config, camera_sensor_id, args)

            suffix = robot_config_path.suffix or ".jsonc"
            output_name = f"{robot_config_path.stem}_{actor_index}_fpv{suffix}"
            (temp_config_dir / output_name).write_text(
                commentjson.dumps(robot_config, indent=2) + "\n",
                encoding="utf-8",
            )
            actor["robot-config"] = output_name

        for env_actor in scene_config.get("environment-actors", []):
            if env_actor.get("type") != "env_actor" or not env_actor.get(
                "env-actor-config"
            ):
                continue
            env_config_path = resolve_config_path(
                env_actor["env-actor-config"],
                str(scene_path.parent),
            )
            shutil.copy2(env_config_path, temp_config_dir / env_config_path.name)
            env_actor["env-actor-config"] = env_config_path.name

        temp_scene_name = scene_path.name
        (temp_config_dir / temp_scene_name).write_text(
            commentjson.dumps(scene_config, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        temp_dir.cleanup()
        raise

    if args.start_as_scene_origin:
        projectairsim_log().info(
            "Runtime scene starts %s at NED %s",
            args.drone_name,
            format_vector3(args.start),
        )
    return temp_dir, temp_scene_name, str(temp_config_dir)


def find_target_robot_config(scene_name: str, sim_config_path: str, drone_name: str):
    scene_path = resolve_config_path(scene_name, sim_config_path)
    scene_config = load_jsonc(scene_path)
    actor = next(
        (
            item
            for item in scene_config.get("actors", [])
            if item.get("type") == "robot" and item.get("name") == drone_name
        ),
        None,
    )
    if actor is None or not actor.get("robot-config"):
        raise RuntimeError(
            f"Could not find robot actor '{drone_name}' with a robot-config"
        )

    robot_config_path = resolve_config_path(actor["robot-config"], str(scene_path.parent))
    return scene_config, load_jsonc(robot_config_path)


def read_camera_fov_degrees(
    scene_name: str,
    sim_config_path: str,
    drone_name: str,
    camera_sensor_id: str,
    default_fov_degrees: float,
) -> float:
    try:
        _, robot_config = find_target_robot_config(
            scene_name, sim_config_path, drone_name
        )
        sensor = next(
            (
                item
                for item in robot_config.get("sensors", [])
                if item.get("id") == camera_sensor_id
            ),
            None,
        )
        if sensor is None:
            return default_fov_degrees
        scene_capture = next(
            (
                capture
                for capture in sensor.get("capture-settings", [])
                if capture.get("image-type") == 0
            ),
            None,
        )
        if scene_capture is None:
            return default_fov_degrees
        return float(scene_capture.get("fov-degrees", default_fov_degrees))
    except Exception as exc:
        projectairsim_log().warning(
            "Could not read camera FOV for %s; using %.1f deg: %s",
            camera_sensor_id,
            default_fov_degrees,
            exc,
        )
        return default_fov_degrees


def read_camera_origin_from_config(args, camera_sensor_id: str):
    default_origin = ((0.5, 0.0, 0.0), (0.0, 0.0, 0.0))
    try:
        _, robot_config = find_target_robot_config(
            args.effective_scene,
            args.effective_sim_config_path,
            args.drone_name,
        )
        sensor = next(
            (
                item
                for item in robot_config.get("sensors", [])
                if item.get("id") == camera_sensor_id
            ),
            None,
        )
        if sensor is None:
            return default_origin

        origin = sensor.get("origin", {})
        translation = parse_float3(origin.get("xyz", "0 0 0"))
        if origin.get("rpy-deg") is not None:
            rotation = [
                math.radians(component)
                for component in parse_float3(origin["rpy-deg"])
            ]
        else:
            rotation = parse_float3(origin.get("rpy", "0 0 0"))
        return translation, rotation
    except Exception as exc:
        projectairsim_log().warning(
            "Could not read configured origin for %s; using default pose: %s",
            camera_sensor_id,
            exc,
        )
        return default_origin


def make_camera_angle_pose(args, camera_sensor_id: str, angle_deg: float) -> Pose:
    translation_xyz, base_rpy = read_camera_origin_from_config(args, camera_sensor_id)
    roll, _, yaw = base_rpy
    pitch = math.radians(-angle_deg)
    w, x, y, z = rpy_to_quaternion(roll, pitch, yaw)
    return Pose(
        {
            "translation": Vector3(
                {
                    "x": translation_xyz[0],
                    "y": translation_xyz[1],
                    "z": translation_xyz[2],
                }
            ),
            "rotation": Quaternion({"w": w, "x": x, "y": y, "z": z}),
            "frame_id": "DEFAULT_ID",
        }
    )


def require_scene_camera_topic(drone: Drone, camera_sensor_id: str) -> str:
    if camera_sensor_id not in drone.sensors:
        raise RuntimeError(
            f"Camera sensor '{camera_sensor_id}' is not available. Available sensors: "
            f"{sorted(drone.sensors.keys())}"
        )
    if "scene_camera" not in drone.sensors[camera_sensor_id]:
        raise RuntimeError(
            f"Sensor '{camera_sensor_id}' has no scene_camera topic. Available topics: "
            f"{sorted(drone.sensors[camera_sensor_id].keys())}"
        )
    return drone.sensors[camera_sensor_id]["scene_camera"]


def quaternion_to_matrix(w: float, x: float, y: float, z: float):
    magnitude = math.sqrt(w * w + x * x + y * y + z * z)
    if magnitude <= 0.0:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    w /= magnitude
    x /= magnitude
    y /= magnitude
    z /= magnitude

    return [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]


def rotate_vector_by_quaternion(vector: Sequence[float], rotation: Dict) -> List[float]:
    matrix = quaternion_to_matrix(
        float(rotation["w"]),
        float(rotation["x"]),
        float(rotation["y"]),
        float(rotation["z"]),
    )
    return [
        matrix[row][0] * vector[0]
        + matrix[row][1] * vector[1]
        + matrix[row][2] * vector[2]
        for row in range(3)
    ]


def camera_pose_from_image(image) -> Optional[Tuple[List[float], List[List[float]]]]:
    pose_keys = ("pos_x", "pos_y", "pos_z", "rot_w", "rot_x", "rot_y", "rot_z")
    if not all(key in image for key in pose_keys):
        return None

    position = [float(image["pos_x"]), float(image["pos_y"]), float(image["pos_z"])]
    rotation_matrix = quaternion_to_matrix(
        float(image["rot_w"]),
        float(image["rot_x"]),
        float(image["rot_y"]),
        float(image["rot_z"]),
    )
    return position, rotation_matrix


def world_to_camera(
    point_ned: Sequence[float],
    camera_position: Sequence[float],
    camera_rotation_matrix,
) -> List[float]:
    delta = [
        point_ned[0] - camera_position[0],
        point_ned[1] - camera_position[1],
        point_ned[2] - camera_position[2],
    ]
    return [
        camera_rotation_matrix[0][axis] * delta[0]
        + camera_rotation_matrix[1][axis] * delta[1]
        + camera_rotation_matrix[2][axis] * delta[2]
        for axis in range(3)
    ]


def project_waypoint(
    point_ned: Sequence[float],
    image,
    fov_degrees: float,
) -> Optional[Tuple[float, float, float, List[float]]]:
    camera_pose = camera_pose_from_image(image)
    if camera_pose is None:
        return None

    width = int(image["width"])
    height = int(image["height"])
    camera_position, camera_rotation_matrix = camera_pose
    point_camera = world_to_camera(point_ned, camera_position, camera_rotation_matrix)
    distance_m = math.sqrt(sum(component * component for component in point_camera))

    forward_m = point_camera[0]
    if forward_m <= 0.05:
        return None

    focal_px = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    pixel_x = width * 0.5 + focal_px * (point_camera[1] / forward_m)
    pixel_y = height * 0.5 + focal_px * (point_camera[2] / forward_m)
    return pixel_x, pixel_y, distance_m, point_camera


class FpvWaypointOverlayDisplay:
    def __init__(
        self,
        window_name: str,
        waypoints: Sequence[Waypoint],
        fov_degrees: float,
        resize_x: Optional[int],
        resize_y: Optional[int],
        draw_edge_indicators: bool,
        reached_distance_m: float,
        max_fps: float,
        video_output_path: Optional[Path],
    ):
        self.window_name = window_name
        self.waypoints = list(waypoints)
        self.fov_degrees = fov_degrees
        self.resize_x = resize_x
        self.resize_y = resize_y
        self.draw_edge_indicators = draw_edge_indicators
        self.reached_distance_m = max(0.1, reached_distance_m)
        self.reached_waypoints = [False for _ in self.waypoints]
        self.max_fps = max(1.0, max_fps)
        self.video_output_path = video_output_path
        self.video_writer = None
        self.image_queue = queue.SimpleQueue()
        self.buffer_size = 3
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.error = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = Thread(target=self.display_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.thread = None

    def receive(self, image):
        if not self.running or image is None:
            return
        while not self.image_queue.empty() and self.image_queue.qsize() > self.buffer_size:
            self.image_queue.get()
        self.image_queue.put(image)

    def set_waypoints(self, waypoints: Sequence[Waypoint]) -> None:
        self.waypoints = list(waypoints)
        self.reached_waypoints = [False for _ in self.waypoints]

    def display_loop(self):
        import cv2

        created = False
        frame_interval_sec = 1.0 / self.max_fps
        next_frame_at = time.monotonic()
        try:
            while self.running:
                now = time.monotonic()
                if now < next_frame_at:
                    wait_ms = max(1, int((next_frame_at - now) * 1000.0))
                    if cv2.waitKey(wait_ms) == 27:
                        self.running = False
                    continue

                if self.image_queue.empty():
                    if cv2.waitKey(1) == 27:
                        self.running = False
                    continue

                image = self.image_queue.get()
                while not self.image_queue.empty():
                    image = self.image_queue.get()

                frame = unpack_image(image)
                if frame is None:
                    continue
                frame = frame.copy()

                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)

                self.frame_count += 1
                self.draw_overlay(cv2, frame, image)

                if self.resize_x is not None and self.resize_y is not None:
                    frame = cv2.resize(frame, (self.resize_x, self.resize_y))

                self.write_video_frame(cv2, frame)

                if not created:
                    cv2.namedWindow(
                        self.window_name,
                        flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE,
                    )
                    created = True

                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) == 27:
                    self.running = False
                next_frame_at = time.monotonic() + frame_interval_sec
        except Exception as exc:
            self.error = exc
            self.running = False
        finally:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            if created:
                cv2.destroyWindow(self.window_name)

    def write_video_frame(self, cv2, frame):
        if self.video_output_path is None:
            return

        if self.video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                str(self.video_output_path),
                fourcc,
                self.max_fps,
                (width, height),
            )
            if not self.video_writer.isOpened():
                self.video_writer = None
                raise RuntimeError(
                    f"Could not open FPV video writer: {self.video_output_path}"
                )
            projectairsim_log().info("Recording FPV video to %s", self.video_output_path)

        self.video_writer.write(frame)

    def draw_overlay(self, cv2, frame, image):
        height, width = frame.shape[:2]
        cv2.putText(
            frame,
            f"{self.window_name} | waypoints={len(self.waypoints)} | frame={self.frame_count}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{self.window_name} | waypoints={len(self.waypoints)} | frame={self.frame_count}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        camera_pose = camera_pose_from_image(image)
        if camera_pose is None:
            cv2.putText(
                frame,
                "camera pose unavailable in image message",
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return

        camera_position, _ = camera_pose
        status_y = 54
        for index, waypoint in enumerate(self.waypoints):
            is_skipped = waypoint.status == "skipped"
            world_distance_m = distance_between(camera_position, waypoint.position)
            if not is_skipped and world_distance_m <= self.reached_distance_m:
                self.reached_waypoints[index] = True

            if is_skipped:
                cv2.putText(
                    frame,
                    f"{waypoint.label}: skipped",
                    (12, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                projection = project_waypoint(waypoint.position, image, self.fov_degrees)
                if projection is not None:
                    pixel_x, pixel_y, distance_m, _ = projection
                    inside = 0 <= pixel_x < width and 0 <= pixel_y < height
                    label = f"{waypoint.label} skipped"
                    if inside:
                        center = (int(round(pixel_x)), int(round(pixel_y)))
                        cv2.circle(frame, center, 9, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.drawMarker(
                            frame,
                            center,
                            (0, 0, 255),
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=18,
                            thickness=2,
                            line_type=cv2.LINE_AA,
                        )
                        self.draw_text_with_shadow(
                            cv2,
                            frame,
                            label,
                            (center[0] + 12, center[1] - 12),
                            color=(0, 0, 255),
                        )
                    elif self.draw_edge_indicators:
                        edge = self.edge_point(width, height, pixel_x, pixel_y)
                        cv2.circle(frame, edge, 8, (0, 0, 255), 2, cv2.LINE_AA)
                        self.draw_text_with_shadow(
                            cv2,
                            frame,
                            label,
                            (edge[0] + 8, edge[1] - 8),
                            color=(0, 0, 255),
                        )
                status_y += 20
                continue

            if self.reached_waypoints[index]:
                cv2.putText(
                    frame,
                    f"{waypoint.label}: reached",
                    (12, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
                status_y += 20
                continue

            projection = project_waypoint(waypoint.position, image, self.fov_degrees)
            if projection is None:
                cv2.putText(
                    frame,
                    f"{waypoint.label}: behind camera",
                    (12, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                status_y += 20
                continue

            pixel_x, pixel_y, distance_m, point_camera = projection
            inside = 0 <= pixel_x < width and 0 <= pixel_y < height
            color = (0, 255, 255) if index == 0 else (80, 220, 255)
            label = f"{waypoint.label} {distance_m:.1f}m"

            if inside:
                center = (int(round(pixel_x)), int(round(pixel_y)))
                cv2.circle(frame, center, 9, color, 2, cv2.LINE_AA)
                cv2.drawMarker(
                    frame,
                    center,
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=18,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                self.draw_text_with_shadow(cv2, frame, label, (center[0] + 12, center[1] - 12))
            elif self.draw_edge_indicators:
                edge = self.edge_point(width, height, pixel_x, pixel_y)
                cv2.circle(frame, edge, 8, color, 2, cv2.LINE_AA)
                self.draw_text_with_shadow(cv2, frame, label, (edge[0] + 8, edge[1] - 8))

            cv2.putText(
                frame,
                f"{waypoint.label}: cam=({point_camera[0]:.1f}, {point_camera[1]:.1f}, {point_camera[2]:.1f})",
                (12, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            status_y += 20

    def draw_text_with_shadow(
        self,
        cv2,
        frame,
        text: str,
        origin: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 255),
    ):
        x = max(4, min(frame.shape[1] - 160, origin[0]))
        y = max(18, min(frame.shape[0] - 8, origin[1]))
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            1,
            cv2.LINE_AA,
        )

    def edge_point(self, width: int, height: int, pixel_x: float, pixel_y: float):
        margin = 14
        center_x = width * 0.5
        center_y = height * 0.5
        delta_x = pixel_x - center_x
        delta_y = pixel_y - center_y
        if abs(delta_x) < 1e-6 and abs(delta_y) < 1e-6:
            return int(center_x), int(center_y)

        scale_x = (center_x - margin) / abs(delta_x) if delta_x else math.inf
        scale_y = (center_y - margin) / abs(delta_y) if delta_y else math.inf
        scale = min(scale_x, scale_y)
        return (
            int(round(center_x + delta_x * scale)),
            int(round(center_y + delta_y * scale)),
        )


def get_pose_position_ned(drone: Drone) -> List[float]:
    position = drone.get_ground_truth_kinematics()["pose"]["position"]
    return [float(position["x"]), float(position["y"]), float(position["z"])]


def yaw_from_quaternion(rotation) -> float:
    w = float(rotation["w"])
    x = float(rotation["x"])
    y = float(rotation["y"])
    z = float(rotation["z"])
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def get_pose_yaw_ned(drone: Drone) -> float:
    pose = drone.get_ground_truth_kinematics()["pose"]
    rotation = pose.get("orientation") or pose.get("rotation")
    if rotation is None:
        return 0.0
    return yaw_from_quaternion(rotation)


async def request_px4_control(drone: Drone):
    projectairsim_log().info("Requesting PX4 control for direct movement")
    request_control_task = await drone.request_control_async()
    await request_control_task


def closest_route_progress(
    path: Sequence[Sequence[float]],
    position: Sequence[float],
) -> Tuple[float, int]:
    if len(path) < 2:
        return 0.0, 0

    best_distance_sq = math.inf
    best_progress_m = 0.0
    best_segment_index = 0
    traveled_m = 0.0

    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        segment = [end[axis] - start[axis] for axis in range(3)]
        segment_length_sq = sum(component * component for component in segment)
        segment_length_m = math.sqrt(segment_length_sq)

        if segment_length_sq <= 1e-9:
            fraction = 0.0
            closest = [float(start[axis]) for axis in range(3)]
        else:
            offset = [position[axis] - start[axis] for axis in range(3)]
            fraction = clamp(
                sum(offset[axis] * segment[axis] for axis in range(3))
                / segment_length_sq,
                0.0,
                1.0,
            )
            closest = [
                float(start[axis]) + segment[axis] * fraction
                for axis in range(3)
            ]

        distance_sq = sum((position[axis] - closest[axis]) ** 2 for axis in range(3))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_progress_m = traveled_m + segment_length_m * fraction
            best_segment_index = index - 1

        traveled_m += segment_length_m

    return best_progress_m, best_segment_index


def next_route_waypoint_index(
    path: Sequence[Sequence[float]],
    position: Sequence[float],
) -> int:
    if len(path) <= 1:
        return 0

    progress_m, _ = closest_route_progress(path, position)
    route_distances = cumulative_route_distances(path)
    for index in range(1, len(route_distances)):
        if route_distances[index] > progress_m + 1e-3:
            return index
    return len(path) - 1


def segment_lookahead_point(
    start: Sequence[float],
    end: Sequence[float],
    position: Sequence[float],
    lookahead_m: float,
) -> List[float]:
    segment = [float(end[axis]) - float(start[axis]) for axis in range(3)]
    segment_length_sq = sum(component * component for component in segment)
    if segment_length_sq <= 1e-9 or lookahead_m <= 0.0:
        return [float(end[0]), float(end[1]), float(end[2])]

    offset = [float(position[axis]) - float(start[axis]) for axis in range(3)]
    projection_fraction = clamp(
        sum(offset[axis] * segment[axis] for axis in range(3))
        / segment_length_sq,
        0.0,
        1.0,
    )
    segment_length_m = math.sqrt(segment_length_sq)
    lookahead_fraction = lookahead_m / max(segment_length_m, 1e-6)
    return interpolate_point(
        start,
        end,
        min(1.0, projection_fraction + lookahead_fraction),
    )


def make_default_waypoint_ahead(drone: Drone, distance_m: float) -> Waypoint:
    pose = drone.get_ground_truth_kinematics()["pose"]
    position = pose["position"]
    rotation = pose.get("orientation") or pose.get("rotation")
    origin = [float(position["x"]), float(position["y"]), float(position["z"])]
    forward = [distance_m, 0.0, 0.0]
    if rotation is not None:
        forward = rotate_vector_by_quaternion(forward, rotation)
    waypoint = [origin[idx] + forward[idx] for idx in range(3)]
    return Waypoint("WP000", waypoint)


def collect_waypoints(args, drone: Drone) -> List[Waypoint]:
    waypoints = []
    for index, point in enumerate(args.waypoint or []):
        waypoints.append(Waypoint(f"WP{index:03d}", point))

    if not waypoints and args.goal is not None:
        waypoints.append(Waypoint("GOAL", args.goal))

    if not waypoints:
        waypoints.append(make_default_waypoint_ahead(drone, args.demo_forward_distance_m))
        projectairsim_log().info(
            "No --waypoint or --goal supplied; generated %s at NED %s",
            waypoints[0].label,
            format_vector3(waypoints[0].position),
        )

    return waypoints


def waypoints_from_path(
    path: Sequence[Sequence[float]],
    final_label: str = "END",
    skipped_points: Optional[Sequence[Sequence[float]]] = None,
) -> List[Waypoint]:
    final_index = len(path) - 1
    waypoints = []
    for index, point in enumerate(path):
        if index == 0:
            label = "START"
        elif index == final_index:
            label = final_label
        else:
            label = f"WP{index:03d}"
        waypoints.append(
            Waypoint(label, [float(point[0]), float(point[1]), float(point[2])])
        )
    for index, point in enumerate(skipped_points or [], start=1):
        waypoints.append(
            Waypoint(
                f"SKIP{index:03d}",
                [float(point[0]), float(point[1]), float(point[2])],
                "skipped",
            )
        )
    return waypoints


def apply_min_altitude(
    point: Sequence[float],
    min_altitude_m: float,
    ground_z_ned: float,
) -> List[float]:
    adjusted = [float(point[0]), float(point[1]), float(point[2])]
    if min_altitude_m <= 0.0:
        return adjusted

    highest_allowed_down = ground_z_ned - min_altitude_m
    if adjusted[2] > highest_allowed_down:
        adjusted[2] = highest_allowed_down
    return adjusted


def plan_astar_route(world: World, args) -> List[List[float]]:
    if args.start is None:
        raise RuntimeError("--goal route planning requires --start")
    if args.goal is None:
        raise RuntimeError("A* route planning requires --goal")

    start = apply_min_altitude(args.start, args.min_altitude, args.ground_z_ned)
    goal = apply_min_altitude(args.goal, args.min_altitude, args.ground_z_ned)
    if start != args.start or goal != args.goal:
        projectairsim_log().info(
            "Applied --min-altitude %.2fm above ground_z_ned %.2f: "
            "planning start=%s goal=%s",
            args.min_altitude,
            args.ground_z_ned,
            start,
            goal,
        )

    map_center = args.map_center or infer_map_center(start, goal)
    map_size = args.map_size or infer_map_size(
        start,
        goal,
        args.map_margin_m,
        args.min_map_size_m,
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

    projectairsim_log().info("Planning path from %s to %s", start, goal)
    dense_path = planner.generate_plan(start, goal)
    if not dense_path:
        raise RuntimeError("A* did not find a path")

    path = sparsify_path(dense_path, args.waypoint_spacing_m)
    projectairsim_log().info(
        "Planned %d dense points, reduced to %d waypoints, path length %.2f m",
        len(dense_path),
        len(path),
        calculate_path_length(path),
    )
    if args.print_waypoints:
        for index, point in enumerate(path):
            projectairsim_log().info("Waypoint %03d: %s", index, point)
    return path


def build_route_path(world: World, args, drone: Drone) -> List[List[float]]:
    del world

    if args.route:
        path = args.route
    else:
        path = []
        path.append(args.start or get_pose_position_ned(drone))
        path.extend(args.waypoint or [])
        if args.goal is not None:
            path.append(args.goal)

    if len(path) < 2:
        raise RuntimeError(
            "Provide a route with --route, or provide at least one --waypoint/--goal "
            "after the optional --start."
        )

    path = [
        [float(point[0]), float(point[1]), float(point[2])]
        for point in path
    ]
    if args.print_waypoints:
        for index, point in enumerate(path):
            projectairsim_log().info("Route point %03d: %s", index, point)
    return path


def create_static_route_scan(
    world: World,
    args,
    path: Sequence[Sequence[float]],
    log_scan: bool = True,
    margin_override_m: Optional[float] = None,
) -> StaticRouteScan:
    resolution_m = max(0.1, args.object_scan_resolution_m)
    scan_margin_base_m = (
        args.object_scan_margin_m
        if margin_override_m is None
        else margin_override_m
    )
    scan_margin_m = max(
        scan_margin_base_m,
        args.object_path_clearance_m + (2.0 * resolution_m),
    )
    map_center, map_size = route_scan_volume(
        path,
        scan_margin_m,
        args.object_scan_min_size_m,
        resolution_m,
    )
    actors_to_ignore = list(args.ignore_actor or [])
    if args.drone_name not in actors_to_ignore:
        actors_to_ignore.append(args.drone_name)

    if log_scan:
        projectairsim_log().info(
            "Scanning route corridor for static objects: center=%s size=%s "
            "resolution=%.2fm clearance=%.2fm",
            format_vector3(map_center),
            format_vector3(map_size),
            resolution_m,
            args.object_path_clearance_m,
        )
    occupancy_grid = world.create_voxel_grid(
        make_pose_ned(map_center),
        map_size[0],
        map_size[1],
        map_size[2],
        resolution_m,
        actors_to_ignore=actors_to_ignore,
    )
    planner = AStarPlanner(
        occupancy_grid,
        map_center,
        map_size,
        resolution_m,
        ground_z_ned=args.ground_z_ned,
    )
    return StaticRouteScan(planner, map_center, map_size, resolution_m)


def find_static_route_obstacle(
    scan: StaticRouteScan,
    args,
    path: Sequence[Sequence[float]],
    log_clear: bool = True,
) -> Optional[StaticRouteObstacle]:
    if not args.replan_on_object or len(path) < 2:
        return None

    start_ignore_m = max(0.0, args.object_scan_start_ignore_m)
    for distance_m, point, segment_index in iter_route_samples(
        path,
        args.object_scan_sample_spacing_m,
    ):
        if distance_m < start_ignore_m:
            continue
        if is_route_corridor_clear(scan.planner, point, args):
            continue

        stop_distance_m = max(0.0, distance_m - args.object_stop_distance_m)
        stop_point, _ = point_at_route_distance(path, stop_distance_m)
        return StaticRouteObstacle(
            obstacle_point=point,
            stop_point=stop_point,
            route_distance_m=distance_m,
            stop_distance_m=stop_distance_m,
            segment_index=segment_index,
        )

    if log_clear:
        projectairsim_log().info("No static object detected on the supplied route corridor")
    return None


def find_clear_reroute_start(
    scan: StaticRouteScan,
    args,
    path: Sequence[Sequence[float]],
    target_distance_m: float,
) -> Tuple[List[float], float]:
    spacing_m = max(0.1, args.object_scan_sample_spacing_m)
    distance_m = max(0.0, target_distance_m)
    while distance_m >= 0.0:
        point, _ = point_at_route_distance(path, distance_m)
        if is_route_corridor_clear(scan.planner, point, args):
            return point, distance_m
        distance_m -= spacing_m

    start = [float(value) for value in path[0]]
    if is_route_corridor_clear(scan.planner, start, args):
        return start, 0.0
    raise RuntimeError(
        "Could not find a clear route point before the detected object to start "
        "the reroute."
    )


def find_explicit_rejoin_index(
    args,
    path: Sequence[Sequence[float]],
    minimum_index: int,
) -> Optional[int]:
    if args.replan_rejoin_point is None:
        return None

    best_index = None
    best_distance = math.inf
    for index in range(max(0, minimum_index), len(path)):
        distance_m = distance_between(path[index], args.replan_rejoin_point)
        if distance_m < best_distance:
            best_index = index
            best_distance = distance_m

    if best_index is None or best_distance > args.replan_rejoin_tolerance_m:
        raise RuntimeError(
            "--replan-rejoin-point was not found in the remaining route within "
            f"{args.replan_rejoin_tolerance_m:.2f}m"
        )
    return best_index


def find_auto_rejoin_index(
    scan: StaticRouteScan,
    args,
    path: Sequence[Sequence[float]],
    obstacle: StaticRouteObstacle,
) -> int:
    minimum_index = min(
        len(path) - 1,
        obstacle.segment_index + max(1, args.replan_rejoin_waypoints_ahead) + 1,
    )
    explicit_index = find_explicit_rejoin_index(
        args,
        path,
        obstacle.segment_index + 1,
    )
    if explicit_index is not None:
        return explicit_index

    if is_route_corridor_clear(scan.planner, path[minimum_index], args):
        return minimum_index

    minimum_distance_m = (
        obstacle.route_distance_m
        + max(0.0, args.replan_rejoin_after_obstacle_m)
    )
    clear_started_at = None
    for distance_m, point, _ in iter_route_samples(
        path,
        args.object_scan_sample_spacing_m,
    ):
        if distance_m <= obstacle.route_distance_m:
            continue

        if is_route_corridor_clear(scan.planner, point, args):
            if clear_started_at is None:
                clear_started_at = distance_m
            has_clear_run = (
                distance_m - clear_started_at >= args.replan_rejoin_clear_distance_m
            )
            if distance_m >= minimum_distance_m and has_clear_run:
                candidate_index = route_waypoint_index_at_distance(
                    path,
                    distance_m,
                    minimum_index,
                )
                while candidate_index < len(path):
                    if is_route_corridor_clear(
                        scan.planner,
                        path[candidate_index],
                        args,
                    ):
                        return candidate_index
                    candidate_index += 1
                break
        else:
            clear_started_at = None

    for index in range(minimum_index, len(path)):
        if is_route_corridor_clear(scan.planner, path[index], args):
            return index

    raise RuntimeError("Could not find a clear downstream route waypoint to rejoin.")


def plan_astar_segment(
    planner: AStarPlanner,
    start: Sequence[float],
    goal: Sequence[float],
    label: str,
) -> List[List[float]]:
    validate_grid_coordinate(planner, start, f"{label} start")
    validate_grid_coordinate(planner, goal, f"{label} goal")
    dense_path = planner.generate_plan(start, goal)
    if not dense_path:
        raise RuntimeError(f"A* did not find a path for {label}")
    return [
        [float(point[0]), float(point[1]), float(point[2])]
        for point in dense_path
    ]


def plan_bypass_path(
    scan: StaticRouteScan,
    args,
    reroute_start: Sequence[float],
    rejoin_point: Sequence[float],
) -> List[List[float]]:
    targets = []
    if args.replan_emergency_node is not None:
        targets.append([float(value) for value in args.replan_emergency_node])
    targets.append([float(value) for value in rejoin_point])

    bypass = []
    start = [float(value) for value in reroute_start]
    for index, target in enumerate(targets, start=1):
        label = (
            "emergency-node bypass"
            if index == 1 and args.replan_emergency_node is not None
            else "route-rejoin bypass"
        )
        projectairsim_log().info(
            "Planning A* %s from %s to %s",
            label,
            format_vector3(start),
            format_vector3(target),
        )
        segment = plan_astar_segment(scan.planner, start, target, label)
        for point in segment:
            append_unique_point(bypass, point)
        start = target

    return sparsify_path(bypass, args.replan_waypoint_spacing_m)


def build_rerouted_path(
    scan: StaticRouteScan,
    args,
    path: Sequence[Sequence[float]],
    obstacle: StaticRouteObstacle,
) -> List[List[float]]:
    reroute_start, reroute_start_distance = find_clear_reroute_start(
        scan,
        args,
        path,
        obstacle.stop_distance_m,
    )
    if distance_between(reroute_start, obstacle.stop_point) > 1e-3:
        projectairsim_log().warning(
            "Requested %.2fm stop-before-object point was not clear; reroute "
            "will start at NED %s instead.",
            args.object_stop_distance_m,
            format_vector3(reroute_start),
        )
        obstacle.stop_point = reroute_start
        obstacle.stop_distance_m = reroute_start_distance

    rejoin_index = find_auto_rejoin_index(scan, args, path, obstacle)
    rejoin_point = [float(value) for value in path[rejoin_index]]
    obstacle.rejoin_index = rejoin_index
    obstacle.rejoin_point = rejoin_point
    obstacle.skipped_points = [
        [float(point[0]), float(point[1]), float(point[2])]
        for point in path[1:rejoin_index]
    ]

    if args.replan_emergency_node is not None:
        projectairsim_log().info(
            "Reroute will pass through emergency node %s before rejoining route "
            "waypoint %03d %s.",
            format_vector3(args.replan_emergency_node),
            rejoin_index,
            format_vector3(rejoin_point),
        )
    else:
        projectairsim_log().info(
            "Planning A* reroute from %s to original route waypoint %03d %s",
            format_vector3(reroute_start),
            rejoin_index,
            format_vector3(rejoin_point),
        )

    bypass = plan_bypass_path(scan, args, reroute_start, rejoin_point)

    mission_path = []
    for point in truncate_route_at_distance(path, reroute_start_distance):
        append_unique_point(mission_path, point)
    for point in bypass:
        append_unique_point(mission_path, point)
    for point in path[rejoin_index + 1 :]:
        append_unique_point(mission_path, point)

    projectairsim_log().warning(
        "Object on path detected at NED %s. Replaced blocked route segment "
        "with %d A* bypass waypoint(s)%s, then rejoined the normal route at "
        "waypoint %03d %s.",
        format_vector3(obstacle.obstacle_point),
        len(bypass),
        (
            f" via emergency node {format_vector3(args.replan_emergency_node)}"
            if args.replan_emergency_node is not None
            else ""
        ),
        rejoin_index,
        format_vector3(rejoin_point),
    )
    if obstacle.skipped_points:
        for index, point in enumerate(obstacle.skipped_points, start=1):
            projectairsim_log().warning(
                "Skipped waypoint %03d because of obstacle: %s",
                index,
                format_vector3(point),
            )
    return mission_path


def remaining_route_from_current(
    active_path: Sequence[Sequence[float]],
    waypoint_index: int,
    current_position: Sequence[float],
) -> List[List[float]]:
    remaining_path = []
    append_unique_point(remaining_path, current_position)
    for point in active_path[waypoint_index:]:
        append_unique_point(remaining_path, point)
    return remaining_path


def lookahead_route_from_current(
    active_path: Sequence[Sequence[float]],
    waypoint_index: int,
    current_position: Sequence[float],
    lookahead_waypoints: int,
) -> List[List[float]]:
    lookahead_path = []
    append_unique_point(lookahead_path, current_position)
    end_index = min(
        len(active_path),
        waypoint_index + max(1, lookahead_waypoints),
    )
    for point in active_path[waypoint_index:end_index]:
        append_unique_point(lookahead_path, point)
    return lookahead_path


def refresh_dynamic_route_visuals(
    world: World,
    display: Optional[FpvWaypointOverlayDisplay],
    active_path: Sequence[Sequence[float]],
    args,
    skipped_points: Optional[Sequence[Sequence[float]]] = None,
) -> None:
    waypoints = waypoints_from_path(active_path, skipped_points=skipped_points)
    if display:
        display.set_waypoints(waypoints)
    plot_world_waypoint_markers(world, waypoints, args)


def scan_active_route_for_obstacle(
    world: World,
    args,
    active_path: Sequence[Sequence[float]],
    waypoint_index: int,
    current_position: Sequence[float],
) -> Optional[StaticRouteObstacle]:
    lookahead_path = lookahead_route_from_current(
        active_path,
        waypoint_index,
        current_position,
        args.dynamic_replan_lookahead_waypoints,
    )
    if len(lookahead_path) < 2:
        return None

    lookahead_scan = create_static_route_scan(
        world,
        args,
        lookahead_path,
        log_scan=False,
        margin_override_m=args.dynamic_replan_detection_margin_m,
    )
    obstacle = find_static_route_obstacle(
        lookahead_scan,
        args,
        lookahead_path,
        log_clear=False,
    )
    return obstacle


def build_dynamic_rerouted_path(
    world: World,
    args,
    active_path: Sequence[Sequence[float]],
    waypoint_index: int,
    current_position: Sequence[float],
    obstacle: StaticRouteObstacle,
) -> List[List[float]]:
    remaining_path = remaining_route_from_current(
        active_path,
        waypoint_index,
        current_position,
    )
    obstacle.stop_point = [float(value) for value in current_position]
    obstacle.stop_distance_m = 0.0
    replan_scan_path = list(remaining_path)
    if args.replan_emergency_node is not None:
        append_unique_point(replan_scan_path, args.replan_emergency_node)
    route_scan = create_static_route_scan(world, args, replan_scan_path)
    rerouted_path = build_rerouted_path(route_scan, args, remaining_path, obstacle)
    if args.dynamic_replan_max_segment_m > 0.0:
        rerouted_path = densify_path(rerouted_path, args.dynamic_replan_max_segment_m)
    return rerouted_path


async def maybe_replan_active_route(
    world: World,
    drone: Drone,
    args,
    active_path: Sequence[Sequence[float]],
    waypoint_index: int,
    current_position: Sequence[float],
    commanded_velocity: Sequence[float],
) -> Tuple[List[List[float]], int, Optional[StaticRouteObstacle], List[float]]:
    obstacle = scan_active_route_for_obstacle(
        world,
        args,
        active_path,
        waypoint_index,
        current_position,
    )
    if obstacle is None:
        return list(active_path), waypoint_index, None, list(commanded_velocity)

    projectairsim_log().warning(
        "Object on path detected on the next route leg at NED %s. Stopping "
        "before live replanning.",
        format_vector3(obstacle.obstacle_point),
    )
    command_duration_sec = max(0.05, args.velocity_command_duration_sec)
    max_velocity_delta = max(0.0, args.acceleration_limit_mps2) * command_duration_sec
    commanded_velocity = await brake_to_stop_by_velocity(
        drone,
        commanded_velocity,
        command_duration_sec,
        max_velocity_delta,
    )
    if args.dynamic_replan_stop_hold_sec > 0.0:
        await hold_waypoint_by_velocity(
            drone,
            args.dynamic_replan_stop_hold_sec,
            command_duration_sec,
            "Dynamic replan stop",
            None,
            None,
        )

    current_position = get_pose_position_ned(drone)
    rerouted_path = build_dynamic_rerouted_path(
        world,
        args,
        active_path,
        waypoint_index,
        current_position,
        obstacle,
    )
    return rerouted_path, 1, obstacle, commanded_velocity


async def fly_one_waypoint_smooth(
    world: World,
    drone: Drone,
    args,
    active_path: Sequence[Sequence[float]],
    waypoint_index: int,
    commanded_velocity: Sequence[float],
    replan_count: int,
) -> Tuple[List[List[float]], int, List[float], Optional[StaticRouteObstacle]]:
    target = active_path[waypoint_index]
    previous_target = active_path[max(0, waypoint_index - 1)]
    is_final_waypoint = waypoint_index == len(active_path) - 1
    command_duration_sec = max(0.05, args.velocity_command_duration_sec)
    max_velocity_delta = max(0.0, args.acceleration_limit_mps2) * command_duration_sec
    max_yaw_rate_radps = math.radians(max(0.0, args.path_yaw_rate_dps))
    slowdown_distance_m = max(args.waypoint_acceptance_m, args.slowdown_distance_m)
    velocity_lookahead_m = max(0.0, args.velocity_lookahead_m)
    started_at = time.time()
    last_report_at = 0.0
    velocity = [
        float(commanded_velocity[0]),
        float(commanded_velocity[1]),
        float(commanded_velocity[2]),
    ]

    while True:
        current = get_pose_position_ned(drone)
        distance = distance_between(current, target)
        if distance <= args.waypoint_acceptance_m:
            if is_final_waypoint:
                velocity = await brake_to_stop_by_velocity(
                    drone,
                    velocity,
                    command_duration_sec,
                    max_velocity_delta,
                )
            projectairsim_log().info(
                "Waypoint %03d reached target %s; pose NED %s; error %.2f m",
                waypoint_index,
                format_vector3(target),
                format_vector3(current),
                distance,
            )
            return list(active_path), waypoint_index + 1, velocity, None

        elapsed = time.time() - started_at
        if args.move_timeout_sec > 0 and elapsed > args.move_timeout_sec:
            drone.cancel_last_task()
            raise RuntimeError(
                f"Waypoint {waypoint_index:03d} timed out after "
                f"{args.move_timeout_sec:.1f}s; target {format_vector3(target)}, "
                f"pose NED {format_vector3(current)}, remaining {distance:.2f} m"
            )

        if (
            args.replan_on_object
            and replan_count < args.dynamic_replan_max_count
            and elapsed >= args.dynamic_replan_min_flight_sec
        ):
            new_path, new_index, obstacle, velocity = await maybe_replan_active_route(
                world,
                drone,
                args,
                active_path,
                waypoint_index,
                current,
                velocity,
            )
            if obstacle is not None:
                return new_path, new_index, velocity, obstacle

        if elapsed - last_report_at >= args.report_every_sec:
            projectairsim_log().info(
                "Waypoint %03d moving toward %s; pose NED %s; remaining %.2f m",
                waypoint_index,
                format_vector3(target),
                format_vector3(current),
                distance,
            )
            last_report_at = elapsed

        steering_target = segment_lookahead_point(
            previous_target,
            target,
            current,
            velocity_lookahead_m,
        )
        if is_final_waypoint and distance <= slowdown_distance_m:
            steering_target = target
        steering_distance = distance_between(current, steering_target)
        if steering_distance <= 1e-6:
            steering_target = target
            steering_distance = distance

        delta = [steering_target[idx] - current[idx] for idx in range(3)]
        speed_scale = 1.0
        if is_final_waypoint:
            speed_scale = min(1.0, max(0.2, distance / slowdown_distance_m))
        desired_speed = min(args.velocity_mps, steering_distance / command_duration_sec)
        desired_speed *= speed_scale
        desired_velocity = [
            (component / steering_distance) * desired_speed
            for component in delta
        ]
        velocity = limit_vector_delta(velocity, desired_velocity, max_velocity_delta)

        yaw_rate_radps = 0.0
        horizontal_speed = math.hypot(velocity[0], velocity[1])
        if args.face_travel_direction and horizontal_speed > 0.05 and max_yaw_rate_radps > 0.0:
            desired_yaw = math.atan2(velocity[1], velocity[0])
            yaw_error = wrap_angle_rad(desired_yaw - get_pose_yaw_ned(drone))
            yaw_rate_radps = clamp(
                yaw_error / command_duration_sec,
                -max_yaw_rate_radps,
                max_yaw_rate_radps,
            )

        await drone.move_by_velocity_async(
            v_north=velocity[0],
            v_east=velocity[1],
            v_down=velocity[2],
            duration=command_duration_sec,
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
            yaw_is_rate=True,
            yaw=yaw_rate_radps,
        )
        await asyncio.sleep(command_duration_sec)


async def fly_path_with_dynamic_replanning_velocity(
    world: World,
    drone: Drone,
    path: List[List[float]],
    args,
    display: Optional[FpvWaypointOverlayDisplay] = None,
) -> Tuple[List[List[float]], int]:
    active_path = (
        densify_path(path, args.dynamic_replan_max_segment_m)
        if args.dynamic_replan_max_segment_m > 0.0
        else [[float(point[0]), float(point[1]), float(point[2])] for point in path]
    )
    waypoint_index = 1
    commanded_velocity = [0.0, 0.0, 0.0]
    replan_count = 0
    skipped_display_points = []

    while waypoint_index < len(active_path):
        active_path, waypoint_index, commanded_velocity, obstacle = (
            await fly_one_waypoint_smooth(
                world,
                drone,
                args,
                active_path,
                waypoint_index,
                commanded_velocity,
                replan_count,
            )
        )
        if obstacle is not None:
            replan_count += 1
            for point in obstacle.skipped_points or []:
                append_unique_point(skipped_display_points, point)
            refresh_dynamic_route_visuals(
                world,
                display,
                active_path,
                args,
                skipped_display_points,
            )
            projectairsim_log().warning(
                "Dynamic reroute %d/%d built while flying. Continuing via "
                "%d active waypoint(s).",
                replan_count,
                args.dynamic_replan_max_count,
                len(active_path),
            )
            continue

        if args.waypoint_hold_sec > 0.0 and waypoint_index < len(active_path):
            await hold_waypoint_by_velocity(
                drone,
                args.waypoint_hold_sec,
                args.velocity_command_duration_sec,
                f"Waypoint {waypoint_index - 1:03d}",
                None,
                None,
            )

    return active_path, replan_count


def infer_path_timeout_sec(path: Sequence[Sequence[float]], args) -> float:
    if args.path_timeout_sec > 0:
        return args.path_timeout_sec
    return max(
        args.move_timeout_sec,
        calculate_path_length(path) / max(args.velocity_mps, 0.1) * 3.0,
    )


async def start_move_on_path_task(
    drone: Drone,
    path: Sequence[Sequence[float]],
    args,
    label: str,
) -> Tuple[asyncio.Task, float]:
    await request_px4_control(drone)
    path_to_follow = [
        [float(point[0]), float(point[1]), float(point[2])]
        for point in path
    ]
    path_timeout_sec = infer_path_timeout_sec(path_to_follow, args)
    projectairsim_log().info(
        "%s using MoveOnPath path-api: %d waypoint(s), %.2fm length, "
        "velocity %.2fm/s, lookahead %.2f, adaptive lookahead %.2f",
        label,
        len(path_to_follow),
        calculate_path_length(path_to_follow),
        args.velocity_mps,
        args.lookahead_m,
        args.adaptive_lookahead,
    )
    path_task = await drone.move_on_path_async(
        path_to_follow,
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
    return path_task, path_timeout_sec


async def finish_cancelled_path_task(task: asyncio.Task, label: str) -> None:
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
    except asyncio.TimeoutError:
        projectairsim_log().info("%s cancel is still settling; continuing", label)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        projectairsim_log().info("%s ended during cancel: %s", label, exc)


async def fly_path_by_path_api(
    drone: Drone,
    path: Sequence[Sequence[float]],
    args,
    label: str = "Follow waypoint path",
) -> None:
    path_task, path_timeout_sec = await start_move_on_path_task(
        drone,
        path,
        args,
        label,
    )
    await await_drone_task(
        drone,
        path_task,
        label,
        path_timeout_sec + 5.0,
        args.report_every_sec,
    )


async def fly_path_with_dynamic_replanning_path_api(
    world: World,
    drone: Drone,
    path: List[List[float]],
    args,
    display: Optional[FpvWaypointOverlayDisplay] = None,
) -> Tuple[List[List[float]], int]:
    active_path = (
        densify_path(path, args.dynamic_replan_max_segment_m)
        if args.dynamic_replan_max_segment_m > 0.0
        else [[float(point[0]), float(point[1]), float(point[2])] for point in path]
    )
    replan_count = 0
    skipped_display_points = []
    check_interval_sec = max(0.1, args.dynamic_replan_check_sec)

    while True:
        label = (
            "Follow dynamic route"
            if replan_count == 0
            else f"Follow rerouted path {replan_count}"
        )
        path_task, path_timeout_sec = await start_move_on_path_task(
            drone,
            active_path,
            args,
            label,
        )
        started_at = time.time()
        last_report_at = started_at
        last_check_at = 0.0
        obstacle = None

        while not path_task.done():
            now = time.time()
            elapsed = now - started_at
            current = get_pose_position_ned(drone)

            if path_timeout_sec > 0 and elapsed > path_timeout_sec:
                drone.cancel_last_task()
                raise RuntimeError(
                    f"{label} timed out after {path_timeout_sec:.1f}s at pose "
                    f"{format_vector3(current)}"
                )

            can_replan = (
                args.replan_on_object
                and replan_count < args.dynamic_replan_max_count
                and elapsed >= args.dynamic_replan_min_flight_sec
            )
            if can_replan and now - last_check_at >= check_interval_sec:
                waypoint_index = next_route_waypoint_index(active_path, current)
                obstacle = scan_active_route_for_obstacle(
                    world,
                    args,
                    active_path,
                    waypoint_index,
                    current,
                )
                last_check_at = now
                if obstacle is not None:
                    projectairsim_log().warning(
                        "Object on path detected on the next route leg at NED %s. "
                        "Stopping path-api flight before live replanning.",
                        format_vector3(obstacle.obstacle_point),
                    )
                    drone.cancel_last_task()
                    await finish_cancelled_path_task(path_task, label)
                    command_duration_sec = max(0.05, args.velocity_command_duration_sec)
                    hold_sec = max(
                        args.dynamic_replan_stop_hold_sec,
                        command_duration_sec,
                    )
                    await hold_waypoint_by_velocity(
                        drone,
                        hold_sec,
                        command_duration_sec,
                        "Dynamic replan stop",
                        None,
                        None,
                    )

                    current = get_pose_position_ned(drone)
                    active_path = build_dynamic_rerouted_path(
                        world,
                        args,
                        active_path,
                        waypoint_index,
                        current,
                        obstacle,
                    )
                    replan_count += 1
                    for point in obstacle.skipped_points or []:
                        append_unique_point(skipped_display_points, point)
                    refresh_dynamic_route_visuals(
                        world,
                        display,
                        active_path,
                        args,
                        skipped_display_points,
                    )
                    projectairsim_log().warning(
                        "Dynamic reroute %d/%d built while flying. Continuing "
                        "with MoveOnPath via %d active waypoint(s).",
                        replan_count,
                        args.dynamic_replan_max_count,
                        len(active_path),
                    )
                    break

            if now - last_report_at >= args.report_every_sec:
                waypoint_index = next_route_waypoint_index(active_path, current)
                projectairsim_log().info(
                    "%s running after %.1fs; pose NED %s; next waypoint %03d",
                    label,
                    elapsed,
                    format_vector3(current),
                    waypoint_index,
                )
                last_report_at = now

            await asyncio.sleep(min(0.25, check_interval_sec))

        if obstacle is not None:
            continue

        result = await path_task
        current = get_pose_position_ned(drone)
        projectairsim_log().info(
            "%s completed with result=%s; pose NED %s",
            label,
            result,
            format_vector3(current),
        )
        if result is False:
            raise RuntimeError(f"{label} returned False")
        return active_path, replan_count


async def fly_path_with_dynamic_replanning(
    world: World,
    drone: Drone,
    path: List[List[float]],
    args,
    display: Optional[FpvWaypointOverlayDisplay] = None,
) -> Tuple[List[List[float]], int]:
    if args.flight_driver == "velocity":
        return await fly_path_with_dynamic_replanning_velocity(
            world,
            drone,
            path,
            args,
            display,
        )
    return await fly_path_with_dynamic_replanning_path_api(
        world,
        drone,
        path,
        args,
        display,
    )


def plot_world_waypoint_markers(world: World, waypoints: Sequence[Waypoint], args):
    if args.no_world_markers:
        return

    active_waypoints = [
        waypoint for waypoint in waypoints if waypoint.status != "skipped"
    ]
    skipped_waypoints = [
        waypoint for waypoint in waypoints if waypoint.status == "skipped"
    ]

    if args.flush_markers:
        world.flush_persistent_markers()

    def label_positions_for(group: Sequence[Waypoint]) -> List[List[float]]:
        return [
            [
                waypoint.position[0],
                waypoint.position[1],
                waypoint.position[2] - args.world_label_z_offset_m,
            ]
            for waypoint in group
        ]

    if active_waypoints:
        world.plot_debug_points(
            [waypoint.position for waypoint in active_waypoints],
            [1.0, 1.0, 0.0, 1.0],
            args.world_marker_size,
            args.world_marker_duration_sec,
            args.persistent_world_markers,
        )
        world.plot_debug_strings(
            [waypoint.label for waypoint in active_waypoints],
            label_positions_for(active_waypoints),
            args.world_label_scale,
            [1.0, 1.0, 1.0, 1.0],
            args.world_marker_duration_sec,
        )

    if skipped_waypoints:
        world.plot_debug_points(
            [waypoint.position for waypoint in skipped_waypoints],
            [1.0, 0.0, 0.0, 1.0],
            args.world_marker_size * 1.25,
            args.world_marker_duration_sec,
            args.persistent_world_markers,
        )
        world.plot_debug_strings(
            [f"{waypoint.label} skipped" for waypoint in skipped_waypoints],
            label_positions_for(skipped_waypoints),
            args.world_label_scale,
            [1.0, 0.0, 0.0, 1.0],
            args.world_marker_duration_sec,
        )

    projectairsim_log().info(
        "Plotted %d waypoint marker(s) in the scene (%d skipped)",
        len(waypoints),
        len(skipped_waypoints),
    )


async def wait_for_px4_ready(drone: Drone, timeout_sec: float):
    timeout_at = time.time() + timeout_sec
    last_message = ""
    last_ready_val = False
    last_can_arm = False
    last_ready_message = ""
    last_log_time = 0.0

    while time.time() < timeout_at:
        state = drone.get_ready_state()
        last_ready_val = bool(state.get("ready_val"))
        last_ready_message = state.get("ready_message") or "Waiting for PX4 controller"
        last_can_arm = drone.can_arm() if last_ready_val else False

        if last_ready_val and last_can_arm:
            projectairsim_log().info("PX4 controller is connected and can arm")
            return

        message = last_ready_message
        if last_ready_val:
            message = (
                "Waiting for PX4 MAVLink/GPS readiness. If PX4 says it is "
                "waiting for simulator TCP port 4560, restart PX4 after this "
                "scene is loaded or check PX4_SIM_HOST_ADDR/local-host-ip."
            )
        now = time.time()
        if message != last_message or now - last_log_time >= 15.0:
            remaining_sec = max(0.0, timeout_at - now)
            projectairsim_log().info("%s (%.0fs remaining)", message, remaining_sec)
            last_message = message
            last_log_time = now
        await asyncio.sleep(1.0)

    raise RuntimeError(
        f"PX4 did not become armable after {timeout_sec:.1f}s. "
        f"Last ready state: ready_val={last_ready_val}, "
        f"can_arm={last_can_arm}, ready_message='{last_ready_message}'. "
        "This happens before takeoff. Make sure the Unreal scene is already "
        "running, then start or restart PX4 with `make px4_sitl_default "
        "none_iris` so it can connect to Project AirSim on TCP port 4560. "
        "If PX4 runs on another host, verify PX4_SIM_HOST_ADDR/local-host-ip "
        "and wait for GPS/home readiness."
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

        projectairsim_log().info("Invoking drone.arm() attempt %d", attempt)
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
):
    started_at = time.time()
    last_report_at = started_at

    while not task.done():
        now = time.time()
        elapsed = now - started_at
        if timeout_sec > 0 and elapsed > timeout_sec:
            drone.cancel_last_task()
            position = get_pose_position_ned(drone)
            raise RuntimeError(
                f"{label} timed out after {timeout_sec:.1f}s at pose "
                f"{format_vector3(position)}"
            )

        if now - last_report_at >= report_interval_sec:
            position = get_pose_position_ned(drone)
            projectairsim_log().info(
                "%s still running after %.1fs; pose NED %s",
                label,
                elapsed,
                format_vector3(position),
            )
            last_report_at = now

        await asyncio.sleep(0.25)

    result = await task
    position = get_pose_position_ned(drone)
    projectairsim_log().info(
        "%s completed with result=%s; pose NED %s",
        label,
        result,
        format_vector3(position),
    )
    if result is False:
        raise RuntimeError(f"{label} returned False")
    return result


async def run_overlay(args):
    camera_sensor_id = camera_arg_to_sensor_id(args.camera)
    if args.route and args.start is None:
        args.start = args.route[0]
    has_explicit_flight_target = bool(
        args.route or args.goal is not None or args.waypoint
    )
    if args.start_as_scene_origin and args.flight_driver == "path-api":
        projectairsim_log().warning(
            "--flight-driver path-api can use a different command frame than "
            "the scene-NED debug markers when --start-as-scene-origin is used. "
            "Use --flight-driver velocity for this demo route."
        )
    route_path = None
    mission_path = None
    temp_config_dir = None
    client = ProjectAirSimClient(
        address=args.server_ip,
        port_topics=args.topics_port,
        port_services=args.services_port,
    )
    display = None
    drone = None
    cleanup_needed = False

    try:
        video_output_path = prepare_video_output_path(args.video_path)
        temp_config_dir, effective_scene, effective_sim_config_path = (
            make_runtime_scene_config(args, camera_sensor_id)
        )
        args.effective_scene = effective_scene
        args.effective_sim_config_path = effective_sim_config_path

        projectairsim_log().info("Connecting to Project AirSim")
        client.connect()

        world = World(
            client,
            effective_scene,
            delay_after_load_sec=args.load_delay_sec,
            sim_config_path=effective_sim_config_path,
        )
        drone = Drone(client, world, args.drone_name)

        if args.teleport_start:
            if args.start is None:
                raise RuntimeError("--teleport-start requires --start")
            projectairsim_log().info("Teleporting drone to %s", format_vector3(args.start))
            drone.set_pose(make_pose_ned(args.start), reset_kinematics=True)
            await asyncio.sleep(args.after_teleport_delay_sec)

        if args.camera_angle_deg is not None:
            pose = make_camera_angle_pose(args, camera_sensor_id, args.camera_angle_deg)
            if not drone.set_camera_pose(camera_sensor_id, pose):
                raise RuntimeError(
                    f"Failed to set {camera_sensor_id} angle to "
                    f"{args.camera_angle_deg:g} deg"
                )
            projectairsim_log().info(
                "Set %s camera angle to %.2f deg down from straight ahead",
                camera_sensor_id,
                args.camera_angle_deg,
            )

        if has_explicit_flight_target:
            route_path = build_route_path(world, args, drone)
            mission_path = route_path
            waypoints = waypoints_from_path(
                mission_path,
                final_label="END",
            )
        else:
            waypoints = collect_waypoints(args, drone)

        for waypoint in waypoints:
            projectairsim_log().info(
                "%s NED %s",
                waypoint.label,
                format_vector3(waypoint.position),
            )

        plot_world_waypoint_markers(world, waypoints, args)

        camera_topic = require_scene_camera_topic(drone, camera_sensor_id)
        fov_degrees = read_camera_fov_degrees(
            effective_scene,
            effective_sim_config_path,
            args.drone_name,
            camera_sensor_id,
            args.camera_fov_degrees,
        )
        window_name = f"FPV {args.camera} ({camera_sensor_id})"
        display = FpvWaypointOverlayDisplay(
            window_name,
            waypoints,
            fov_degrees,
            args.preview_width,
            args.preview_height,
            not args.no_edge_indicators,
            args.waypoint_acceptance_m,
            args.max_fps,
            video_output_path,
        )
        display.start()
        client.subscribe(camera_topic, lambda _, image: display.receive(image))
        projectairsim_log().info("Subscribed FPV camera topic: %s", camera_topic)
        projectairsim_log().info("Press Esc in the OpenCV window or Ctrl+C to stop")

        if not args.skip_takeoff:
            await wait_for_px4_ready(drone, args.px4_ready_timeout_sec)
            before_arming_pose = get_pose_position_ned(drone)
            projectairsim_log().info(
                "Before arming pose NED %s",
                format_vector3(before_arming_pose),
            )

            if not drone.enable_api_control():
                raise RuntimeError("Project AirSim did not enable API control")
            cleanup_needed = True
            await arm_with_retry(drone, args.arm_timeout_sec)

            projectairsim_log().info("Taking off")
            takeoff_task = await drone.takeoff_async(timeout_sec=args.takeoff_timeout_sec)
            await await_drone_task(
                drone,
                takeoff_task,
                "Takeoff",
                args.takeoff_timeout_sec + 5.0,
                args.report_every_sec,
            )
            if has_explicit_flight_target and not args.no_fly_to_waypoint:
                if args.flight_driver == "path-api":
                    await request_px4_control(drone)
                    start_task = await drone.move_to_position_async(
                        north=mission_path[0][0],
                        east=mission_path[0][1],
                        down=mission_path[0][2],
                        velocity=args.velocity_mps,
                        timeout_sec=args.move_timeout_sec,
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
                        start_task,
                        "Move to start",
                        args.move_timeout_sec + 5.0,
                        args.report_every_sec,
                    )
                else:
                    await fly_to_point_by_velocity(
                        drone,
                        mission_path[0],
                        args.velocity_mps,
                        args.waypoint_acceptance_m,
                        args.move_timeout_sec,
                        args.report_every_sec,
                        args.face_travel_direction,
                        "Move to start",
                        args.velocity_command_duration_sec,
                        args.acceleration_limit_mps2,
                        args.slowdown_distance_m,
                        args.path_yaw_rate_dps,
                        None,
                        True,
                        True,
                        None,
                        None,
                    )
                if args.replan_on_object:
                    projectairsim_log().info(
                        "Dynamic route replanning is enabled. The drone will "
                        "scan ahead while flying with %s and splice in an A* "
                        "bypass if the route is blocked.",
                        args.flight_driver,
                    )
                    mission_path, replan_count = await fly_path_with_dynamic_replanning(
                        world,
                        drone,
                        mission_path,
                        args,
                        display,
                    )
                    if replan_count:
                        projectairsim_log().info(
                            "Completed mission with %d dynamic reroute(s).",
                            replan_count,
                        )
                else:
                    if args.flight_driver == "path-api":
                        await fly_path_by_path_api(drone, mission_path, args)
                    else:
                        await fly_path_by_velocity(
                            drone,
                            mission_path,
                            args.velocity_mps,
                            args.waypoint_acceptance_m,
                            args.move_timeout_sec,
                            args.report_every_sec,
                            args.face_travel_direction,
                            args.velocity_command_duration_sec,
                            args.acceleration_limit_mps2,
                            args.slowdown_distance_m,
                            args.waypoint_hold_sec,
                            args.path_yaw_rate_dps,
                            None,
                            None,
                        )
                projectairsim_log().info(
                    "Reached route destination %s",
                    mission_path[-1],
                )
                if args.land_at_goal:
                    projectairsim_log().info("Landing at destination")
                    land_task = await drone.land_async(timeout_sec=args.land_timeout_sec)
                    await await_drone_task(
                        drone,
                        land_task,
                        "Landing",
                        args.land_timeout_sec + 5.0,
                        args.report_every_sec,
                    )
                if not args.keep_overlay_after_mission:
                    return
            elif not has_explicit_flight_target:
                projectairsim_log().info(
                    "No explicit --route/--goal/--waypoint supplied; holding after "
                    "takeoff"
                )
        elif args.wait_for_px4_ready:
            await wait_for_px4_ready(drone, args.px4_ready_timeout_sec)

        started_at = time.time()
        last_report_at = 0.0
        while args.duration_sec <= 0 or time.time() - started_at < args.duration_sec:
            elapsed = time.time() - started_at
            if (
                display.frame_count == 0
                and args.first_frame_timeout_sec > 0
                and elapsed > args.first_frame_timeout_sec
            ):
                raise RuntimeError(
                    f"No frames received from {camera_sensor_id} after "
                    f"{args.first_frame_timeout_sec:.1f}s. Check that the sim is "
                    "running and the selected camera has scene capture enabled."
                )

            if elapsed - last_report_at >= args.report_every_sec:
                actual = get_pose_position_ned(drone)
                projectairsim_log().info(
                    "FPV overlay running; pose NED %s; frames=%d",
                    format_vector3(actual),
                    display.frame_count,
                )
                last_report_at = elapsed

            if not display.running:
                if display.error:
                    raise RuntimeError("FPV display thread failed") from display.error
                break
            await asyncio.sleep(0.2)

        if display.frame_count == 0:
            raise RuntimeError(
                f"No frames received from {camera_sensor_id}. Check that the "
                "sim is running and the selected camera has scene capture enabled."
            )

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
        if display:
            display.stop()
        if client.state:
            client.disconnect()
        if temp_config_dir:
            temp_config_dir.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show a drone FPV RGB camera with visible waypoint overlays."
    )
    parser.add_argument(
        "--camera",
        default="front_rgb",
        help=(
            "FPV RGB camera to display. Friendly names include front_rgb, "
            "down_rgb, and chase; direct sensor IDs are also accepted."
        ),
    )
    parser.add_argument("--scene", default="scene_px4_sitl.jsonc")
    parser.add_argument(
        "--sim-config-path",
        default="../example_user_scripts/sim_config/",
    )
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--topics-port", type=int, default=8989)
    parser.add_argument("--services-port", type=int, default=8990)
    parser.add_argument("--load-delay-sec", type=float, default=2.0)
    parser.add_argument("--start", type=parse_vector3, help="NED x,y,z")
    parser.add_argument(
        "--goal",
        type=parse_vector3,
        help="Final route point as NED x,y,z.",
    )
    parser.add_argument(
        "--waypoint",
        action="append",
        type=parse_vector3,
        default=None,
        help="Intermediate route point as NED x,y,z. Repeat for multiple waypoints.",
    )
    parser.add_argument(
        "--route",
        type=parse_route,
        default=None,
        help=(
            "Full route as semicolon-separated NED points, for example "
            "\"0,0,-5; 20,0,-5; 20,15,-5\". Overrides --start/--waypoint/--goal."
        ),
    )
    parser.add_argument(
        "--start-as-scene-origin",
        action="store_true",
        help=(
            "Load a runtime scene config with the drone actor origin set to --start, "
            "matching the preferred PX4 workflow in px4_astar_autopilot.py."
        ),
    )
    parser.add_argument("--teleport-start", action="store_true")
    parser.add_argument("--after-teleport-delay-sec", type=float, default=2.0)
    parser.add_argument(
        "--demo-forward-distance-m",
        type=float,
        default=20.0,
        help="If no waypoint/goal is supplied, place one this many meters ahead.",
    )
    parser.add_argument("--preview-width", type=int, default=848)
    parser.add_argument("--preview-height", type=int, default=480)
    parser.add_argument(
        "--max-fps",
        type=float,
        default=60.0,
        help="Maximum FPV preview and MP4 recording frame rate.",
    )
    parser.add_argument(
        "--video-path",
        default="/DroneDigitalTwin/video",
        help=(
            "Directory for timestamped FPV MP4 recordings, or an explicit .mp4 "
            "file path. Use an empty string to disable recording."
        ),
    )
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--first-frame-timeout-sec", type=float, default=20.0)
    parser.add_argument("--report-every-sec", type=float, default=2.0)
    parser.add_argument("--takeoff-timeout-sec", type=float, default=20.0)
    parser.add_argument("--land-timeout-sec", type=float, default=30.0)
    parser.add_argument("--arm-timeout-sec", type=float, default=60.0)
    parser.add_argument(
        "--px4-ready-timeout-sec",
        type=float,
        default=300.0,
        help="Seconds to wait for PX4 to connect, finish GPS/home readiness, and become armable.",
    )
    parser.add_argument("--velocity-mps", type=float, default=2.0)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--map-center", type=parse_vector3)
    parser.add_argument("--map-size", type=parse_size3)
    parser.add_argument("--map-margin-m", type=float, default=20.0)
    parser.add_argument("--min-map-size-m", type=float, default=50.0)
    parser.add_argument("--ground-z-ned", type=float, default=0.0)
    parser.set_defaults(replan_on_object=True)
    parser.add_argument(
        "--replan-on-object",
        "--stop-on-object",
        dest="replan_on_object",
        action="store_true",
        help=(
            "While flying, scan the route ahead for static objects, replace "
            "blocked route points with an A* bypass, and continue to the "
            "original goal. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-replan-on-object",
        "--no-stop-on-object",
        dest="replan_on_object",
        action="store_false",
        help="Disable the static object reroute check.",
    )
    parser.add_argument(
        "--dynamic-replan-lookahead-waypoints",
        type=int,
        default=1,
        help=(
            "Number of upcoming route legs checked before each move. The default "
            "checks only the immediate next leg so the drone follows the supplied "
            "route until it reaches a blocked segment."
        ),
    )
    parser.add_argument(
        "--dynamic-replan-max-count",
        type=int,
        default=3,
        help="Maximum number of live A* reroutes allowed during one mission.",
    )
    parser.add_argument(
        "--dynamic-replan-max-segment-m",
        type=float,
        default=0.0,
        help=(
            "Densify the active route so live replan checks happen at least this "
            "often in meters. Use 0 to keep only supplied/generated waypoints."
        ),
    )
    parser.add_argument(
        "--dynamic-replan-stop-hold-sec",
        type=float,
        default=1.0,
        help="Seconds to hover after detecting an object and before planning bypass.",
    )
    parser.add_argument(
        "--dynamic-replan-min-flight-sec",
        type=float,
        default=0.5,
        help="Minimum time to fly toward a waypoint before checking for live reroute.",
    )
    parser.add_argument(
        "--dynamic-replan-check-sec",
        type=float,
        default=0.5,
        help="Seconds between live obstacle checks while the path API is flying.",
    )
    parser.add_argument(
        "--dynamic-replan-detection-margin-m",
        type=float,
        default=4.0,
        help=(
            "Small margin around the immediate route leg used for live object "
            "detection. The larger --object-scan-margin-m is used only for A* "
            "reroute planning."
        ),
    )
    parser.add_argument(
        "--object-stop-distance-m",
        type=float,
        default=2.0,
        help="Distance before the first blocked route point where the reroute starts.",
    )
    parser.add_argument(
        "--object-path-clearance-m",
        type=float,
        default=1.0,
        help="Horizontal route corridor radius to treat as blocked by objects.",
    )
    parser.add_argument(
        "--object-scan-resolution-m",
        type=float,
        default=1.0,
        help="Voxel-grid resolution for static object route scanning.",
    )
    parser.add_argument(
        "--object-scan-sample-spacing-m",
        type=float,
        default=0.5,
        help="Spacing between route samples checked for static object occupancy.",
    )
    parser.add_argument(
        "--object-scan-margin-m",
        type=float,
        default=20.0,
        help=(
            "Extra meters added around the route when creating the scan/replan "
            "grid."
        ),
    )
    parser.add_argument(
        "--object-scan-min-size-m",
        type=float,
        default=10.0,
        help="Minimum x/y/z size for the static object scan grid.",
    )
    parser.add_argument(
        "--object-scan-start-ignore-m",
        type=float,
        default=1.0,
        help="Ignore occupied samples this close to the route start.",
    )
    parser.add_argument(
        "--replan-waypoint-spacing-m",
        type=float,
        default=4.0,
        help="Minimum spacing between generated A* reroute waypoints.",
    )
    parser.add_argument(
        "--replan-rejoin-waypoints-ahead",
        type=int,
        default=3,
        help=(
            "Number of upcoming route waypoints to skip/replace before "
            "automatically rejoining the normal route."
        ),
    )
    parser.add_argument(
        "--replan-rejoin-after-obstacle-m",
        type=float,
        default=4.0,
        help="Minimum route distance after the detected object before rejoining.",
    )
    parser.add_argument(
        "--replan-rejoin-clear-distance-m",
        type=float,
        default=3.0,
        help="Required clear route distance before automatic rejoin is accepted.",
    )
    parser.add_argument(
        "--replan-rejoin-point",
        type=parse_vector3,
        default=None,
        help=(
            "Optional explicit downstream point from the original route to rejoin "
            "after bypassing the object, for example \"45,17,-13\"."
        ),
    )
    parser.add_argument(
        "--replan-emergency-node",
        type=parse_vector3,
        default=None,
        help=(
            "Optional free NED waypoint, not required to be in --route, that the "
            "live reroute must pass through before reconnecting to the normal route."
        ),
    )
    parser.add_argument(
        "--replan-rejoin-tolerance-m",
        type=float,
        default=1.0,
        help="Tolerance for matching --replan-rejoin-point to the original route.",
    )
    parser.add_argument(
        "--min-altitude",
        "--min-altitude-m",
        dest="min_altitude",
        type=float,
        default=0.0,
        help=(
            "Minimum planned route altitude in meters above --ground-z-ned. "
            "In NED this clamps waypoint z to ground_z_ned - min_altitude. "
            "Use 0 to keep requested/planned z values unchanged."
        ),
    )
    parser.add_argument(
        "--waypoint-spacing-m",
        "--waypoint-distance-m",
        dest="waypoint_spacing_m",
        type=float,
        default=8.0,
        help=(
            "Minimum spacing in meters between planned A* route waypoints after "
            "sparsifying the dense grid path. Larger values create fewer middle "
            "waypoints."
        ),
    )
    parser.add_argument("--waypoint-acceptance-m", type=float, default=1.0)
    parser.add_argument(
        "--waypoint-hold-sec",
        type=float,
        default=0.0,
        help="Seconds to brake and hover at each intermediate planned waypoint.",
    )
    parser.add_argument(
        "--velocity-command-duration-sec",
        type=float,
        default=0.1,
        help=(
            "Duration of each PX4 velocity setpoint in velocity mode and while "
            "holding for live replans."
        ),
    )
    parser.add_argument(
        "--acceleration-limit-mps2",
        type=float,
        default=1.0,
        help="Maximum velocity change rate in velocity flight mode.",
    )
    parser.add_argument(
        "--velocity-lookahead-m",
        type=float,
        default=8.0,
        help=(
            "Distance ahead on the current route segment used by velocity mode. "
            "This prevents side-to-side waypoint-center chasing."
        ),
    )
    parser.add_argument(
        "--slowdown-distance-m",
        type=float,
        default=4.0,
        help="Distance over which velocity mode eases down near each waypoint.",
    )
    parser.add_argument(
        "--path-yaw-rate-dps",
        type=float,
        default=30.0,
        help="Maximum yaw rate used while facing the planned route.",
    )
    parser.add_argument(
        "--move-timeout-sec",
        type=float,
        default=45.0,
        help="Per-waypoint movement timeout while following the planned route.",
    )
    parser.add_argument(
        "--path-timeout-sec",
        type=float,
        default=0.0,
        help="MoveOnPath timeout. Use 0 to infer from path length and velocity.",
    )
    parser.add_argument(
        "--flight-driver",
        choices=["path-api", "velocity"],
        default="velocity",
        help=(
            "Use the PX4 offboard velocity route follower, or Project AirSim's "
            "MoveOnPath path API. The velocity driver matches scene-NED routes "
            "used with --start-as-scene-origin."
        ),
    )
    parser.set_defaults(face_travel_direction=True)
    parser.add_argument(
        "--face-travel-direction",
        dest="face_travel_direction",
        action="store_true",
        help="Yaw toward each planned waypoint while flying. Enabled by default.",
    )
    parser.add_argument(
        "--no-face-travel-direction",
        dest="face_travel_direction",
        action="store_false",
        help="Keep existing yaw while flying the planned route.",
    )
    parser.add_argument(
        "--lookahead-m",
        type=float,
        default=-1.0,
        help="MoveOnPath lookahead distance. Use -1 for automatic lookahead.",
    )
    parser.add_argument(
        "--adaptive-lookahead",
        type=float,
        default=1.0,
        help="MoveOnPath adaptive lookahead setting.",
    )
    parser.add_argument(
        "--skip-takeoff",
        action="store_true",
        help="Only load the scene and show the FPV overlay; do not arm or take off.",
    )
    parser.add_argument(
        "--no-fly-to-waypoint",
        action="store_true",
        help="Take off and hold while showing waypoint overlays; do not fly the route.",
    )
    parser.add_argument(
        "--keep-armed",
        action="store_true",
        help="Leave PX4 armed/API control enabled when the overlay exits.",
    )
    parser.set_defaults(land_at_goal=True)
    parser.add_argument(
        "--land-at-goal",
        dest="land_at_goal",
        action="store_true",
        help="Land after reaching the final planned waypoint.",
    )
    parser.add_argument(
        "--no-land-at-goal",
        dest="land_at_goal",
        action="store_false",
        help="Hold after reaching the final planned waypoint instead of landing.",
    )
    parser.add_argument(
        "--keep-overlay-after-mission",
        action="store_true",
        help="Keep the FPV overlay open after reaching/landing at the goal.",
    )
    parser.add_argument(
        "--camera-angle-deg",
        "--front-rgb-angle",
        dest="camera_angle_deg",
        type=float,
        default=None,
        metavar="DEG",
        help="Tilt the selected camera down by DEG degrees from straight ahead.",
    )
    parser.add_argument("--camera-capture-width", type=int, default=1280)
    parser.add_argument("--camera-capture-height", type=int, default=720)
    parser.add_argument("--camera-capture-interval-sec", type=float, default=0.03)
    parser.add_argument("--camera-fov-degrees", type=float, default=90.0)
    parser.add_argument("--no-edge-indicators", action="store_true")
    parser.add_argument("--no-world-markers", action="store_true")
    parser.add_argument("--flush-markers", action="store_true")
    parser.add_argument("--world-marker-size", type=float, default=16.0)
    parser.add_argument("--world-label-scale", type=float, default=1.0)
    parser.add_argument("--world-label-z-offset-m", type=float, default=1.0)
    parser.add_argument("--world-marker-duration-sec", type=float, default=3600.0)
    parser.add_argument(
        "--ignore-actor",
        action="append",
        default=None,
        help="Actor to ignore in the occupancy grid. Repeatable.",
    )
    parser.set_defaults(print_waypoints=True)
    parser.add_argument(
        "--print-waypoints",
        dest="print_waypoints",
        action="store_true",
        help="Print every planned waypoint.",
    )
    parser.add_argument(
        "--no-print-waypoints",
        dest="print_waypoints",
        action="store_false",
        help="Suppress per-waypoint route logging.",
    )
    parser.add_argument(
        "--persistent-world-markers",
        action="store_true",
        help="Keep debug point markers until flushed by another script/tool.",
    )
    parser.add_argument(
        "--wait-for-px4-ready",
        action="store_true",
        help="Wait for PX4 readiness even when --skip-takeoff is used.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args(normalize_vector_args(sys.argv[1:]))
    asyncio.run(run_overlay(parsed_args))
