"""
FPV waypoint overlay example for a PX4-style Project AirSim drone scene.

The script loads the same default PX4 scene used by px4_astar_autopilot.py,
opens a front RGB camera preview, and draws waypoint labels on top of that
camera image. It also drops debug markers in the Unreal scene so the waypoint
exists as a visible 3D reference in the FPV view.
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
from projectairsim.planners import AStarPlanner
from projectairsim.types import Pose, Quaternion, Vector3
from projectairsim.utils import (
    calculate_path_length,
    projectairsim_log,
    rpy_to_quaternion,
    unpack_image,
)

from px4_astar_autopilot import (
    fly_path_by_velocity,
    fly_to_point_by_velocity,
    infer_map_center,
    infer_map_size,
    parse_size3,
    sparsify_path,
    validate_grid_coordinate,
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
    vector_options = {"--start", "--goal", "--waypoint", "--map-center", "--map-size"}
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
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
    ):
        self.window_name = window_name
        self.waypoints = list(waypoints)
        self.fov_degrees = fov_degrees
        self.resize_x = resize_x
        self.resize_y = resize_y
        self.draw_edge_indicators = draw_edge_indicators
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

    def display_loop(self):
        import cv2

        created = False
        try:
            while self.running:
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

                if not created:
                    cv2.namedWindow(
                        self.window_name,
                        flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE,
                    )
                    created = True

                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) == 27:
                    self.running = False
        except Exception as exc:
            self.error = exc
            self.running = False
        finally:
            if created:
                cv2.destroyWindow(self.window_name)

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

        if camera_pose_from_image(image) is None:
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

        status_y = 54
        for index, waypoint in enumerate(self.waypoints):
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

    def draw_text_with_shadow(self, cv2, frame, text: str, origin: Tuple[int, int]):
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
            (0, 255, 255),
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


def waypoints_from_path(path: Sequence[Sequence[float]]) -> List[Waypoint]:
    final_index = len(path) - 1
    waypoints = []
    for index, point in enumerate(path):
        if index == 0:
            label = "START"
        elif index == final_index:
            label = "GOAL"
        else:
            label = f"WP{index:03d}"
        waypoints.append(
            Waypoint(label, [float(point[0]), float(point[1]), float(point[2])])
        )
    return waypoints


def plan_astar_route(world: World, args) -> List[List[float]]:
    if args.start is None:
        raise RuntimeError("--goal route planning requires --start")
    if args.goal is None:
        raise RuntimeError("A* route planning requires --goal")

    start = args.start
    goal = args.goal
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
    if args.goal is not None:
        return plan_astar_route(world, args)

    manual_waypoints = collect_waypoints(args, drone)
    if args.waypoint:
        start = args.start or get_pose_position_ned(drone)
        path = [start] + [waypoint.position for waypoint in manual_waypoints]
        if args.print_waypoints:
            for index, point in enumerate(path):
                projectairsim_log().info("Waypoint %03d: %s", index, point)
        return path

    return [waypoint.position for waypoint in manual_waypoints]


def plot_world_waypoint_markers(world: World, waypoints: Sequence[Waypoint], args):
    if args.no_world_markers:
        return

    points = [waypoint.position for waypoint in waypoints]
    labels = [waypoint.label for waypoint in waypoints]
    label_positions = [
        [
            waypoint.position[0],
            waypoint.position[1],
            waypoint.position[2] - args.world_label_z_offset_m,
        ]
        for waypoint in waypoints
    ]

    if args.flush_markers:
        world.flush_persistent_markers()

    world.plot_debug_points(
        points,
        [1.0, 1.0, 0.0, 1.0],
        args.world_marker_size,
        args.world_marker_duration_sec,
        args.persistent_world_markers,
    )
    world.plot_debug_strings(
        labels,
        label_positions,
        args.world_label_scale,
        [1.0, 1.0, 1.0, 1.0],
        args.world_marker_duration_sec,
    )
    projectairsim_log().info("Plotted %d waypoint marker(s) in the scene", len(points))


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
        "PX4 did not become armable. Check that PX4 is running, connected to "
        "Project AirSim on TCP port 4560, and has completed GPS/home readiness."
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
    has_explicit_flight_target = bool(args.goal is not None or args.waypoint)
    route_path = None
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
            waypoints = waypoints_from_path(route_path)
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
                await fly_to_point_by_velocity(
                    drone,
                    route_path[0],
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
                await fly_path_by_velocity(
                    drone,
                    route_path,
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
                    "Reached destination %s",
                    args.goal or route_path[-1],
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
                    "No explicit --goal/--waypoint supplied; holding after takeoff"
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
    parser.add_argument("--sim-config-path", default="sim_config/")
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--topics-port", type=int, default=8989)
    parser.add_argument("--services-port", type=int, default=8990)
    parser.add_argument("--load-delay-sec", type=float, default=2.0)
    parser.add_argument("--start", type=parse_vector3, help="NED x,y,z")
    parser.add_argument("--goal", type=parse_vector3, help="NED x,y,z waypoint")
    parser.add_argument(
        "--waypoint",
        action="append",
        type=parse_vector3,
        default=None,
        help="NED x,y,z waypoint to draw. Repeat for multiple waypoints.",
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
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=720)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--first-frame-timeout-sec", type=float, default=20.0)
    parser.add_argument("--report-every-sec", type=float, default=2.0)
    parser.add_argument("--takeoff-timeout-sec", type=float, default=20.0)
    parser.add_argument("--land-timeout-sec", type=float, default=30.0)
    parser.add_argument("--arm-timeout-sec", type=float, default=60.0)
    parser.add_argument("--px4-ready-timeout-sec", type=float, default=60.0)
    parser.add_argument("--velocity-mps", type=float, default=2.0)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--map-center", type=parse_vector3)
    parser.add_argument("--map-size", type=parse_size3)
    parser.add_argument("--map-margin-m", type=float, default=20.0)
    parser.add_argument("--min-map-size-m", type=float, default=50.0)
    parser.add_argument("--ground-z-ned", type=float, default=0.0)
    parser.add_argument("--waypoint-spacing-m", type=float, default=3.0)
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
        help="Duration of each PX4 velocity setpoint while following waypoints.",
    )
    parser.add_argument(
        "--acceleration-limit-mps2",
        type=float,
        default=2.0,
        help="Maximum velocity change rate while following waypoints.",
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
