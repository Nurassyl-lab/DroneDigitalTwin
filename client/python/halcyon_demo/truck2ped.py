"""
Keyboard drone + BP_Truck + BP_ThirdPersonCharacter controller for the
truck-to-pedestrian demo.

Examples:
    python truck2ped.py --start 0 0 -2
    python truck2ped.py --start "10,0,-3" --truck BP_Truck_1 --pedestrian BP_ThirdPersonCharacter_C_1
    python truck2ped.py --start 0 0 -2 --video-path .\video

Controls:
    W/S: drone forward/back
    A/D: drone left/right
    Up/Down: drone up/down
    Left/Right: drone yaw
    N: stop truck
    M: set truck speed to 600
    Z: set pedestrian speed to 3
    X: stop pedestrian
    L: land
    Q: quit

The front RGB window records to MP4 by default and includes a third-person
drone view in the upper-right corner. Use --video-path "" to disable recording.

The truck movement remains inside BP_Truck Tick. The pedestrian movement should
remain inside BP_ThirdPersonCharacter Tick using CharacterMovement/AddMovementInput.
Python only sends SetTruckSpeed(NewSpeed) and SetPedestrianSpeed(NewSpeed).
"""

import argparse
import asyncio
import math
from pathlib import Path
import queue
import re
import shutil
import tempfile
from threading import Lock, Thread
import time

import commentjson
import projectairsim
from projectairsim import Drone, World
from projectairsim.types import BoxAlignment, Pose, Quaternion, Vector3
from projectairsim.utils import projectairsim_log, rpy_to_quaternion, unpack_image

keyboard = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SIM_CONFIG_PATH = SCRIPT_DIR.parent / "example_user_scripts" / "sim_config"


def parse_vector3(value):
    if isinstance(value, list):
        parts = value
    else:
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


def parse_start(values):
    if len(values) == 1:
        return parse_vector3(values[0])
    return parse_vector3(values)


def parse_float3(value, default=None):
    if value is None:
        return default if default is not None else [0.0, 0.0, 0.0]
    return [float(part) for part in value.replace(",", " ").split()]


def make_pose_ned(position_ned):
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


def make_camera_pose(translation_xyz, angle_down_deg):
    w, x, y, z = rpy_to_quaternion(0.0, math.radians(-angle_down_deg), 0.0)
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


def make_third_person_camera_pose(distance_m, height_m, pitch_deg):
    distance_m = max(0.1, float(distance_m))
    height_m = max(0.0, float(height_m))
    w, x, y, z = rpy_to_quaternion(0.0, math.radians(-pitch_deg), 0.0)
    return Pose(
        {
            "translation": Vector3(
                {
                    "x": -distance_m,
                    "y": 0.0,
                    "z": -height_m,
                }
            ),
            "rotation": Quaternion({"w": w, "x": x, "y": y, "z": z}),
            "frame_id": "DEFAULT_ID",
        }
    )


def get_pose_position_ned(drone):
    position = drone.get_ground_truth_kinematics()["pose"]["position"]
    return [float(position["x"]), float(position["y"]), float(position["z"])]


def print_live_ned(drone, last_print_at, interval_sec, force=False):
    now = time.time()
    if not force and now - last_print_at < interval_sec:
        return last_print_at

    position = get_pose_position_ned(drone)
    print(
        f"[LIVE NED] x={position[0]:8.2f}  "
        f"y={position[1]:8.2f}  z={position[2]:8.2f}",
        flush=True,
    )
    return now


def move_toward(current, target, max_delta):
    if current < target:
        return min(current + max_delta, target)
    if current > target:
        return max(current - max_delta, target)
    return current


def quaternion_to_matrix(w, x, y, z):
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


def camera_pose_from_image(image):
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


def world_to_camera(point_ned, camera_position, camera_rotation_matrix):
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


def project_ned_point(point_ned, image, fov_degrees):
    camera_pose = camera_pose_from_image(image)
    if camera_pose is None:
        return None

    width = int(image["width"])
    height = int(image["height"])
    camera_position, camera_rotation_matrix = camera_pose
    point_camera = world_to_camera(point_ned, camera_position, camera_rotation_matrix)
    forward_m = point_camera[0]
    if forward_m <= 0.05:
        return None

    focal_px = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    pixel_x = width * 0.5 + focal_px * (point_camera[1] / forward_m)
    pixel_y = height * 0.5 + focal_px * (point_camera[2] / forward_m)
    return pixel_x, pixel_y, point_camera


def dict_vector_to_list(vector):
    return [float(vector["x"]), float(vector["y"]), float(vector["z"])]


def pose_translation_ned(pose):
    return dict_vector_to_list(pose["translation"])


def bbox_corners_from_center_size(center, size):
    half = [component * 0.5 for component in size]
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(
                    [
                        center[0] + sx * half[0],
                        center[1] + sy * half[1],
                        center[2] + sz * half[2],
                    ]
                )
    return corners


def fallback_pedestrian_bbox_corners(position_ned, args):
    center = [
        position_ned[0],
        position_ned[1],
        position_ned[2] - args.pedestrian_box_height_m * 0.5,
    ]
    size = [
        args.pedestrian_box_depth_m,
        args.pedestrian_box_width_m,
        args.pedestrian_box_height_m,
    ]
    return bbox_corners_from_center_size(center, size)


def resolve_config_path(config_name, sim_config_path):
    path = Path(config_name)
    if path.is_absolute():
        return path
    return Path(sim_config_path) / config_name


def load_jsonc(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return commentjson.load(handle)


def write_jsonc(path, data):
    Path(path).write_text(commentjson.dumps(data, indent=2) + "\n", encoding="utf-8")


def prepare_video_output_path(video_path):
    if video_path is None or str(video_path) == "":
        print("FPV video recording disabled.")
        return None

    requested_path = Path(video_path).expanduser()
    if requested_path.suffix.lower() == ".mp4":
        output_dir = requested_path.parent
        output_path = requested_path
    else:
        output_dir = requested_path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"truck2ped_fpv_{timestamp}.mp4"

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise RuntimeError(f"FPV video path is not a directory: {output_dir}")

    return output_path


def default_front_camera_sensor(args):
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


def default_third_person_camera_sensor(args):
    return {
        "id": args.third_person_camera,
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
        "gimbal": {
            "lock-roll": True,
            "lock-pitch": True,
            "lock-yaw": False,
        },
        "origin": {
            "xyz": (
                f"-{max(0.1, args.third_person_camera_distance_m):g} "
                f"0.0 -{max(0.0, args.third_person_camera_height_m):g}"
            ),
            "rpy-deg": f"0 {-args.third_person_camera_pitch_deg:g} 0",
        },
    }


def ensure_scene_camera_capture(sensor, args):
    capture_settings = sensor.setdefault("capture-settings", [])
    scene_capture = next(
        (capture for capture in capture_settings if capture.get("image-type") == 0),
        None,
    )
    if scene_capture is None:
        capture_settings.append(default_front_camera_sensor(args)["capture-settings"][0])
        return

    scene_capture["capture-enabled"] = True
    scene_capture["streaming-enabled"] = True
    scene_capture.setdefault("width", args.camera_capture_width)
    scene_capture.setdefault("height", args.camera_capture_height)
    scene_capture.setdefault("fov-degrees", args.camera_fov_degrees)
    scene_capture.setdefault("pixels-as-float", False)
    scene_capture.setdefault("compress", False)
    scene_capture.setdefault("target-gamma", 2.5)


def ensure_requested_camera(robot_config, camera_sensor_id, args):
    sensors = robot_config.setdefault("sensors", [])
    sensor = next(
        (candidate for candidate in sensors if candidate.get("id") == camera_sensor_id),
        None,
    )

    if sensor is None and camera_sensor_id == "FrontCamera":
        sensors.append(default_front_camera_sensor(args))
        print("Added runtime FrontCamera RGB sensor to the drone config.")
        return

    if sensor is None and camera_sensor_id == args.third_person_camera:
        sensors.append(default_third_person_camera_sensor(args))
        print(
            f"Added runtime {args.third_person_camera} third-person RGB sensor "
            "to the drone config."
        )
        return

    if sensor is not None and sensor.get("type") == "camera":
        sensor["enabled"] = True
        ensure_scene_camera_capture(sensor, args)


def make_runtime_scene_config(args):
    scene_path = resolve_config_path(args.sceneconfigfile, args.simconfigpath)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene config not found: {scene_path}")

    scene_config = load_jsonc(scene_path)
    target_actor = next(
        (
            actor
            for actor in scene_config.get("actors", [])
            if actor.get("type") == "robot" and actor.get("name") == args.drone_name
        ),
        None,
    )
    if target_actor is None or not target_actor.get("robot-config"):
        raise RuntimeError(
            f"Could not find robot actor '{args.drone_name}' with a robot-config"
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="truck2ped_scene_")
    temp_config_dir = Path(temp_dir.name)

    try:
        for actor_index, actor in enumerate(scene_config.get("actors", [])):
            if actor.get("type") != "robot" or not actor.get("robot-config"):
                continue

            robot_config_path = resolve_config_path(
                actor["robot-config"],
                scene_path.parent,
            )
            robot_config = load_jsonc(robot_config_path)

            if actor is target_actor:
                ensure_requested_camera(robot_config, args.camera, args)
                if not args.no_third_person_overlay:
                    ensure_requested_camera(
                        robot_config,
                        args.third_person_camera,
                        args,
                    )

            output_name = f"{robot_config_path.stem}_{actor_index}_truck2ped.jsonc"
            write_jsonc(temp_config_dir / output_name, robot_config)
            actor["robot-config"] = output_name

        for env_actor in scene_config.get("environment-actors", []):
            if env_actor.get("type") != "env_actor" or not env_actor.get(
                "env-actor-config"
            ):
                continue
            env_config_path = resolve_config_path(
                env_actor["env-actor-config"],
                scene_path.parent,
            )
            shutil.copy2(env_config_path, temp_config_dir / env_config_path.name)
            env_actor["env-actor-config"] = env_config_path.name

        write_jsonc(temp_config_dir / scene_path.name, scene_config)
    except Exception:
        temp_dir.cleanup()
        raise

    return temp_dir, scene_path.name, str(temp_config_dir)


def read_camera_origin(scene_name, sim_config_path, drone_name, camera_sensor_id):
    try:
        scene_config = load_jsonc(resolve_config_path(scene_name, sim_config_path))
        actor = next(
            (
                item
                for item in scene_config.get("actors", [])
                if item.get("type") == "robot" and item.get("name") == drone_name
            ),
            None,
        )
        if actor is None or not actor.get("robot-config"):
            return [0.5, 0.0, 0.0]

        robot_config = load_jsonc(
            resolve_config_path(actor["robot-config"], sim_config_path)
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
            return [0.5, 0.0, 0.0]
        return parse_float3(sensor.get("origin", {}).get("xyz"), [0.5, 0.0, 0.0])
    except Exception as exc:
        projectairsim_log().warning(
            "Could not read %s origin; using default front camera origin: %s",
            camera_sensor_id,
            exc,
        )
        return [0.5, 0.0, 0.0]


def read_camera_fov_degrees(
    scene_name,
    sim_config_path,
    drone_name,
    camera_sensor_id,
    default_fov_degrees,
):
    try:
        scene_config = load_jsonc(resolve_config_path(scene_name, sim_config_path))
        actor = next(
            (
                item
                for item in scene_config.get("actors", [])
                if item.get("type") == "robot" and item.get("name") == drone_name
            ),
            None,
        )
        if actor is None or not actor.get("robot-config"):
            return default_fov_degrees

        robot_config = load_jsonc(
            resolve_config_path(actor["robot-config"], sim_config_path)
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
            "Could not read %s FOV; using %.1f deg: %s",
            camera_sensor_id,
            default_fov_degrees,
            exc,
        )
        return default_fov_degrees


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fly Drone1 while controlling BP_Truck and BP_ThirdPersonCharacter."
        )
    )
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--topicsport", type=int, default=8989)
    parser.add_argument("--servicesport", type=int, default=8990)
    parser.add_argument("--sceneconfigfile", default="scene_basic_drone.jsonc")
    parser.add_argument(
        "--simconfigpath",
        default=str(DEFAULT_SIM_CONFIG_PATH),
        help="Directory containing ProjectAirSim scene config files.",
    )
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument(
        "--start",
        nargs="+",
        type=str,
        default=None,
        help="Optional drone spawn NED x y z, for example --start 0 0 -2.",
    )
    parser.add_argument(
        "--truck",
        default="Truck",
        help="Truck actor name/substring or Unreal tag. Defaults to 'Truck'.",
    )
    parser.add_argument(
        "--pedestrian",
        default="BP_ThirdPersonCharacter",
        help=(
            "Pedestrian actor name/substring or Unreal tag. Defaults to "
            "'BP_ThirdPersonCharacter'."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Optional initial truck TargetSpeed before keyboard control starts.",
    )
    parser.add_argument(
        "--pedestrian-speed",
        type=float,
        default=None,
        help="Optional initial PedestrianSpeed before keyboard control starts.",
    )
    parser.add_argument("--truck-stop-speed", type=float, default=0.0)
    parser.add_argument("--truck-move-speed", type=float, default=600.0)
    parser.add_argument("--pedestrian-stop-speed", type=float, default=0.0)
    parser.add_argument("--pedestrian-walk-speed", type=float, default=0.3)
    parser.add_argument("--pedestrian-z-speed", type=float, default=3.0)
    parser.add_argument(
        "--flight-speed",
        type=float,
        default=5.0,
        help="Maximum drone body-frame speed in m/s.",
    )
    parser.add_argument(
        "--flight-acceleration",
        type=float,
        default=2.0,
        help="How quickly drone velocity changes in m/s^2.",
    )
    parser.add_argument(
        "--yaw-speed",
        type=float,
        default=20.0,
        help="Maximum drone yaw rate in deg/s.",
    )
    parser.add_argument(
        "--yaw-acceleration",
        type=float,
        default=45.0,
        help="How quickly drone yaw rate changes in deg/s^2.",
    )
    parser.add_argument("--command-duration-sec", type=float, default=0.1)
    parser.add_argument("--live-ned-interval-sec", type=float, default=0.5)
    parser.add_argument("--no-live-ned", action="store_true")
    parser.add_argument(
        "--camera",
        default="FrontCamera",
        help="RGB camera sensor id to show and tilt. Defaults to FrontCamera.",
    )
    parser.add_argument(
        "--front-rgb-angle",
        "--camera-angle-deg",
        dest="camera_angle_deg",
        type=float,
        default=22.0,
        help="Tilt the selected front RGB camera down by this many degrees.",
    )
    parser.add_argument("--camera-capture-width", type=int, default=1280)
    parser.add_argument("--camera-capture-height", type=int, default=720)
    parser.add_argument("--camera-capture-interval-sec", type=float, default=0.03)
    parser.add_argument("--camera-fov-degrees", type=float, default=90.0)
    parser.add_argument("--camera-display-width", type=int, default=960)
    parser.add_argument("--camera-display-height", type=int, default=540)
    parser.add_argument(
        "--max-fps",
        type=float,
        default=60.0,
        help="Maximum front RGB preview and MP4 recording frame rate.",
    )
    parser.add_argument(
        "--video-path",
        default=str(SCRIPT_DIR / "video"),
        help=(
            "Directory for timestamped FPV MP4 recordings, or an explicit .mp4 "
            "file path. Use an empty string to disable recording."
        ),
    )
    parser.add_argument(
        "--third-person-camera",
        default="Chase",
        help="Camera used for the upper-right third-person drone inset.",
    )
    parser.add_argument(
        "--third-person-camera-distance-m",
        type=float,
        default=10.0,
        help="How far behind the drone the third-person camera is placed.",
    )
    parser.add_argument(
        "--third-person-camera-height-m",
        type=float,
        default=1.5,
        help="How far above the drone the third-person camera is placed.",
    )
    parser.add_argument(
        "--third-person-camera-pitch-deg",
        type=float,
        default=12.0,
        help="Downward pitch angle for the third-person camera.",
    )
    parser.add_argument(
        "--third-person-overlay-width-frac",
        type=float,
        default=0.28,
        help="Inset width as a fraction of the front RGB frame width.",
    )
    parser.add_argument(
        "--third-person-overlay-margin-px",
        type=int,
        default=16,
        help="Pixel margin from the top/right edge for the third-person inset.",
    )
    parser.add_argument(
        "--no-third-person-overlay",
        action="store_true",
        help="Disable the upper-right third-person picture-in-picture inset.",
    )
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--no-pedestrian-box",
        action="store_true",
        help="Disable the red pedestrian box overlay in the front RGB window.",
    )
    parser.add_argument(
        "--pedestrian-position-interval-sec",
        type=float,
        default=0.2,
        help="How often to read/print pedestrian NED position.",
    )
    parser.add_argument(
        "--no-pedestrian-position",
        action="store_true",
        help="Disable periodic pedestrian NED position printing.",
    )
    parser.add_argument("--pedestrian-box-width-m", type=float, default=0.8)
    parser.add_argument("--pedestrian-box-depth-m", type=float, default=0.8)
    parser.add_argument("--pedestrian-box-height-m", type=float, default=1.8)
    return parser


def find_matching_actors(world, pattern):
    try:
        return world.list_objects(pattern)
    except Exception:
        return []


def resolve_unreal_actor(world, actor_name_or_tag, fallback_pattern, label):
    escaped = re.escape(actor_name_or_tag)
    exact_or_contains = find_matching_actors(world, f".*{escaped}.*")
    if exact_or_contains:
        if actor_name_or_tag in exact_or_contains:
            return actor_name_or_tag
        return exact_or_contains[0]

    unreal_instance_match = re.match(r"^(.*)_(\d+)$", actor_name_or_tag)
    if unreal_instance_match and not actor_name_or_tag.endswith("_C"):
        candidate = f"{unreal_instance_match.group(1)}_C_{unreal_instance_match.group(2)}"
        candidate_matches = find_matching_actors(world, f".*{re.escape(candidate)}.*")
        if candidate_matches:
            return candidate_matches[0]

    bp_matches = find_matching_actors(world, fallback_pattern)
    if bp_matches:
        print(f"Available {label}-like actor names:")
        for name in bp_matches:
            print(f"  {name}")
        return bp_matches[0]

    return actor_name_or_tag


def resolve_truck_actor(world, truck_name_or_tag):
    return resolve_unreal_actor(
        world,
        truck_name_or_tag,
        r".*BP_Truck.*",
        "BP_Truck",
    )


def resolve_pedestrian_actor(world, pedestrian_name_or_tag):
    return resolve_unreal_actor(
        world,
        pedestrian_name_or_tag,
        r".*(BP_ThirdPersonCharacter|Pedestrian|ThirdPerson).*",
        "BP_ThirdPersonCharacter",
    )


def set_truck_speed(world, truck_name_or_tag, speed):
    truck_actor = resolve_truck_actor(world, truck_name_or_tag)
    if truck_actor != truck_name_or_tag:
        print(f"Resolved truck '{truck_name_or_tag}' to actor '{truck_actor}'.")

    print(f"Setting {truck_actor} target speed to {speed:g}...")

    try:
        called = world.call_actor_event(
            truck_actor,
            "SetTruckSpeed",
            {"NewSpeed": speed},
        )
    except RuntimeError as exc:
        if "CallActorFloatEvent" in str(exc) and "not supported" in str(exc):
            print(
                "The running Unreal plugin does not expose CallActorFloatEvent. "
                "Recompile/rebuild the WalkingNPCs ProjectAirSim plugin, restart "
                "the editor, press Play, then run this script again."
            )
        return False

    if called:
        print(
            "Called BP_Truck.SetTruckSpeed(NewSpeed); "
            "Blueprint will interpolate CurrentSpeed."
        )
        return True

    print("SetTruckSpeed was not found; trying direct TargetSpeed variable set...")
    set_property = world.set_actor_float_property(truck_actor, "TargetSpeed", speed)
    if set_property:
        print("Set BP_Truck TargetSpeed variable directly.")
        return True

    print(
        "Could not find the truck/event/property. Check the actor name/tag, "
        "SetTruckSpeed(NewSpeed), and TargetSpeed variable."
    )
    return False


def set_pedestrian_speed(world, pedestrian_name_or_tag, speed):
    pedestrian_actor = resolve_pedestrian_actor(world, pedestrian_name_or_tag)
    if pedestrian_actor != pedestrian_name_or_tag:
        print(
            f"Resolved pedestrian '{pedestrian_name_or_tag}' "
            f"to actor '{pedestrian_actor}'."
        )

    print(f"Setting {pedestrian_actor} pedestrian speed to {speed:g}...")

    try:
        called = world.call_actor_event(
            pedestrian_actor,
            "SetPedestrianSpeed",
            {"NewSpeed": speed},
        )
    except RuntimeError as exc:
        if "CallActorFloatEvent" in str(exc) and "not supported" in str(exc):
            print(
                "The running Unreal plugin does not expose CallActorFloatEvent. "
                "Recompile/rebuild the WalkingNPCs ProjectAirSim plugin, restart "
                "the editor, press Play, then run this script again."
            )
        return False

    if called:
        print("Called BP_ThirdPersonCharacter.SetPedestrianSpeed(NewSpeed).")
        return True

    print(
        "SetPedestrianSpeed was not found; trying direct "
        "PedestrianSpeed variable set..."
    )
    set_property = world.set_actor_float_property(
        pedestrian_actor,
        "PedestrianSpeed",
        speed,
    )
    if set_property:
        print("Set BP_ThirdPersonCharacter PedestrianSpeed variable directly.")
        return True

    print(
        "Could not find the pedestrian/event/property. Check the actor name/tag, "
        "SetPedestrianSpeed(NewSpeed), and PedestrianSpeed variable."
    )
    return False


def get_pedestrian_position_ned(world, pedestrian_actor):
    return pose_translation_ned(world.get_object_pose(pedestrian_actor))


def update_pedestrian_tracking(world, args, display, last_error):
    try:
        position = None
        pose_error = None
        try:
            position = get_pedestrian_position_ned(world, args.pedestrian)
        except Exception as exc:
            pose_error = exc

        bbox_corners = None
        if display is not None and not args.no_pedestrian_box:
            try:
                bbox = world.get_3d_bounding_box(
                    args.pedestrian,
                    BoxAlignment.WORLD_AXIS,
                )
                if bbox and bbox.get("center") and bbox.get("size"):
                    center = dict_vector_to_list(bbox["center"])
                    size = dict_vector_to_list(bbox["size"])
                    if all(component > 0.01 for component in size):
                        bbox_corners = bbox_corners_from_center_size(center, size)
                        if position is None:
                            position = center
            except Exception:
                pass

            if bbox_corners is None and position is not None:
                bbox_corners = fallback_pedestrian_bbox_corners(position, args)
            if position is not None:
                display.set_pedestrian_state(args.pedestrian, position, bbox_corners)

        if position is None and pose_error is not None:
            raise pose_error
        if position is None:
            raise RuntimeError("pedestrian position unavailable")
        return position, None
    except Exception as exc:
        error = str(exc)
        if error != last_error:
            print(f"Could not read pedestrian position for {args.pedestrian}: {exc}")
        return None, error


class PedestrianOverlayDisplay:
    def __init__(
        self,
        window_name,
        fov_degrees,
        resize_x,
        resize_y,
        max_fps,
        video_output_path,
        third_person_overlay,
        third_person_overlay_width_frac,
        third_person_overlay_margin_px,
        third_person_label,
        draw_box=True,
    ):
        self.window_name = window_name
        self.fov_degrees = fov_degrees
        self.resize_x = resize_x
        self.resize_y = resize_y
        self.max_fps = max(1.0, float(max_fps))
        self.video_output_path = video_output_path
        self.video_writer = None
        self.third_person_overlay = third_person_overlay
        self.third_person_overlay_width_frac = max(
            0.1,
            min(0.6, float(third_person_overlay_width_frac)),
        )
        self.third_person_overlay_margin_px = max(0, int(third_person_overlay_margin_px))
        self.third_person_label = third_person_label
        self.third_person_image = None
        self.draw_box = draw_box
        self.running = False
        self.thread = None
        self.image_queue = queue.SimpleQueue()
        self.buffer_size = 3
        self.lock = Lock()
        self.pedestrian_actor = None
        self.pedestrian_position_ned = None
        self.pedestrian_bbox_corners_ned = None
        self.error = None
        self.frame_count = 0
        self._last_frame_time = None
        self._frame_intervals = []
        self._pending_first_frame = None

    def start(self):
        if self.thread:
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

    def receive_third_person(self, image):
        if not self.running or image is None:
            return
        self.third_person_image = image

    def set_pedestrian_state(self, actor_name, position_ned, bbox_corners_ned):
        with self.lock:
            self.pedestrian_actor = actor_name
            self.pedestrian_position_ned = position_ned
            self.pedestrian_bbox_corners_ned = bbox_corners_ned

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

                now = time.monotonic()
                if self._last_frame_time is not None:
                    interval = now - self._last_frame_time
                    if interval > 0.0:
                        self._frame_intervals.append(interval)
                        if len(self._frame_intervals) > 30:
                            self._frame_intervals.pop(0)
                self._last_frame_time = now

                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)

                self.frame_count += 1
                if self.draw_box:
                    self.draw_pedestrian_overlay(cv2, frame, image)
                self.draw_third_person_overlay(cv2, frame)

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

    def _compute_video_fps(self):
        if not self._frame_intervals:
            return self.max_fps
        average_interval = sum(self._frame_intervals) / len(self._frame_intervals)
        return min(self.max_fps, max(1.0, 1.0 / max(average_interval, 1e-6)))

    def write_video_frame(self, cv2, frame):
        if self.video_output_path is None:
            return

        if self.video_writer is None:
            if self._pending_first_frame is None:
                self._pending_first_frame = frame.copy()
                return

            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_fps = self._compute_video_fps()
            self.video_writer = cv2.VideoWriter(
                str(self.video_output_path),
                fourcc,
                video_fps,
                (width, height),
            )
            if not self.video_writer.isOpened():
                self.video_writer = None
                raise RuntimeError(
                    f"Could not open FPV video writer: {self.video_output_path}"
                )
            print(f"Recording FPV video to {self.video_output_path} at {video_fps:.1f} FPS.")
            self.video_writer.write(self._pending_first_frame)
            self._pending_first_frame = None

        self.video_writer.write(frame)

    def draw_third_person_overlay(self, cv2, frame):
        if not self.third_person_overlay or self.third_person_image is None:
            return

        inset = unpack_image(self.third_person_image)
        if inset is None:
            return
        if inset.ndim == 2:
            inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        elif inset.ndim == 3 and inset.shape[2] == 1:
            inset = cv2.cvtColor(inset[:, :, 0], cv2.COLOR_GRAY2BGR)

        frame_height, frame_width = frame.shape[:2]
        inset_height, inset_width = inset.shape[:2]
        if frame_height <= 0 or frame_width <= 0 or inset_height <= 0 or inset_width <= 0:
            return

        margin = self.third_person_overlay_margin_px
        available_width = max(1, frame_width - (2 * margin))
        available_height = max(1, frame_height - (2 * margin))
        target_width = min(
            available_width,
            max(80, int(frame_width * self.third_person_overlay_width_frac)),
        )
        target_height = max(1, int(target_width * inset_height / inset_width))
        max_height = max(1, int(frame_height * 0.42))
        if target_height > max_height:
            target_height = min(max_height, available_height)
            target_width = max(1, int(target_height * inset_width / inset_height))

        target_width = min(target_width, available_width)
        target_height = min(target_height, available_height)
        x0 = max(0, frame_width - margin - target_width)
        y0 = max(0, margin)
        x1 = min(frame_width, x0 + target_width)
        y1 = min(frame_height, y0 + target_height)
        if x1 <= x0 or y1 <= y0:
            return

        inset_resized = cv2.resize(inset, (x1 - x0, y1 - y0))
        frame[y0:y1, x0:x1] = inset_resized
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 2)
        cv2.rectangle(frame, (x0, y0), (x1 - 1, min(y1 - 1, y0 + 24)), (20, 20, 20), -1)
        cv2.putText(
            frame,
            self.third_person_label,
            (x0 + 8, y0 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_pedestrian_overlay(self, cv2, frame, image):
        height, width = frame.shape[:2]
        box = self.annotation_box(image)
        with self.lock:
            actor_name = self.pedestrian_actor
            position_ned = (
                list(self.pedestrian_position_ned)
                if self.pedestrian_position_ned is not None
                else None
            )
            corners = (
                list(self.pedestrian_bbox_corners_ned)
                if self.pedestrian_bbox_corners_ned is not None
                else None
            )

        if box is None and corners:
            box = self.project_bbox_to_image(corners, image, width, height)

        if box is None and position_ned:
            projection = project_ned_point(position_ned, image, self.fov_degrees)
            if projection is not None:
                x, y, _ = projection
                marker_size = 24
                box = (
                    x - marker_size,
                    y - marker_size,
                    x + marker_size,
                    y + marker_size,
                )

        if box is None:
            return

        clipped_box = self.clip_box(box, width, height)
        if clipped_box is None:
            return

        x1, y1, x2, y2 = clipped_box
        red = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2, cv2.LINE_AA)
        label = "Pedestrian"
        if actor_name:
            label = actor_name
        if position_ned:
            label += (
                f" NED {position_ned[0]:.1f}, "
                f"{position_ned[1]:.1f}, {position_ned[2]:.1f}"
            )
        self.draw_text_with_shadow(cv2, frame, label, (x1 + 4, max(18, y1 - 8)))

    def annotation_box(self, image):
        annotations = image.get("annotations") or []
        with self.lock:
            actor_name = self.pedestrian_actor
        for annotation in annotations:
            object_id = str(annotation.get("object_id", ""))
            if actor_name and actor_name not in object_id and object_id not in actor_name:
                continue
            bbox = annotation.get("bbox2d")
            if not bbox:
                continue
            center = bbox.get("center", {})
            size = bbox.get("size", {})
            if not center or not size:
                continue
            width = float(size["x"])
            height = float(size["y"])
            if width <= 1.0 or height <= 1.0:
                continue
            x = float(center["x"])
            y = float(center["y"])
            return x - width * 0.5, y - height * 0.5, x + width * 0.5, y + height * 0.5
        return None

    def project_bbox_to_image(self, corners_ned, image, width, height):
        projected = []
        for corner in corners_ned:
            projection = project_ned_point(corner, image, self.fov_degrees)
            if projection is None:
                continue
            projected.append((projection[0], projection[1]))

        if not projected:
            return None

        xs = [point[0] for point in projected]
        ys = [point[1] for point in projected]
        margin = 4.0
        box = min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin
        return self.clip_box(box, width, height, return_none_if_outside=True)

    def clip_box(self, box, width, height, return_none_if_outside=True):
        x1, y1, x2, y2 = box
        if x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
            return None if return_none_if_outside else (0, 0, 0, 0)
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def draw_text_with_shadow(self, cv2, frame, text, origin):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        cv2.putText(
            frame,
            text,
            origin,
            font,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            font,
            scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )


async def takeoff(drone):
    print("Arming the drone...")
    drone.arm()
    print("Taking off...")
    await drone.takeoff_async()
    time.sleep(1)


async def land(drone):
    print("Landing...")
    await drone.land_async()
    print("Disarming the drone...")
    drone.disarm()


def require_keyboard():
    global keyboard
    if keyboard is not None:
        return keyboard

    try:
        import keyboard as keyboard_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This script needs the 'keyboard' Python package. Install it in "
            "DroneSimDev_ENV with: pip install keyboard"
        ) from exc

    keyboard = keyboard_module
    return keyboard


def require_scene_camera_topic(drone, camera_sensor_id):
    if camera_sensor_id not in drone.sensors:
        raise RuntimeError(
            f"Camera sensor '{camera_sensor_id}' is not available. "
            f"Available sensors: {sorted(drone.sensors.keys())}"
        )
    if "scene_camera" not in drone.sensors[camera_sensor_id]:
        raise RuntimeError(
            f"Sensor '{camera_sensor_id}' has no RGB scene_camera topic. "
            f"Available topics: {sorted(drone.sensors[camera_sensor_id].keys())}"
        )
    return drone.sensors[camera_sensor_id]["scene_camera"]


def start_front_rgb_display(client, drone, args, scene_name, sim_config_path):
    camera_origin = read_camera_origin(
        scene_name,
        sim_config_path,
        args.drone_name,
        args.camera,
    )
    pose = make_camera_pose(camera_origin, args.camera_angle_deg)
    if drone.set_camera_pose(args.camera, pose):
        print(
            f"Set {args.camera} RGB camera angle to "
            f"{args.camera_angle_deg:g} degrees down."
        )
    else:
        print(f"Warning: failed to set {args.camera} camera pose.")

    camera_topic = require_scene_camera_topic(drone, args.camera)
    fov_degrees = read_camera_fov_degrees(
        scene_name,
        sim_config_path,
        args.drone_name,
        args.camera,
        args.camera_fov_degrees,
    )
    window_name = f"Front RGB ({args.camera})"

    third_person_topic = None
    third_person_enabled = not args.no_third_person_overlay
    if third_person_enabled:
        try:
            third_person_pose = make_third_person_camera_pose(
                args.third_person_camera_distance_m,
                args.third_person_camera_height_m,
                args.third_person_camera_pitch_deg,
            )
            if drone.set_camera_pose(args.third_person_camera, third_person_pose):
                print(
                    f"Set {args.third_person_camera} third-person camera "
                    f"{args.third_person_camera_distance_m:g}m behind, "
                    f"{args.third_person_camera_height_m:g}m above, "
                    f"pitch {args.third_person_camera_pitch_deg:g} degrees down."
                )
            else:
                print(
                    f"Warning: failed to set {args.third_person_camera} "
                    "third-person camera pose; using configured pose."
                )
            third_person_topic = require_scene_camera_topic(
                drone,
                args.third_person_camera,
            )
        except Exception as exc:
            print(
                f"Third-person inset disabled because camera "
                f"'{args.third_person_camera}' is not available: {exc}"
            )
            third_person_enabled = False

    video_output_path = prepare_video_output_path(args.video_path)
    image_display = PedestrianOverlayDisplay(
        window_name,
        fov_degrees,
        resize_x=args.camera_display_width,
        resize_y=args.camera_display_height,
        max_fps=args.max_fps,
        video_output_path=video_output_path,
        third_person_overlay=third_person_enabled,
        third_person_overlay_width_frac=args.third_person_overlay_width_frac,
        third_person_overlay_margin_px=args.third_person_overlay_margin_px,
        third_person_label="3rd Person View",
        draw_box=not args.no_pedestrian_box,
    )
    image_display.start()
    client.subscribe(
        camera_topic,
        lambda _, image: image_display.receive(image),
    )
    camera_topics = [camera_topic]
    print(f"Subscribed front RGB camera topic: {camera_topic}")
    if third_person_enabled and third_person_topic:
        client.subscribe(
            third_person_topic,
            lambda _, image: image_display.receive_third_person(image),
        )
        camera_topics.append(third_person_topic)
        print(f"Subscribed third-person camera topic: {third_person_topic}")
    if not args.no_pedestrian_box:
        print(f"Drawing red pedestrian box using {fov_degrees:g} degree camera FOV.")
    return image_display, camera_topics


async def run_keyboard_control(drone, world, args, pedestrian_display=None):
    keyboard_module = require_keyboard()

    drone.enable_api_control()
    await takeoff(drone)

    print("\n--- Drone + Truck Keyboard Control ---")
    print("W/S: drone forward/back")
    print("A/D: drone left/right")
    print("Up/Down: drone up/down")
    print("Left/Right: drone yaw")
    print(f"N: truck TargetSpeed={args.truck_stop_speed:g}")
    print(f"M: truck TargetSpeed={args.truck_move_speed:g}")
    print(f"Z: pedestrian speed={args.pedestrian_z_speed:g}")
    print(f"X: pedestrian speed={args.pedestrian_stop_speed:g}")
    print(
        f"Drone speed={args.flight_speed:g} m/s, "
        f"accel={args.flight_acceleration:g} m/s^2"
    )
    print(
        f"Drone yaw={args.yaw_speed:g} deg/s, "
        f"yaw accel={args.yaw_acceleration:g} deg/s^2"
    )
    print("L: land")
    print("Q: quit")
    if not args.no_live_ned:
        print(f"Live NED: printing every {args.live_ned_interval_sec:g}s")
    print("-------------------------------------")

    keep_running = True
    last_live_ned_at = 0.0
    n_was_pressed = False
    m_was_pressed = False
    z_was_pressed = False
    x_was_pressed = False
    current_vx = 0.0
    current_vy = 0.0
    current_vz = 0.0
    current_yaw_rate = 0.0
    last_control_at = time.monotonic()
    last_pedestrian_update_at = 0.0
    last_pedestrian_error = None

    while keep_running:
        now = time.monotonic()
        dt = min(max(now - last_control_at, 0.0), 0.25)
        last_control_at = now

        if not args.no_live_ned:
            last_live_ned_at = print_live_ned(
                drone,
                last_live_ned_at,
                args.live_ned_interval_sec,
            )

        if now - last_pedestrian_update_at >= args.pedestrian_position_interval_sec:
            last_pedestrian_update_at = now
            pedestrian_position, last_pedestrian_error = update_pedestrian_tracking(
                world,
                args,
                pedestrian_display,
                last_pedestrian_error,
            )
            if pedestrian_position is not None and not args.no_pedestrian_position:
                print(
                    f"[PEDESTRIAN NED] x={pedestrian_position[0]:8.2f}  "
                    f"y={pedestrian_position[1]:8.2f}  "
                    f"z={pedestrian_position[2]:8.2f}",
                    flush=True,
                )

        target_vx, target_vy, target_vz, target_yaw_rate = 0.0, 0.0, 0.0, 0.0

        if keyboard_module.is_pressed("w"):
            target_vx = args.flight_speed
        elif keyboard_module.is_pressed("s"):
            target_vx = -args.flight_speed

        if keyboard_module.is_pressed("a"):
            target_vy = -args.flight_speed
        elif keyboard_module.is_pressed("d"):
            target_vy = args.flight_speed

        if keyboard_module.is_pressed("up"):
            target_vz = -args.flight_speed
        elif keyboard_module.is_pressed("down"):
            target_vz = args.flight_speed

        if keyboard_module.is_pressed("left"):
            target_yaw_rate = -args.yaw_speed
        elif keyboard_module.is_pressed("right"):
            target_yaw_rate = args.yaw_speed

        if args.flight_acceleration <= 0.0:
            current_vx, current_vy, current_vz = target_vx, target_vy, target_vz
        else:
            velocity_step = args.flight_acceleration * dt
            current_vx = move_toward(current_vx, target_vx, velocity_step)
            current_vy = move_toward(current_vy, target_vy, velocity_step)
            current_vz = move_toward(current_vz, target_vz, velocity_step)

        if args.yaw_acceleration <= 0.0:
            current_yaw_rate = target_yaw_rate
        else:
            yaw_step = args.yaw_acceleration * dt
            current_yaw_rate = move_toward(
                current_yaw_rate,
                target_yaw_rate,
                yaw_step,
            )

        n_is_pressed = keyboard_module.is_pressed("n")
        if n_is_pressed and not n_was_pressed:
            set_truck_speed(world, args.truck, args.truck_stop_speed)
        n_was_pressed = n_is_pressed

        m_is_pressed = keyboard_module.is_pressed("m")
        if m_is_pressed and not m_was_pressed:
            set_truck_speed(world, args.truck, args.truck_move_speed)
        m_was_pressed = m_is_pressed

        z_is_pressed = keyboard_module.is_pressed("z")
        if z_is_pressed and not z_was_pressed:
            set_pedestrian_speed(
                world,
                args.pedestrian,
                args.pedestrian_z_speed,
            )
        z_was_pressed = z_is_pressed

        x_is_pressed = keyboard_module.is_pressed("x")
        if x_is_pressed and not x_was_pressed:
            set_pedestrian_speed(
                world,
                args.pedestrian,
                args.pedestrian_stop_speed,
            )
        x_was_pressed = x_is_pressed

        if keyboard_module.is_pressed("l"):
            await land(drone)
            if not args.no_live_ned:
                last_live_ned_at = print_live_ned(
                    drone,
                    last_live_ned_at,
                    args.live_ned_interval_sec,
                    force=True,
                )
            keep_running = False

        if keyboard_module.is_pressed("q"):
            if not args.no_live_ned:
                last_live_ned_at = print_live_ned(
                    drone,
                    last_live_ned_at,
                    args.live_ned_interval_sec,
                    force=True,
                )
            keep_running = False

        if current_vx != 0.0 or current_vy != 0.0 or current_vz != 0.0:
            await drone.move_by_velocity_body_frame_async(
                current_vx,
                current_vy,
                current_vz,
                args.command_duration_sec,
            )
        if current_yaw_rate != 0.0:
            await drone.rotate_by_yaw_rate_async(
                current_yaw_rate,
                args.command_duration_sec,
            )

        await asyncio.sleep(0.01)


async def main():
    args = build_parser().parse_args()
    if args.start is not None:
        args.start = parse_start(args.start)

    temp_scene_dir = None
    image_display = None
    camera_topics = []
    drone = None

    client = projectairsim.ProjectAirSimClient(
        address=args.address,
        port_topics=args.topicsport,
        port_services=args.servicesport,
    )

    try:
        temp_scene_dir, scene_name, sim_config_path = make_runtime_scene_config(args)

        client.connect()
        world = World(
            client=client,
            scene_config_name=scene_name,
            sim_config_path=sim_config_path,
        )
        drone = Drone(client, world, args.drone_name)

        resolved_truck = resolve_truck_actor(world, args.truck)
        if resolved_truck != args.truck:
            print(f"Resolved truck '{args.truck}' to actor '{resolved_truck}'.")
            args.truck = resolved_truck
        resolved_pedestrian = resolve_pedestrian_actor(world, args.pedestrian)
        if resolved_pedestrian != args.pedestrian:
            print(
                f"Resolved pedestrian '{args.pedestrian}' "
                f"to actor '{resolved_pedestrian}'."
            )
            args.pedestrian = resolved_pedestrian
        print(f"Controlling truck actor: {args.truck}")
        print(f"Controlling pedestrian actor: {args.pedestrian}")

        if args.start is not None:
            print(f"Spawning {args.drone_name} at NED {args.start}...")
            drone.set_pose(make_pose_ned(args.start), reset_kinematics=True)
            time.sleep(1)
            if not args.no_live_ned:
                print_live_ned(
                    drone,
                    last_print_at=0.0,
                    interval_sec=args.live_ned_interval_sec,
                    force=True,
                )

        if args.speed is not None:
            set_truck_speed(world, args.truck, args.speed)
        if args.pedestrian_speed is not None:
            set_pedestrian_speed(world, args.pedestrian, args.pedestrian_speed)

        if not args.no_camera:
            image_display, camera_topics = start_front_rgb_display(
                client,
                drone,
                args,
                scene_name,
                sim_config_path,
            )

        await run_keyboard_control(drone, world, args, image_display)
        return 0

    except Exception as exc:
        print(f"An error occurred: {exc}")
        return 1

    finally:
        for camera_topic in camera_topics:
            try:
                client.unsubscribe(camera_topic)
            except Exception:
                pass
        if image_display:
            image_display.stop()
        if drone:
            try:
                drone.disarm()
                drone.disable_api_control()
            except Exception:
                pass
        client.disconnect()
        if temp_scene_dir:
            temp_scene_dir.cleanup()
        print("Cleaned up and disconnected.")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
