"""
Offshore wind farm PX4 demo.

Run this while the OffShoreWindFarm Unreal map is playing. The script loads a
PX4 scene, spawns Drone1 at OFFSHORE_ROUTE[0], and shows FPV with a
chase-camera picture-in-picture. E/R teleport between the route points. Use
--mode-flight auto-flight to teleport to Origin and fly OFFSHORE_ROUTE exactly
as listed. Add --video to record the preview window to client/python/halcyon_demo/video.
Add --wind to apply spatially varying WRF wind and show it in the FPV telemetry.

Controls:
  w/s/a/d  manual forward/back/left/right in manual-direct or manual-px4
  up/down  manual altitude up/down in manual-direct or manual-px4
  left/right  manual yaw left/right in manual-direct or manual-px4
  k/l  increase/decrease manual speed
  e  teleport to next route point
  r  teleport to previous route point
  t  enter custom teleport as x,y,z,angle
  q/esc  stop the route and exit
"""

import argparse
import asyncio
from contextlib import suppress
import math
import queue
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Sequence, Tuple

import commentjson

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.drone import YawControlMode
from projectairsim.types import ImageType, Pose, Quaternion, Vector3
from projectairsim.utils import projectairsim_log, unpack_image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
VIDEO_DIR = SCRIPT_DIR / "video"
DEFAULT_WRF_FILE = REPO_ROOT / "wind_data" / "file.nc"
EXAMPLE_SCRIPTS_DIR = SCRIPT_DIR.parent / "example_user_scripts"
if str(EXAMPLE_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SCRIPTS_DIR))

from px4_astar_autopilot import (  # noqa: E402
    arm_with_retry,
    await_drone_task,
    brake_to_stop_by_velocity,
    clamp,
    distance_between,
    format_vector3,
    get_pose_position_ned,
    get_pose_yaw_ned,
    heading_deg_360,
    limit_vector_delta,
    load_keyboard_module,
    move_scalar_toward,
    request_px4_control,
    sparsify_path,
    wait_for_px4_ready,
    wrap_angle_rad,
)

from route_replan_static import (  # noqa: E402
    create_static_route_scan,
    find_static_route_obstacle,
    is_route_corridor_clear,
)

from wrf_wind import WRFWindField, WRFWindSample  # noqa: E402


# Old route 
# OFFSHORE_ROUTE = [
#     ("Origin", [0.0, 148.0, -2.0], 270.0),
#     ("Waypoint 1", [1.28, -14.66, -115.16], 127.6),
#     ("Waypoint 2", [893.69, 7.89, -112.77], 324.3),
#     ("Waypoint 3", [899.44, -892.28, -12.28], 279.5),
#     ("Waypoint 4", [-3.35, -907.1, -114.81], 55.2),
#     ("Waypoint 5", [-894.61, -901.37, -114.0], 159.9),
#     ("Waypoint 6", [-899.67, -5.87, -7.82], 111.0),
#     ("Return Origin", [0.0, 148.0, -2.0], 90.0),
# ]
OFFSHORE_ROUTE = [
    ("Origin", [0.0, 148.0, -2.0], 270.0),
    ("Around waypoint 1", [12.85,-11.4,-115], 127.6),
    ("Waypoint 1", [1.28, -14.66, -115.16], 127.6),
    ("Around waypoint 2", [881.6,17,-113], 324.3),
    ("Waypoint 2", [893.69, 7.89, -112.77], 324.3),
    ("Around waypoint 3", [895,-870,-20], 279.5),
    ("Waypoint 3", [899.44, -892.28, -12.28], 279.5),
    ("Around waypoint 4", [21,-908,-111], 55.2),
    ("Waypoint 4", [6.35, -910.1, -117.81], 55.2),
    ("Around waypoint 5", [-870, -901.37, -20], 159.9),
    ("Waypoint 5", [-894.61, -901.37, -114.0], 159.9),
    ("Around waypoint 6", [-890, -11, -12.82], 111.0),
    ("Waypoint 6", [-899.67, -5.87, -7.82], 111.0),
    ("Return Origin", [0.0, 148.0, -2.0], 90.0),
]

FULL_OFFSHORE_ROUTE = [
    ("Origin", [0.0, 148.0, -2.0], 270.0),
    ("Mission Start", [0.0, 130.0, -2.0], 270.0),
    ("Around waypoint 1", [12.85,-11.4,-115], 127.6),
    ("Waypoint 1", [1.28, -14.66, -115.16], 127.6),
    ("Around waypoint 2", [881.6,17,-113], 324.3),
    ("Waypoint 2", [893.69, 7.89, -112.77], 324.3),
    ("Around waypoint 3", [895,-870,-20], 279.5),
    ("Waypoint 3", [899.44, -892.28, -12.28], 279.5),
    ("Around waypoint 4", [21, -908.0, -111.0], 90.0),
    ("Waypoint 4", [6.35, -910.1, -117.81], 55.2),
    ("Around waypoint 5", [-880, -895, -16], 180.0),
    ("Waypoint 5", [-892, -902, -12.0], 164),
    ("Around waypoint 6", [-890, -11, -12.82], 111.0),
    ("Waypoint 6", [-899.67, -5.87, -7.82], 111.0),
    ("Return Origin", [0.0, 148.0, -2.0], 90.0),
]
# OFFSHORE_ROUTE = [
#     ("Origin", [-30.7, -905.2, -50.4], 178.4),
#     ("Mission Start", [-40.7, -909.2, -112.4], 178.4),
#     ("Around waypoint 5", [-880, -895, -16], 180.0),
#     ("Waypoint 5", [-892, -902, -12.0], 164),
#     ("Around waypoint 6", [-890, -11, -12.82], 111.0),
#     ("Waypoint 6", [-899.67, -5.87, -7.82], 111.0),
#     ("Return Origin", [0.0, 148.0, -2.0], 90.0),
# ]



TELEPORT_WARN_ERROR_M = 5.0
FPV_MAX_CAMERA_OFFSET_M = 8.0
CHASE_MAX_CAMERA_OFFSET_M = 13.0
BATTERY_START_LABEL = "Mission Start"
BATTERY_SECONDS_PER_PERCENT = 0.4 * 60.0
COVERAGE_UNFEASIBLE_HOLD_SEC = 2.0
INSPECTION_TARGET_OBJECT = "Blade1_Object1"
INSPECTION_NORMAL_OBJECT = "Blade1_Normal1"
INSPECTION_ROOT_OBJECT = "Blade1_Root"
INSPECTION_TIP_OBJECT = "Blade1_Tip"
INSPECTION_DEFAULT_RADIUS_M = 5.0
INSPECTION_DEFAULT_PASS_THRESHOLD = 0.80
INSPECTION_DEFAULT_WEIGHTS = (0.20, 0.25, 0.30, 0.25)
INSPECTION_STABILITY_WINDOW_SEC = 1.0
INSPECTION_STABILITY_MAX_JITTER_NORM = 0.035
INSPECTION_SPHERE_BASE_DIAMETER_M = 1.0
INSPECTION_REGION_WIDTH_M = 0.5
INSPECTION_REGION_HEIGHT_M = 0.5
INSPECTION_GOOD_FOOTPRINT_PX = 48.0
INSPECTION_FRAME_COMFORT_MARGIN_FRACTION = 0.12
INSPECTION_SHARPNESS_MIN_VARIANCE = 40.0
INSPECTION_SHARPNESS_GOOD_VARIANCE = 180.0
INSPECTION_VISUAL_DEBUG_INTERVAL_SEC = 1.0
WARNED_UNAVAILABLE = set()


@dataclass(frozen=True)
class RoutePoint:
    label: str
    position: List[float]
    yaw_deg: float


def route_to_ned(position: Sequence[float]) -> List[float]:
    return [float(position[0]), float(position[1]), float(position[2])]


def ned_to_route(position_ned: Sequence[float]) -> List[float]:
    return [float(position_ned[0]), float(position_ned[1]), float(position_ned[2])]


def interpolate_point(
    start: Sequence[float],
    end: Sequence[float],
    fraction: float,
) -> List[float]:
    fraction = clamp(float(fraction), 0.0, 1.0)
    return [
        float(start[index]) + (float(end[index]) - float(start[index])) * fraction
        for index in range(3)
    ]


def segment_lookahead_point(
    start: Sequence[float],
    end: Sequence[float],
    position: Sequence[float],
    lookahead_m: float,
) -> List[float]:
    segment = [float(end[index]) - float(start[index]) for index in range(3)]
    segment_length_sq = sum(component * component for component in segment)
    if segment_length_sq <= 1e-9 or lookahead_m <= 0.0:
        return [float(end[0]), float(end[1]), float(end[2])]

    projection_fraction = clamp(segment_projection_fraction(start, end, position), 0.0, 1.0)
    segment_length_m = math.sqrt(segment_length_sq)
    lookahead_fraction = lookahead_m / max(segment_length_m, 1e-6)
    return interpolate_point(start, end, projection_fraction + lookahead_fraction)


def segment_projection_fraction(
    start: Sequence[float],
    end: Sequence[float],
    position: Sequence[float],
) -> float:
    segment = [float(end[index]) - float(start[index]) for index in range(3)]
    segment_length_sq = sum(component * component for component in segment)
    if segment_length_sq <= 1e-9:
        return 0.0

    offset = [float(position[index]) - float(start[index]) for index in range(3)]
    return sum(offset[index] * segment[index] for index in range(3)) / segment_length_sq


def closest_point_on_segment(
    start: Sequence[float],
    end: Sequence[float],
    position: Sequence[float],
) -> List[float]:
    if distance_between(start, end) <= 1e-9:
        return [float(start[0]), float(start[1]), float(start[2])]

    fraction = clamp(segment_projection_fraction(start, end, position), 0.0, 1.0)
    return interpolate_point(start, end, fraction)


def clamp_leg_altitude(
    point: Sequence[float],
    start: Sequence[float],
    target: Sequence[float],
    margin_m: float,
) -> List[float]:
    margin_m = max(0.0, float(margin_m))
    min_z = min(float(start[2]), float(target[2])) - margin_m
    max_z = max(float(start[2]), float(target[2])) + margin_m
    return [float(point[0]), float(point[1]), clamp(float(point[2]), min_z, max_z)]


def is_named_waypoint(point: RoutePoint) -> bool:
    return not point.label.startswith("Bypass ")


def append_unique_point(
    path: List[List[float]],
    point: Sequence[float],
    tolerance_m: float = 1e-3,
):
    candidate = [float(point[0]), float(point[1]), float(point[2])]
    if not path or distance_between(path[-1], candidate) > tolerance_m:
        path.append(candidate)


def segment_yaw_deg(
    start: Sequence[float],
    end: Sequence[float],
    fallback_yaw_deg: float,
) -> float:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if math.hypot(dx, dy) <= 1e-6:
        return float(fallback_yaw_deg) % 360.0
    return math.degrees(math.atan2(dy, dx)) % 360.0


def route_points_from_path(
    path: Sequence[Sequence[float]],
    original_route: Sequence[RoutePoint],
) -> List[RoutePoint]:
    route = []
    bypass_count = 0
    for index, point in enumerate(path):
        position = [float(point[0]), float(point[1]), float(point[2])]
        matched = None
        for original in original_route:
            if distance_between(position, original.position) <= 0.5:
                matched = original
                break

        if matched is not None:
            label = matched.label
            yaw_deg = matched.yaw_deg
        else:
            bypass_count += 1
            label = f"Bypass {bypass_count:03d}"
            if index + 1 < len(path):
                yaw_deg = segment_yaw_deg(position, path[index + 1], original_route[0].yaw_deg)
            elif index > 0:
                yaw_deg = segment_yaw_deg(path[index - 1], position, original_route[-1].yaw_deg)
            else:
                yaw_deg = original_route[0].yaw_deg

        route.append(RoutePoint(label, position, yaw_deg))
    return route


def vector_or_none(values: Optional[Sequence[float]]) -> str:
    if values is None:
        return "n/a"
    return f"{values[0]:8.2f} {values[1]:8.2f} {values[2]:8.2f}"


def compact_vector(values: Optional[Sequence[float]]) -> str:
    if values is None:
        return "n/a"
    return f"[{values[0]:.1f}, {values[1]:.1f}, {values[2]:.1f}]"


def finite_vector(values: Optional[Sequence[float]], length: int = 3) -> Optional[List[float]]:
    if values is None:
        return None
    try:
        result = [float(values[index]) for index in range(length)]
    except (IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in result):
        return None
    return result


def rotation_yaw_deg(rotation) -> Optional[float]:
    if not rotation:
        return None
    try:
        w = float(rotation["w"])
        x = float(rotation["x"])
        y = float(rotation["y"])
        z = float(rotation["z"])
    except (KeyError, TypeError, ValueError):
        return None
    yaw_rad = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return heading_deg_360(yaw_rad)


def extract_pose_position(pose_or_kinematics) -> Optional[List[float]]:
    if not isinstance(pose_or_kinematics, dict):
        return None
    pose = pose_or_kinematics.get("pose", pose_or_kinematics)
    if not isinstance(pose, dict):
        return None
    position = (
        pose.get("position")
        or pose.get("translation")
        or pose.get("Position")
        or pose.get("Translation")
    )
    if not isinstance(position, dict):
        return None
    try:
        result = [
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in result):
        return None
    return result


def extract_pose_yaw_deg(pose_or_kinematics) -> Optional[float]:
    if not isinstance(pose_or_kinematics, dict):
        return None
    pose = pose_or_kinematics.get("pose", pose_or_kinematics)
    if not isinstance(pose, dict):
        return None
    return rotation_yaw_deg(pose.get("orientation") or pose.get("rotation"))


def vector_length(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def vector_subtract(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [float(a[index]) - float(b[index]) for index in range(3)]


def vector_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def vector_cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def normalized_vector(values: Sequence[float]) -> Optional[List[float]]:
    length = vector_length(values)
    if length <= 1e-9:
        return None
    return [float(value) / length for value in values]


def angle_between_vectors_deg(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    unit_a = normalized_vector(a)
    unit_b = normalized_vector(b)
    if unit_a is None or unit_b is None:
        return None
    dot = clamp(vector_dot(unit_a, unit_b), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def span_percent_along_blade(
    target: Sequence[float],
    root: Sequence[float],
    tip: Sequence[float],
) -> Optional[float]:
    root_to_tip = vector_subtract(tip, root)
    length_sq = vector_dot(root_to_tip, root_to_tip)
    if length_sq <= 1e-9:
        return None
    root_to_target = vector_subtract(target, root)
    fraction = clamp(vector_dot(root_to_target, root_to_tip) / length_sq, 0.0, 1.0)
    return 100.0 * fraction


def sphere_size_from_scale(
    scale: Optional[Sequence[float]],
) -> Tuple[Optional[float], Optional[float]]:
    scale_values = finite_vector(scale)
    if scale_values is None:
        return None, None
    diameter_m = INSPECTION_SPHERE_BASE_DIAMETER_M * (
        sum(abs(value) for value in scale_values) / len(scale_values)
    )
    if diameter_m <= 0.0:
        return None, None
    surface_area_m2 = math.pi * diameter_m * diameter_m
    return diameter_m, surface_area_m2


def finite_quaternion_from_mapping(value) -> Optional[List[float]]:
    if not isinstance(value, dict):
        return None
    try:
        quaternion = [
            float(value["w"]),
            float(value["x"]),
            float(value["y"]),
            float(value["z"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in quaternion):
        return None
    return quaternion


def normalized_quaternion(quaternion: Sequence[float]) -> Optional[List[float]]:
    try:
        result = [float(quaternion[index]) for index in range(4)]
    except (IndexError, TypeError, ValueError):
        return None
    norm = math.sqrt(sum(component * component for component in result))
    if norm <= 1e-9 or not math.isfinite(norm):
        return None
    return [component / norm for component in result]


def quaternion_conjugate(quaternion: Sequence[float]) -> List[float]:
    return [
        float(quaternion[0]),
        -float(quaternion[1]),
        -float(quaternion[2]),
        -float(quaternion[3]),
    ]


def quaternion_rotate_vector(
    quaternion: Sequence[float],
    vector: Sequence[float],
) -> List[float]:
    q = normalized_quaternion(quaternion)
    if q is None:
        return [float(vector[0]), float(vector[1]), float(vector[2])]
    w = q[0]
    q_vec = [q[1], q[2], q[3]]
    v = [float(vector[0]), float(vector[1]), float(vector[2])]
    first_cross = vector_cross(q_vec, v)
    second_cross = vector_cross(q_vec, first_cross)
    return [
        v[index] + 2.0 * (w * first_cross[index] + second_cross[index])
        for index in range(3)
    ]


def project_ned_point_to_camera_pixel(
    target_position: Sequence[float],
    camera_position: Sequence[float],
    camera_rotation: Sequence[float],
    horizontal_fov_degrees: float,
    image_width: int,
    image_height: int,
) -> Optional[Dict]:
    target = finite_vector(target_position)
    camera = finite_vector(camera_position)
    rotation = normalized_quaternion(camera_rotation)
    if target is None or camera is None or rotation is None:
        return None

    width = max(1, int(image_width))
    height = max(1, int(image_height))
    offset_world = vector_subtract(target, camera)
    offset_camera = quaternion_rotate_vector(quaternion_conjugate(rotation), offset_world)
    depth_m = float(offset_camera[0])
    if depth_m <= 1e-6:
        return {
            "pixel": None,
            "depth_m": depth_m,
            "in_front": False,
            "focal_px": None,
        }

    fov_rad = math.radians(clamp(float(horizontal_fov_degrees), 1.0, 179.0))
    focal_px = width / (2.0 * math.tan(fov_rad / 2.0))
    pixel_x = (width - 1) * 0.5 + (float(offset_camera[1]) / depth_m) * focal_px
    pixel_y = (height - 1) * 0.5 + (float(offset_camera[2]) / depth_m) * focal_px
    return {
        "pixel": (pixel_x, pixel_y),
        "depth_m": depth_m,
        "in_front": True,
        "focal_px": focal_px,
    }


def parse_inspection_confidence_weights(value: str) -> Tuple[float, float, float, float]:
    parts = value.replace(",", " ").split()
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Expected four weights: distance,angle,visual,stability"
        )
    try:
        weights = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Inspection confidence weights must be numeric") from exc
    if any(weight < 0.0 for weight in weights):
        raise argparse.ArgumentTypeError("Inspection confidence weights must be non-negative")
    total = sum(weights)
    if total <= 1e-9:
        raise argparse.ArgumentTypeError("At least one inspection confidence weight must be positive")
    return tuple(weight / total for weight in weights)


def get_ground_truth_velocity_ned(drone: Drone) -> List[float]:
    twist = drone.get_ground_truth_kinematics().get("twist", {})
    linear = twist.get("linear", {})
    return [
        float(linear.get("x", 0.0)),
        float(linear.get("y", 0.0)),
        float(linear.get("z", 0.0)),
    ]


def image_pose_position(image) -> Optional[List[float]]:
    if not isinstance(image, dict):
        return None
    keys = ("pos_x", "pos_y", "pos_z")
    if not all(key in image for key in keys):
        return None
    try:
        position = [float(image["pos_x"]), float(image["pos_y"]), float(image["pos_z"])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in position):
        return None
    return position


def image_pose_rotation(image) -> Optional[List[float]]:
    if not isinstance(image, dict):
        return None
    keys = ("rot_w", "rot_x", "rot_y", "rot_z")
    if not all(key in image for key in keys):
        return None
    return finite_quaternion_from_mapping(
        {
            "w": image["rot_w"],
            "x": image["rot_x"],
            "y": image["rot_y"],
            "z": image["rot_z"],
        }
    )


def image_resolution(image) -> Optional[Tuple[int, int]]:
    if not isinstance(image, dict):
        return None
    try:
        width = int(image["width"])
        height = int(image["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def image_pose_yaw_deg(image) -> Optional[float]:
    if not isinstance(image, dict):
        return None
    keys = ("rot_w", "rot_x", "rot_y", "rot_z")
    if not all(key in image for key in keys):
        return None
    return rotation_yaw_deg(
        {
            "w": image["rot_w"],
            "x": image["rot_x"],
            "y": image["rot_y"],
            "z": image["rot_z"],
        }
    )


def geo_summary(value) -> str:
    if not isinstance(value, dict):
        return "n/a"
    geo = value.get("geo_point") or value.get("gnss", {}).get("geo_point") or value
    if not isinstance(geo, dict):
        return "n/a"
    lat = geo.get("latitude", geo.get("lat"))
    lon = geo.get("longitude", geo.get("lon"))
    alt = geo.get("altitude", geo.get("alt"))
    if lat is None or lon is None or alt is None:
        return "n/a"
    try:
        return f"lat={float(lat):.7f} lon={float(lon):.7f} alt={float(alt):.2f}"
    except (TypeError, ValueError):
        return "n/a"


class DemoState:
    def __init__(self, route: Sequence[RoutePoint], battery_start_percent: float = 100.0):
        self.route = list(route)
        self.battery_start_percent = clamp(float(battery_start_percent), 0.0, 100.0)
        self.current_index = 0
        self.target_index = 1
        self.position = list(self.route[0].position)
        self.heading_deg = self.route[0].yaw_deg
        self.last_requested_label = "Scene origin"
        self.last_requested_position = list(self.route[0].position)
        self.last_requested_yaw_deg = self.route[0].yaw_deg
        self.diagnostics = {}
        self.fpv_camera_position = None
        self.fpv_camera_rotation = None
        self.fpv_camera_resolution = None
        self.inspection = {
            "active": False,
            "status": "outside",
        }
        self.mode = "teleport"
        self.mode_detail = "manual teleport mode"
        self.auto_flight_running = False
        self.camera_epoch = 0
        self.battery_started_at = None
        self.battery_percent = self.battery_start_percent
        self.coverage_unfeasible = False
        self.coverage_unfeasible_logged = False
        self.wind_enabled = False
        self.wind_sample = None
        self.wind_status = ""
        self.teleport_requests = queue.SimpleQueue()
        self.manual_teleport_requests = queue.SimpleQueue()
        self.stop_requested = False
        self.lock = Lock()

    def snapshot(self):
        with self.lock:
            self.battery_percent = self._battery_percent_locked()
            return {
                "current_index": self.current_index,
                "target_index": self.target_index,
                "position": list(self.position),
                "heading_deg": self.heading_deg,
                "last_requested_label": self.last_requested_label,
                "last_requested_position": list(self.last_requested_position),
                "last_requested_yaw_deg": self.last_requested_yaw_deg,
                "diagnostics": dict(self.diagnostics),
                "fpv_camera_position": (
                    list(self.fpv_camera_position)
                    if self.fpv_camera_position is not None
                    else None
                ),
                "fpv_camera_rotation": (
                    list(self.fpv_camera_rotation)
                    if self.fpv_camera_rotation is not None
                    else None
                ),
                "fpv_camera_resolution": (
                    list(self.fpv_camera_resolution)
                    if self.fpv_camera_resolution is not None
                    else None
                ),
                "inspection": dict(self.inspection),
                "mode": self.mode,
                "mode_detail": self.mode_detail,
                "auto_flight_running": self.auto_flight_running,
                "camera_epoch": self.camera_epoch,
                "battery_active": self.battery_started_at is not None,
                "battery_percent": self.battery_percent,
                "coverage_unfeasible": self.coverage_unfeasible,
                "wind_enabled": self.wind_enabled,
                "wind_sample": self.wind_sample,
                "wind_status": self.wind_status,
                "stop_requested": self.stop_requested,
            }

    def _battery_percent_locked(self, now: Optional[float] = None) -> float:
        if self.battery_started_at is None:
            return self.battery_start_percent
        now = time.time() if now is None else now
        elapsed_sec = max(0.0, now - self.battery_started_at)
        depleted_percent = int(elapsed_sec // BATTERY_SECONDS_PER_PERCENT)
        return float(clamp(self.battery_start_percent - depleted_percent, 0.0, 100.0))

    def _start_battery_if_trigger_locked(self, index: int) -> bool:
        if self.battery_started_at is not None or index < 0 or index >= len(self.route):
            return False
        if self.route[index].label.strip().casefold() != BATTERY_START_LABEL.casefold():
            return False
        self.battery_started_at = time.time()
        self.battery_percent = self.battery_start_percent
        return True

    def check_battery_depleted(self) -> bool:
        with self.lock:
            self.battery_percent = self._battery_percent_locked()
            if (
                self.battery_started_at is not None
                and self.battery_percent <= 0.0
                and not self.coverage_unfeasible
            ):
                self.coverage_unfeasible = True
                self.mode = "Coverage Unfeasible"
                self.mode_detail = "battery depleted"
                return True
            return False

    def take_coverage_unfeasible_log_event(self) -> bool:
        with self.lock:
            if self.coverage_unfeasible and not self.coverage_unfeasible_logged:
                self.coverage_unfeasible_logged = True
                return True
            return False

    def update_pose(self, position: Sequence[float], heading_deg: float):
        with self.lock:
            self.position = [float(position[0]), float(position[1]), float(position[2])]
            self.heading_deg = float(heading_deg) % 360.0

    def update_requested(
        self,
        label: str,
        position: Sequence[float],
        yaw_deg: float,
    ):
        with self.lock:
            self.last_requested_label = label
            self.last_requested_position = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ]
            self.last_requested_yaw_deg = float(yaw_deg) % 360.0

    def update_diagnostic(
        self,
        name: str,
        position: Optional[Sequence[float]] = None,
        yaw_deg: Optional[float] = None,
        extra: str = "",
    ):
        with self.lock:
            self.diagnostics[name] = {
                "position": (
                    [float(position[0]), float(position[1]), float(position[2])]
                    if position is not None
                    else None
                ),
                "yaw_deg": float(yaw_deg) % 360.0 if yaw_deg is not None else None,
                "extra": extra,
                "time": time.time(),
            }

    def set_mode(self, mode: str, detail: str = ""):
        with self.lock:
            self.mode = mode
            self.mode_detail = detail

    def set_auto_flight_running(self, running: bool):
        with self.lock:
            self.auto_flight_running = bool(running)

    def set_wind_enabled(self, enabled: bool, status: str = ""):
        with self.lock:
            self.wind_enabled = bool(enabled)
            self.wind_status = status
            if not enabled:
                self.wind_sample = None

    def update_wind_sample(self, sample: WRFWindSample, status: str = ""):
        with self.lock:
            self.wind_enabled = True
            self.wind_sample = sample
            self.wind_status = status

    def update_image_diagnostic(self, name: str, image):
        position = image_pose_position(image)
        yaw_deg = image_pose_yaw_deg(image)
        extra = ""
        if isinstance(image, dict):
            width = image.get("width")
            height = image.get("height")
            if width is not None and height is not None:
                extra = f"{width}x{height}"
        self.update_diagnostic(name, position, yaw_deg, extra)

    def update_fpv_camera_position(self, position: Optional[Sequence[float]]):
        if position is None:
            return
        with self.lock:
            self.fpv_camera_position = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ]

    def update_fpv_camera_pose(
        self,
        position: Optional[Sequence[float]],
        rotation: Optional[Sequence[float]],
        resolution: Optional[Sequence[int]] = None,
    ):
        with self.lock:
            if position is not None:
                self.fpv_camera_position = [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ]
            if rotation is not None:
                self.fpv_camera_rotation = [
                    float(rotation[0]),
                    float(rotation[1]),
                    float(rotation[2]),
                    float(rotation[3]),
                ]
            if resolution is not None:
                self.fpv_camera_resolution = [
                    int(resolution[0]),
                    int(resolution[1]),
                ]

    def update_inspection_geometry(self, inspection: Dict):
        with self.lock:
            self.inspection = dict(inspection)

    def clear_inspection_geometry(self, status: str = "outside"):
        with self.lock:
            self.inspection = {
                "active": False,
                "status": status,
                "time": time.time(),
            }

    def bump_camera_epoch(self):
        with self.lock:
            self.camera_epoch += 1

    def mark_reached(self, index: int):
        with self.lock:
            self.current_index = index
            self.target_index = min(index + 1, len(self.route) - 1)
            return self._start_battery_if_trigger_locked(index)

    def set_route_index(self, index: int):
        with self.lock:
            self.current_index = index
            self.target_index = min(index + 1, len(self.route) - 1)

    def queue_teleport(self, delta: int):
        self.teleport_requests.put(delta)

    def queue_manual_teleport(self):
        self.manual_teleport_requests.put(True)

    def request_stop(self):
        with self.lock:
            self.stop_requested = True


class OffshorePreview:
    def __init__(
        self,
        state: DemoState,
        window_name: str,
        width: int,
        height: int,
        pip_scale: float,
        max_fps: float,
        record_video: bool = False,
        video_dir: Optional[Path] = None,
        video_fps: float = 0.0,
        coordinate_diagnostics: bool = False,
        front_camera_stabilized: bool = False,
        route_overlay: Optional[Sequence[RoutePoint]] = None,
        camera_fov_degrees: float = 90.0,
        inspection_region_width_m: float = INSPECTION_REGION_WIDTH_M,
        inspection_region_height_m: float = INSPECTION_REGION_HEIGHT_M,
        inspection_good_footprint_px: float = INSPECTION_GOOD_FOOTPRINT_PX,
        inspection_frame_margin_fraction: float = INSPECTION_FRAME_COMFORT_MARGIN_FRACTION,
        inspection_sharpness_min_variance: float = INSPECTION_SHARPNESS_MIN_VARIANCE,
        inspection_sharpness_good_variance: float = INSPECTION_SHARPNESS_GOOD_VARIANCE,
        inspection_weights: Sequence[float] = INSPECTION_DEFAULT_WEIGHTS,
        inspection_pass_threshold: float = INSPECTION_DEFAULT_PASS_THRESHOLD,
    ):
        self.state = state
        self.route_overlay = list(route_overlay) if route_overlay is not None else list(state.route)
        self.window_name = window_name
        self.width = width
        self.height = height
        self.pip_scale = max(0.1, min(0.5, pip_scale))
        self.max_fps = max(1.0, max_fps)
        self.coordinate_diagnostics = bool(coordinate_diagnostics)
        self.front_camera_stabilized = bool(front_camera_stabilized)
        self.camera_fov_degrees = float(camera_fov_degrees)
        self.inspection_region_width_m = max(0.01, float(inspection_region_width_m))
        self.inspection_region_height_m = max(0.01, float(inspection_region_height_m))
        self.inspection_good_footprint_px = max(1.0, float(inspection_good_footprint_px))
        self.inspection_frame_margin_fraction = clamp(
            float(inspection_frame_margin_fraction),
            0.01,
            0.45,
        )
        self.inspection_sharpness_min_variance = max(
            0.0,
            float(inspection_sharpness_min_variance),
        )
        self.inspection_sharpness_good_variance = max(
            self.inspection_sharpness_min_variance + 1e-6,
            float(inspection_sharpness_good_variance),
        )
        self.inspection_weights = tuple(float(weight) for weight in inspection_weights)
        self.inspection_pass_threshold = clamp(float(inspection_pass_threshold), 0.0, 1.0)
        self.inspection_centroid_history = []
        self.last_visual_debug_at = 0.0
        self.record_video = bool(record_video)
        self.video_dir = Path(video_dir) if video_dir is not None else VIDEO_DIR
        self.video_fps = self.max_fps if video_fps <= 0.0 else max(1.0, video_fps)
        self.video_writer = None
        self.video_path = None
        self.video_frame_count = 0
        self.video_started_at = None
        self.video_last_frame_at = None
        self.video_finalized = False
        self.fpv_images = queue.SimpleQueue()
        self.chase_images = queue.SimpleQueue()
        self.running = False
        self.thread = None
        self.error = None
        self.last_key_at = {}
        self.key_debounce_sec = 0.25
        self.camera_epoch_seen = state.snapshot()["camera_epoch"]
        self.waiting_for_fresh_fpv = False

        xs = [point.position[0] for point in self.route_overlay]
        ys = [point.position[1] for point in self.route_overlay]
        self.map_min_x = min(xs)
        self.map_max_x = max(xs)
        self.map_min_y = min(ys)
        self.map_max_y = max(ys)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = Thread(target=self.display_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if not self.thread or not self.thread.is_alive():
            self.close_video_writer()
        self.thread = None

    def ensure_video_writer(self, cv2):
        if not self.record_video or self.video_writer is not None:
            return

        self.video_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = self.video_dir / f"offshore_demo_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.video_fps,
            (self.width, self.height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open video writer: {self.video_path}")

        self.video_writer = writer
        projectairsim_log().info(
            "Recording preview video to %s at %.1f FPS",
            self.video_path,
            self.video_fps,
        )

    def write_video_frame(self, cv2, frame):
        if not self.record_video:
            return
        self.ensure_video_writer(cv2)
        now = time.monotonic()
        if self.video_started_at is None:
            self.video_started_at = now

        target_frame_count = int((now - self.video_started_at) * self.video_fps) + 1
        while self.video_frame_count < target_frame_count:
            self.video_writer.write(frame)
            self.video_frame_count += 1
        self.video_last_frame_at = now

    def close_video_writer(self):
        if self.video_finalized:
            return
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            duration_sec = 0.0
            if self.video_started_at is not None and self.video_last_frame_at is not None:
                duration_sec = max(0.0, self.video_last_frame_at - self.video_started_at)
            projectairsim_log().info(
                "Saved preview video to %s (%d frames, %.1fs)",
                self.video_path,
                self.video_frame_count,
                duration_sec,
            )
        elif self.record_video and self.video_path is None:
            projectairsim_log().warning("Video requested, but no preview frames were recorded.")
        if self.record_video:
            self.video_finalized = True

    def receive_fpv(self, _, image):
        self.state.update_image_diagnostic("FPV camera msg", image)
        if not self.camera_frame_is_near_drone(
            image,
            FPV_MAX_CAMERA_OFFSET_M,
            "FPV camera msg",
        ):
            return
        self.state.update_fpv_camera_pose(
            image_pose_position(image),
            image_pose_rotation(image),
            image_resolution(image),
        )
        self._push_latest(self.fpv_images, image)

    def receive_chase(self, _, image):
        self.state.update_image_diagnostic("Chase camera msg", image)
        if not self.camera_frame_is_near_drone(
            image,
            CHASE_MAX_CAMERA_OFFSET_M,
            "Chase camera msg",
        ):
            return
        self._push_latest(self.chase_images, image)

    def camera_frame_is_near_drone(
        self,
        image,
        max_offset_m: float,
        diagnostic_name: str,
    ) -> bool:
        camera_position = image_pose_position(image)
        if camera_position is None:
            return True
        drone_position = self.state.snapshot()["position"]
        offset_m = distance_between(camera_position, drone_position)
        if offset_m <= max_offset_m:
            return True
        self.state.update_diagnostic(
            diagnostic_name,
            camera_position,
            image_pose_yaw_deg(image),
            f"STALE offset={offset_m:.1f}m",
        )
        return False

    def _push_latest(self, image_queue, image):
        if not self.running or image is None:
            return
        while not image_queue.empty() and image_queue.qsize() > 2:
            image_queue.get()
        image_queue.put(image)

    def _pop_latest_frame(self, image_queue):
        image = None
        while not image_queue.empty():
            image = image_queue.get()
        if image is None:
            return None
        frame = unpack_image(image)
        if frame is None:
            return None
        return frame.copy()

    @staticmethod
    def drain_queue(image_queue):
        while not image_queue.empty():
            image_queue.get()

    def make_waiting_frame(self, cv2):
        import numpy as np

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.draw_text(
            cv2,
            frame,
            "Waiting for fresh FPV camera frame after teleport",
            (18, 32),
            scale=0.65,
        )
        return frame

    def display_loop(self):
        import cv2

        created = False
        frame_interval_sec = 1.0 / self.max_fps
        next_frame_at = time.monotonic()
        try:
            while self.running:
                snapshot = self.state.snapshot()
                if snapshot["camera_epoch"] != self.camera_epoch_seen:
                    self.drain_queue(self.fpv_images)
                    self.drain_queue(self.chase_images)
                    self.camera_epoch_seen = snapshot["camera_epoch"]
                    self.waiting_for_fresh_fpv = True

                now = time.monotonic()
                if now < next_frame_at:
                    key = cv2.waitKey(max(1, int((next_frame_at - now) * 1000.0)))
                    self.handle_key(key)
                    continue

                fpv = self._pop_latest_frame(self.fpv_images)
                if fpv is None:
                    if self.waiting_for_fresh_fpv:
                        frame = self.make_waiting_frame(cv2)
                        self.draw_route_overlay(cv2, frame)
                        self.draw_status(cv2, frame)
                        if self.coordinate_diagnostics:
                            self.draw_diagnostics(cv2, frame)
                        self.draw_battery(cv2, frame)
                        self.draw_coverage_unfeasible(cv2, frame)
                        if not created:
                            cv2.namedWindow(
                                self.window_name,
                                flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE,
                            )
                            created = True
                        self.write_video_frame(cv2, frame)
                        cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(1)
                    self.handle_key(key)
                    next_frame_at = time.monotonic() + frame_interval_sec
                    continue

                frame = fpv
                self.waiting_for_fresh_fpv = False
                frame = self.ensure_bgr(cv2, frame)
                frame = cv2.resize(frame, (self.width, self.height))
                chase = self._pop_latest_frame(self.chase_images)
                if chase is not None:
                    chase = self.ensure_bgr(cv2, chase)
                    self.draw_chase_pip(cv2, frame, chase)

                self.draw_route_overlay(cv2, frame)
                self.draw_status(cv2, frame)
                if self.coordinate_diagnostics:
                    self.draw_diagnostics(cv2, frame)
                self.draw_battery(cv2, frame)
                self.draw_inspection_confidence(cv2, frame)
                self.draw_coverage_unfeasible(cv2, frame)

                if not created:
                    cv2.namedWindow(
                        self.window_name,
                        flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE,
                    )
                    created = True

                self.write_video_frame(cv2, frame)
                cv2.imshow(self.window_name, frame)
                self.handle_key(cv2.waitKey(1))
                next_frame_at = time.monotonic() + frame_interval_sec
        except Exception as exc:
            self.error = exc
            self.state.request_stop()
        finally:
            self.close_video_writer()
            if created:
                cv2.destroyWindow(self.window_name)

    def handle_key(self, key: int):
        if key < 0:
            return
        key &= 0xFF
        if key == ord("e"):
            if self.accept_key("e"):
                self.state.queue_teleport(1)
        elif key == ord("r"):
            if self.accept_key("r"):
                self.state.queue_teleport(-1)
        elif key in (ord("t"), ord("T")):
            if self.accept_key("t"):
                self.state.queue_manual_teleport()
        elif key in (ord("q"), 27):
            self.state.request_stop()
            self.running = False

    def accept_key(self, key_name: str) -> bool:
        now = time.monotonic()
        last = self.last_key_at.get(key_name, 0.0)
        if now - last < self.key_debounce_sec:
            return False
        self.last_key_at[key_name] = now
        return True

    @staticmethod
    def ensure_bgr(cv2, frame):
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def chase_pip_rect(self, frame) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        pip_w = int(width * self.pip_scale)
        pip_h = int(pip_w * 9 / 16)
        pip_h = min(pip_h, int(height * 0.35))
        x0 = width - pip_w - 16
        y0 = 16
        return x0, y0, pip_w, pip_h

    def draw_chase_pip(self, cv2, frame, chase):
        x0, y0, pip_w, pip_h = self.chase_pip_rect(frame)
        pip = cv2.resize(chase, (pip_w, pip_h))
        frame[y0 : y0 + pip_h, x0 : x0 + pip_w] = pip
        cv2.rectangle(frame, (x0, y0), (x0 + pip_w, y0 + pip_h), (255, 255, 255), 2)
        self.draw_text(cv2, frame, "TPV", (x0 + 8, y0 + 22), scale=0.55)

    def draw_route_overlay(self, cv2, frame):
        height, width = frame.shape[:2]
        origin_x = 18
        origin_y = 46
        map_size = int(clamp(min(width * 0.26, height * 0.32), 165.0, 250.0))
        map_w = map_size
        map_h = map_size
        pad_m = 80.0
        span_x = max(1.0, self.map_max_x - self.map_min_x + pad_m * 2)
        span_y = max(1.0, self.map_max_y - self.map_min_y + pad_m * 2)

        def to_px(position: Sequence[float]) -> Tuple[int, int]:
            x_norm = (position[0] - self.map_min_x + pad_m) / span_x
            y_norm = (position[1] - self.map_min_y + pad_m) / span_y
            px = origin_x + int(x_norm * map_w)
            py = origin_y + map_h - int(y_norm * map_h)
            return px, py

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (origin_x - 10, origin_y - 30),
            (origin_x + map_w + 10, origin_y + map_h + 10),
            (18, 18, 18),
            -1,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)
        cv2.rectangle(
            frame,
            (origin_x - 10, origin_y - 30),
            (origin_x + map_w + 10, origin_y + map_h + 10),
            (235, 235, 235),
            1,
        )
        self.draw_text(cv2, frame, "Route", (origin_x, origin_y - 8), scale=0.43)

        route_pixels = [to_px(point.position) for point in self.route_overlay]
        for start, end in zip(route_pixels, route_pixels[1:]):
            cv2.line(frame, start, end, (255, 0, 0), 2, cv2.LINE_AA)

        snapshot = self.state.snapshot()
        for point in self.route_overlay:
            color = (0, 0, 255)
            radius = max(3, int(map_size * 0.018))
            cv2.circle(frame, to_px(point.position), radius, color, -1, cv2.LINE_AA)

        active_radius = max(4, int(map_size * 0.024))
        current_point = self.state.route[snapshot["current_index"]]
        target_point = self.state.route[snapshot["target_index"]]
        cv2.circle(frame, to_px(current_point.position), active_radius, (0, 210, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, to_px(target_point.position), active_radius, (0, 255, 255), -1, cv2.LINE_AA)

        pos_px = to_px(snapshot["position"])
        cv2.drawMarker(
            frame,
            pos_px,
            (80, 255, 80),
            markerType=cv2.MARKER_TRIANGLE_UP,
            markerSize=max(10, int(map_size * 0.055)),
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    def draw_status(self, cv2, frame):
        snapshot = self.state.snapshot()
        target = self.state.route[snapshot["target_index"]]
        position = snapshot["position"]
        distance = distance_between(position, target.position)
        camera_status = (
            "Front Camera Stabilized"
            if self.front_camera_stabilized
            else "Front Camera not stabilized"
        )
        lines = [
            f"Target: {target.label}  {distance:.1f} m",
            f"NED: x={position[0]:.1f} y={position[1]:.1f} z={position[2]:.1f}",
            f"Heading: {snapshot['heading_deg']:.1f} deg  Mode: {snapshot['mode']}",
            camera_status,
        ]
        if snapshot["wind_enabled"]:
            wind_sample = snapshot["wind_sample"]
            if wind_sample is None:
                status = snapshot["wind_status"] or "initializing"
                lines.append(f"WIND {status}")
            else:
                # Direction TO: 0 deg = North, 90 deg = East, matching the AirSim N/E vector.
                lines.append(
                    f"WIND {wind_sample.horizontal_speed_mps:.1f} m/s @ "
                    f"{wind_sample.direction_to_deg:.0f} deg  "
                    f"VERT {wind_sample.w_up_mps:+.2f} m/s"
                )
        control_hint = "e/r teleport | t custom | q/esc quit"
        if str(snapshot["mode"]).startswith("manual "):
            control_hint = "WASD/arrows fly | K/L speed | e/r/t teleport | q/esc quit"
        lines.extend([snapshot["mode_detail"], control_hint])
        height, width = frame.shape[:2]
        text_scale = 0.38 if width <= 800 or height <= 650 else 0.44
        line_spacing = 17 if text_scale <= 0.38 else 20
        y = height - (line_spacing * len(lines) + 8)
        for line in lines:
            is_wind_line = line.startswith("WIND ")
            if is_wind_line and snapshot.get("wind_sample") is not None:
                arrow_radius = 12 if text_scale <= 0.38 else 14
                self.draw_wind_arrow(cv2, frame, snapshot, (36, y - 6), arrow_radius)
                text_x = 56 if text_scale <= 0.38 else 64
            else:
                text_x = 18
            self.draw_text(cv2, frame, line, (text_x, y), scale=text_scale)
            y += line_spacing

    def draw_wind_arrow(self, cv2, frame, snapshot: Dict, center: Tuple[int, int], radius: int):
        wind_sample = snapshot["wind_sample"]
        if wind_sample is None or wind_sample.horizontal_speed_mps <= 1e-6:
            return

        # Arrow is relative to the drone/camera: up=forward, right=drone right.
        relative_deg = (wind_sample.direction_to_deg - snapshot["heading_deg"]) % 360.0
        relative_rad = math.radians(relative_deg)
        dx = math.sin(relative_rad)
        dy = -math.cos(relative_rad)
        start = (
            int(center[0] - dx * radius * 0.45),
            int(center[1] - dy * radius * 0.45),
        )
        end = (
            int(center[0] + dx * radius * 0.90),
            int(center[1] + dy * radius * 0.90),
        )
        blue = (255, 190, 60)
        cv2.circle(frame, center, radius, blue, 1, cv2.LINE_AA)
        cv2.arrowedLine(frame, start, end, blue, 2, cv2.LINE_AA, tipLength=0.35)

    def score_projected_inspection_target(
        self,
        cv2,
        frame,
        target_position: Sequence[float],
        camera_position: Optional[Sequence[float]],
        camera_rotation: Optional[Sequence[float]],
        camera_resolution: Optional[Sequence[int]],
    ) -> Dict:
        height, width = frame.shape[:2]
        if camera_resolution is None:
            source_width, source_height = width, height
        else:
            source_width = max(1, int(camera_resolution[0]))
            source_height = max(1, int(camera_resolution[1]))
        projection = project_ned_point_to_camera_pixel(
            target_position,
            camera_position,
            camera_rotation,
            self.camera_fov_degrees,
            source_width,
            source_height,
        )
        if projection is None:
            result = {
                "score": 0.0,
                "centroid": None,
                "projected_pixel": None,
                "in_frame_score": 0.0,
                "expected_width_px": 0.0,
                "expected_height_px": 0.0,
                "laplacian_variance": 0.0,
                "sharpness_score": 0.0,
            }
            self.log_visual_debug(result)
            return result

        pixel = projection.get("pixel")
        depth_m = float(projection.get("depth_m", 0.0))
        focal_px = projection.get("focal_px")
        if pixel is None or focal_px is None or depth_m <= 1e-6:
            result = {
                "score": 0.0,
                "centroid": None,
                "projected_pixel": None,
                "in_frame_score": 0.0,
                "expected_width_px": 0.0,
                "expected_height_px": 0.0,
                "laplacian_variance": 0.0,
                "sharpness_score": 0.0,
            }
            self.log_visual_debug(result)
            return result

        source_pixel_x, source_pixel_y = pixel
        scale_x = width / float(source_width)
        scale_y = height / float(source_height)
        pixel_x = source_pixel_x * scale_x
        pixel_y = source_pixel_y * scale_y
        border_distance_px = min(pixel_x, width - 1 - pixel_x, pixel_y, height - 1 - pixel_y)
        comfort_margin_px = max(1.0, min(width, height) * self.inspection_frame_margin_fraction)
        in_frame_score = clamp(border_distance_px / comfort_margin_px, 0.0, 1.0)

        expected_width_px = (
            float(focal_px) * self.inspection_region_width_m / depth_m * scale_x
        )
        expected_height_px = (
            float(focal_px) * self.inspection_region_height_m / depth_m * scale_y
        )
        if in_frame_score <= 0.0:
            apparent_size_score = 0.0
            laplacian_variance = 0.0
            sharpness_score = 0.0
        else:
            apparent_size_score = clamp(
                min(expected_width_px, expected_height_px)
                / self.inspection_good_footprint_px,
                0.0,
                1.0,
            )
            laplacian_variance = self.laplacian_variance_near_pixel(
                cv2,
                frame,
                pixel_x,
                pixel_y,
                expected_width_px,
                expected_height_px,
            )
            sharpness_score = clamp(
                (laplacian_variance - self.inspection_sharpness_min_variance)
                / (
                    self.inspection_sharpness_good_variance
                    - self.inspection_sharpness_min_variance
                ),
                0.0,
                1.0,
            )
        visual_score = clamp(
            0.40 * in_frame_score
            + 0.35 * apparent_size_score
            + 0.25 * sharpness_score,
            0.0,
            1.0,
        )
        centroid = (
            pixel_x / float(width),
            pixel_y / float(height),
        ) if in_frame_score > 0.0 else None
        result = {
            "score": visual_score,
            "centroid": centroid,
            "projected_pixel": (pixel_x, pixel_y),
            "in_frame_score": in_frame_score,
            "expected_width_px": expected_width_px,
            "expected_height_px": expected_height_px,
            "laplacian_variance": laplacian_variance,
            "sharpness_score": sharpness_score,
        }
        self.log_visual_debug(result)
        return result

    def laplacian_variance_near_pixel(
        self,
        cv2,
        frame,
        pixel_x: float,
        pixel_y: float,
        expected_width_px: float,
        expected_height_px: float,
    ) -> float:
        height, width = frame.shape[:2]
        if pixel_x < 0.0 or pixel_x >= width or pixel_y < 0.0 or pixel_y >= height:
            return 0.0

        roi_w = int(
            clamp(
                max(24.0, expected_width_px * 1.5),
                24.0,
                max(24.0, width * 0.25),
            )
        )
        roi_h = int(
            clamp(
                max(24.0, expected_height_px * 1.5),
                24.0,
                max(24.0, height * 0.25),
            )
        )
        x0 = max(0, int(round(pixel_x - roi_w * 0.5)))
        x1 = min(width, int(round(pixel_x + roi_w * 0.5)))
        y0 = max(0, int(round(pixel_y - roi_h * 0.5)))
        y1 = min(height, int(round(pixel_y + roi_h * 0.5)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return 0.0

        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def log_visual_debug(self, visual: Dict):
        now = time.monotonic()
        if now - self.last_visual_debug_at < INSPECTION_VISUAL_DEBUG_INTERVAL_SEC:
            return
        self.last_visual_debug_at = now
        projected_pixel = visual.get("projected_pixel")
        if projected_pixel is None:
            pixel_text = "x=n/a y=n/a"
        else:
            pixel_text = f"x={projected_pixel[0]:.1f} y={projected_pixel[1]:.1f}"
        projectairsim_log().info(
            "visual_debug projected_pixel %s in_frame_score=%.2f "
            "expected_width_px=%.1f expected_height_px=%.1f "
            "laplacian_variance=%.1f sharpness_score=%.2f visual_score=%.2f",
            pixel_text,
            float(visual.get("in_frame_score", 0.0)),
            float(visual.get("expected_width_px", 0.0)),
            float(visual.get("expected_height_px", 0.0)),
            float(visual.get("laplacian_variance", 0.0)),
            float(visual.get("sharpness_score", 0.0)),
            float(visual.get("score", 0.0)),
        )

    def update_inspection_stability(self, centroid: Optional[Tuple[float, float]]) -> Optional[float]:
        now = time.monotonic()
        if centroid is not None:
            self.inspection_centroid_history.append((now, centroid[0], centroid[1]))
        cutoff = now - INSPECTION_STABILITY_WINDOW_SEC
        self.inspection_centroid_history = [
            sample for sample in self.inspection_centroid_history if sample[0] >= cutoff
        ]
        if len(self.inspection_centroid_history) < 3:
            return None

        mean_x = sum(sample[1] for sample in self.inspection_centroid_history) / len(
            self.inspection_centroid_history
        )
        mean_y = sum(sample[2] for sample in self.inspection_centroid_history) / len(
            self.inspection_centroid_history
        )
        jitter = math.sqrt(
            sum(
                (sample[1] - mean_x) * (sample[1] - mean_x)
                + (sample[2] - mean_y) * (sample[2] - mean_y)
                for sample in self.inspection_centroid_history
            )
            / len(self.inspection_centroid_history)
        )
        return clamp(1.0 - jitter / INSPECTION_STABILITY_MAX_JITTER_NORM, 0.0, 1.0)

    def draw_inspection_confidence(self, cv2, frame):
        snapshot = self.state.snapshot()
        inspection = snapshot.get("inspection", {})
        target_position = inspection.get("target_position")
        drone_position = inspection.get("drone_position")
        camera_position = inspection.get("camera_position")
        score_camera_position = snapshot.get("fpv_camera_position") or camera_position
        score_camera_rotation = snapshot.get("fpv_camera_rotation")
        score_camera_resolution = snapshot.get("fpv_camera_resolution")
        if target_position is None or drone_position is None:
            self.inspection_centroid_history.clear()
            return

        active = bool(inspection.get("active"))
        if active:
            visual = self.score_projected_inspection_target(
                cv2,
                frame,
                target_position,
                score_camera_position,
                score_camera_rotation,
                score_camera_resolution,
            )
            stability_score = self.update_inspection_stability(visual["centroid"])
            stability_for_confidence = 0.5 if stability_score is None else stability_score
            visual_score = float(visual["score"])
        else:
            self.inspection_centroid_history.clear()
            stability_score = None
            stability_for_confidence = 0.0
            visual_score = 0.0

        distance_score = float(inspection.get("distance_score", 0.0))
        angle_score = float(inspection.get("viewing_angle_score", 0.0))
        weights = self.inspection_weights
        confidence = clamp(
            weights[0] * distance_score
            + weights[1] * angle_score
            + weights[2] * visual_score
            + weights[3] * stability_for_confidence,
            0.0,
            1.0,
        )
        result = "PASS" if confidence >= self.inspection_pass_threshold else "FAIL"
        stable_text = "n/a" if not active else "..." if stability_score is None else f"{stability_score:.2f}"
        span_percent = inspection.get("span_percent")
        span_text = "n/a" if span_percent is None else f"{float(span_percent):.0f}%"
        confidence_text = f"{confidence:.2f} {result}" if active else "n/a outside radius"
        camera_distance_m = inspection.get("camera_distance_m", inspection.get("distance_m", 0.0))
        object_diameter_m = inspection.get("object_diameter_m")
        object_area_m2 = inspection.get("object_area_m2")
        diameter_text = "n/a" if object_diameter_m is None else f"{float(object_diameter_m):.2f} m"
        area_text = "n/a" if object_area_m2 is None else f"{float(object_area_m2):.2f} m^2"

        lines = [
            "INSPECTION",
            str(inspection.get("target_name", INSPECTION_TARGET_OBJECT)),
            (
                f"OBJ    {target_position[0]:.1f}, "
                f"{target_position[1]:.1f}, {target_position[2]:.1f}"
            ),
            f"DIAM   {diameter_text}",
            f"AREA   {area_text}",
            (
                f"DRONE  {drone_position[0]:.1f}, "
                f"{drone_position[1]:.1f}, {drone_position[2]:.1f}"
            ),
            (
                f"CAM    {camera_position[0]:.1f}, "
                f"{camera_position[1]:.1f}, {camera_position[2]:.1f}"
                if camera_position is not None
                else "CAM    n/a"
            ),
            f"DRONE-DIST {float(inspection.get('drone_distance_m', 0.0)):.1f} m",
            f"CAM-DIST   {float(camera_distance_m):.1f} m",
            f"ANGLE  {float(inspection.get('angle_deg', 0.0)):.1f} deg",
            f"VISUAL {visual_score:.2f}",
            f"STABLE {stable_text}",
            f"SPAN   {span_text}",
            "----------------",
            f"CONF   {confidence_text}",
        ]

        height, width = frame.shape[:2]
        compact = width <= 800 or height <= 650
        scale = 0.38 if compact else 0.44
        line_h = 16 if compact else 19
        panel_w = 280 if compact else 330
        panel_h = 18 + line_h * len(lines)
        x0 = max(18, width - panel_w - 16)
        y0 = max(18, height - panel_h - 16)
        x1 = min(width - 12, x0 + panel_w)
        y1 = min(height - 12, y0 + panel_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0.0, frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)

        y = y0 + line_h
        for index, line in enumerate(lines):
            color = (255, 255, 255)
            if index == 0:
                color = (80, 220, 255)
            elif line.startswith("CONF"):
                color = (90, 240, 90) if result == "PASS" else (0, 0, 255)
            self.draw_text(cv2, frame, line, (x0 + 10, y), scale=scale, color=color)
            y += line_h

    def draw_battery(self, cv2, frame):
        snapshot = self.state.snapshot()
        percent = clamp(float(snapshot["battery_percent"]), 0.0, 100.0)
        red = (0, 0, 255)
        height, width = frame.shape[:2]
        body_w = 63
        body_h = 15
        tip_w = 4
        percent_w = 15
        pip_x, pip_y, pip_w, pip_h = self.chase_pip_rect(frame)
        x = max(18, pip_x)
        y = max(18, min(height - body_h - 18, pip_y + pip_h + 12))
        if x + body_w + tip_w + percent_w + 12 > width - 18:
            x = max(18, width - body_w - tip_w - percent_w - 30)

        cv2.rectangle(frame, (x, y), (x + body_w, y + body_h), red, 2)
        cv2.rectangle(
            frame,
            (x + body_w + 2, y + 8),
            (x + body_w + tip_w, y + body_h - 8),
            red,
            -1,
        )

        fill_w = int((body_w - 8) * (percent / 100.0))
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (x + 4, y + 4),
                (x + 4 + fill_w, y + body_h - 4),
                red,
                -1,
            )

        self.draw_text(
            cv2,
            frame,
            f"{percent:3.0f}%",
            (x + body_w + tip_w + 8, y + 13),
            scale=0.38,
            color=red,
        )

    def draw_coverage_unfeasible(self, cv2, frame):
        snapshot = self.state.snapshot()
        if not snapshot["coverage_unfeasible"]:
            return

        text = "Coverage Unfeasible"
        scale = 1.25
        thickness = 2
        red = (0, 0, 255)
        height, width = frame.shape[:2]
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
        x = max(18, (width - text_w) // 2)
        y = max(text_h + 18, (height + text_h) // 2)
        pad_x = 28
        pad_y = 18

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - pad_x, y - text_h - pad_y),
            (x + text_w + pad_x, y + baseline + pad_y),
            (18, 18, 18),
            -1,
        )
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
        cv2.rectangle(
            frame,
            (x - pad_x, y - text_h - pad_y),
            (x + text_w + pad_x, y + baseline + pad_y),
            red,
            2,
        )
        self.draw_text(cv2, frame, text, (x, y), scale=scale, color=red)

    def draw_diagnostics(self, cv2, frame):
        snapshot = self.state.snapshot()
        diagnostics = snapshot["diagnostics"]
        panel_x = 18
        panel_y = 385
        panel_w = 620
        row_h = 20
        rows = [
            (
                "Requested",
                snapshot["last_requested_position"],
                snapshot["last_requested_yaw_deg"],
                snapshot["last_requested_label"],
            )
        ]
        for name in (
            "AirSim GT kin",
            "AirSim GT pose",
            "World object pose",
            "PX4 estimated kin",
            "Actual pose topic",
            "FPV camera msg",
            "Chase camera msg",
            "FPV get_images",
            "Chase get_images",
        ):
            item = diagnostics.get(name, {})
            rows.append(
                (
                    name,
                    item.get("position"),
                    item.get("yaw_deg"),
                    item.get("extra", ""),
                )
            )

        panel_h = 36 + row_h * len(rows)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x - 10, panel_y - 24),
            (panel_x + panel_w, panel_y + panel_h),
            (18, 18, 18),
            -1,
        )
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)
        cv2.rectangle(
            frame,
            (panel_x - 10, panel_y - 24),
            (panel_x + panel_w, panel_y + panel_h),
            (235, 235, 235),
            1,
        )
        self.draw_text(
            cv2,
            frame,
            "Coordinate Diagnostics          x        y        z      yaw",
            (panel_x, panel_y - 4),
            scale=0.48,
        )
        y = panel_y + 18
        for name, position, yaw_deg, extra in rows:
            yaw_text = "n/a" if yaw_deg is None else f"{yaw_deg:6.1f}"
            line = f"{name[:18]:18s} {vector_or_none(position)} {yaw_text:>6s}  {extra[:18]}"
            self.draw_text(cv2, frame, line, (panel_x, y), scale=0.46)
            y += row_h

    @staticmethod
    def draw_text(
        cv2,
        frame,
        text: str,
        origin: Tuple[int, int],
        scale: float = 0.6,
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def make_pose_ned_yaw(position_ned: Sequence[float], yaw_deg: float) -> Pose:
    half_yaw = math.radians(yaw_deg) / 2.0
    return Pose(
        {
            "translation": Vector3(
                {
                    "x": float(position_ned[0]),
                    "y": float(position_ned[1]),
                    "z": float(position_ned[2]),
                }
            ),
            "rotation": Quaternion(
                {
                    "w": math.cos(half_yaw),
                    "x": 0.0,
                    "y": 0.0,
                    "z": math.sin(half_yaw),
                }
            ),
            "frame_id": "DEFAULT_ID",
        }
    )


def format_scene_origin_xyz(position_ned: Sequence[float]) -> str:
    return " ".join(f"{component:g}" for component in position_ned)


def format_scene_origin_rpy(yaw_deg: float) -> str:
    return f"0 0 {yaw_deg:g}"


def resolve_config_path(config_name: str, sim_config_path: str) -> Path:
    config_path = Path(config_name)
    if config_path.is_absolute():
        return config_path

    config_dir = Path(sim_config_path)
    candidates = [
        config_dir / config_path,
        SCRIPT_DIR / config_dir / config_path,
        EXAMPLE_SCRIPTS_DIR / config_dir / config_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_jsonc(path: Path):
    return commentjson.loads(path.read_text(encoding="utf-8"))


def ensure_scene_camera_capture(
    sensor: Dict,
    width: int,
    height: int,
    fov_degrees: float,
    capture_interval_sec: float,
):
    sensor["enabled"] = True
    sensor["capture-interval"] = capture_interval_sec
    capture_settings = sensor.setdefault("capture-settings", [])
    scene_capture = next(
        (capture for capture in capture_settings if capture.get("image-type") == 0),
        None,
    )
    if scene_capture is None:
        scene_capture = {"image-type": 0}
        capture_settings.append(scene_capture)

    scene_capture.update(
        {
            "width": width,
            "height": height,
            "fov-degrees": fov_degrees,
            "capture-enabled": True,
            "streaming-enabled": True,
            "pixels-as-float": False,
            "compress": False,
            "target-gamma": 2.5,
        }
    )


def default_front_camera_sensor(
    width: int,
    height: int,
    fov_degrees: float,
    capture_interval_sec: float,
    enable_gimbal: bool = False,
):
    sensor = {
        "id": "FrontCamera",
        "type": "camera",
        "enabled": True,
        "parent-link": "Frame",
        "capture-interval": capture_interval_sec,
        "capture-settings": [
            {
                "image-type": 0,
                "width": width,
                "height": height,
                "fov-degrees": fov_degrees,
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
    if enable_gimbal:
        sensor["gimbal"] = {
            "lock-roll": True,
            "lock-pitch": True,
            "lock-yaw": False,
        }
    return sensor


def ensure_camera(
    robot_config: Dict,
    camera_id: str,
    width: int,
    height: int,
    fov_degrees: float,
    capture_interval_sec: float,
    enable_gimbal: bool = False,
):
    sensors = robot_config.setdefault("sensors", [])
    sensor = next((candidate for candidate in sensors if candidate.get("id") == camera_id), None)
    if sensor is None:
        if camera_id != "FrontCamera":
            raise RuntimeError(f"Camera '{camera_id}' is not present in the robot config")
        sensors.append(
            default_front_camera_sensor(
                width,
                height,
                fov_degrees,
                capture_interval_sec,
                enable_gimbal=enable_gimbal,
            )
        )
        projectairsim_log().info("Runtime config added FrontCamera for FPV")
        return
    if sensor.get("type") != "camera":
        raise RuntimeError(f"Sensor '{camera_id}' exists but is not a camera")
    ensure_scene_camera_capture(sensor, width, height, fov_degrees, capture_interval_sec)
    if camera_id == "FrontCamera" and enable_gimbal:
        sensor["gimbal"] = {
            "lock-roll": True,
            "lock-pitch": True,
            "lock-yaw": False,
        }


def ensure_px4_route_speed_params(robot_config: Dict, args):
    controller = robot_config.get("controller", {})
    if controller.get("type") != "px4-api":
        return

    px4_settings = controller.setdefault("px4-settings", {})
    parameters = px4_settings.setdefault("parameters", {})
    xy_speed_mps = max(0.1, float(args.route_speed_mps))
    manual_px4_mode = args.mode_flight in {"manual-px4", "px4"}
    if manual_px4_mode:
        xy_speed_mps = max(
            xy_speed_mps,
            float(args.manual_px4_param_speed_limit_mps),
            manual_initial_speed_mps(args),
        )
    configured_vertical_speed_mps = float(args.route_vertical_speed_mps)
    vertical_speed_mps = max(
        0.1,
        configured_vertical_speed_mps if configured_vertical_speed_mps > 0.0 else xy_speed_mps,
    )
    if manual_px4_mode:
        vertical_speed_mps = max(
            vertical_speed_mps,
            float(args.manual_px4_param_vertical_speed_limit_mps),
        )
    parameters.update(
        {
            "MPC_VEL_MANUAL": xy_speed_mps,
            "MPC_XY_CRUISE": xy_speed_mps,
            "MPC_XY_VEL_MAX": xy_speed_mps,
            "MPC_Z_VEL_MAX_UP": vertical_speed_mps,
            "MPC_Z_VEL_MAX_DN": vertical_speed_mps,
        }
    )
    projectairsim_log().info(
        "Runtime config set PX4 route speed params: xy=%.1f m/s vertical-down=%.1f m/s",
        xy_speed_mps,
        vertical_speed_mps,
    )


def make_runtime_scene_config(args, route: Sequence[RoutePoint]):
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

    start = route[0]
    origin = target_actor.setdefault("origin", {})
    origin["xyz"] = format_scene_origin_xyz(route_to_ned(start.position))
    origin["rpy-deg"] = format_scene_origin_rpy(start.yaw_deg)

    temp_dir = tempfile.TemporaryDirectory(prefix="offshore_demo_")
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
                ensure_camera(
                    robot_config,
                    args.fpv_camera,
                    args.camera_width,
                    args.camera_height,
                    args.camera_fov_degrees,
                    args.camera_capture_interval_sec,
                    enable_gimbal=args.gimbal,
                )
                ensure_camera(
                    robot_config,
                    args.chase_camera,
                    args.camera_width,
                    args.camera_height,
                    args.camera_fov_degrees,
                    args.camera_capture_interval_sec,
                )
                ensure_px4_route_speed_params(robot_config, args)

            suffix = robot_config_path.suffix or ".jsonc"
            output_name = f"{robot_config_path.stem}_{actor_index}_offshore{suffix}"
            (temp_config_dir / output_name).write_text(
                commentjson.dumps(robot_config, indent=2) + "\n",
                encoding="utf-8",
            )
            actor["robot-config"] = output_name

        for env_actor in scene_config.get("environment-actors", []):
            if env_actor.get("type") != "env_actor" or not env_actor.get("env-actor-config"):
                continue
            env_config_path = resolve_config_path(
                env_actor["env-actor-config"],
                str(scene_path.parent),
            )
            shutil.copy2(env_config_path, temp_config_dir / env_config_path.name)
            env_actor["env-actor-config"] = env_config_path.name

        scene_name = scene_path.name
        (temp_config_dir / scene_name).write_text(
            commentjson.dumps(scene_config, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        temp_dir.cleanup()
        raise

    projectairsim_log().info(
        "Runtime offshore scene: %s with %s origin xyz=%s rpy-deg=%s",
        temp_config_dir / scene_name,
        args.drone_name,
        format_scene_origin_xyz(route_to_ned(start.position)),
        format_scene_origin_rpy(start.yaw_deg),
    )
    return temp_dir, scene_name, str(temp_config_dir)


def route_points_from_constants(route_constants) -> List[RoutePoint]:
    return [
        RoutePoint(label, [float(pos[0]), float(pos[1]), float(pos[2])], float(yaw_deg))
        for label, pos, yaw_deg in route_constants
    ]


def route_from_constants() -> List[RoutePoint]:
    return route_points_from_constants(OFFSHORE_ROUTE)


def full_route_from_constants() -> List[RoutePoint]:
    return route_points_from_constants(globals().get("FULL_OFFSHORE_ROUTE", OFFSHORE_ROUTE))


def nearest_valid_planner_point(
    planner,
    args,
    point: Sequence[float],
    label: str,
) -> List[float]:
    requested = [float(point[0]), float(point[1]), float(point[2])]
    if planner.check_coordinate_validity(requested, is_NED=True):
        return requested

    step = max(0.5, float(args.object_scan_resolution_m))
    horizontal_radius_m = max(step, float(args.replan_endpoint_search_radius_m))
    vertical_radius_m = max(0.0, float(args.replan_endpoint_vertical_search_m))
    horizontal_cells = int(math.ceil(horizontal_radius_m / step))
    vertical_cells = int(math.ceil(vertical_radius_m / step))

    best = None
    best_distance = math.inf
    for z_cell in range(-vertical_cells, vertical_cells + 1):
        dz = z_cell * step
        for x_cell in range(-horizontal_cells, horizontal_cells + 1):
            dx = x_cell * step
            for y_cell in range(-horizontal_cells, horizontal_cells + 1):
                dy = y_cell * step
                horizontal_distance = math.hypot(dx, dy)
                if horizontal_distance > horizontal_radius_m + 1e-6:
                    continue

                candidate = [requested[0] + dx, requested[1] + dy, requested[2] + dz]
                if not planner.check_coordinate_validity(candidate, is_NED=True):
                    continue

                distance = distance_between(candidate, requested)
                if distance < best_distance:
                    best = candidate
                    best_distance = distance

    if best is None:
        raise RuntimeError(
            f"{label} point is occupied or outside the A* grid, and no nearby free "
            f"point was found within horizontal {horizontal_radius_m:.1f}m / "
            f"vertical {vertical_radius_m:.1f}m: {requested}"
        )

    projectairsim_log().warning(
        "%s point %s is not free for A*. Using nearest free point %s instead "
        "(offset %.1fm).",
        label,
        format_vector3(requested),
        format_vector3(best),
        best_distance,
    )
    return best


def point_on_segment_by_fraction(
    start: Sequence[float],
    end: Sequence[float],
    fraction: float,
) -> List[float]:
    return interpolate_point(start, end, fraction)


def first_blocked_polyline_sample(
    planner,
    args,
    points: Sequence[Sequence[float]],
    first_segment_start_ignore_m: float = 0.0,
) -> Optional[Tuple[List[float], float]]:
    spacing_m = max(0.5, float(args.object_scan_sample_spacing_m))
    ignored_first_m = max(0.0, float(first_segment_start_ignore_m))
    travelled_m = 0.0

    for segment_index in range(1, len(points)):
        start = points[segment_index - 1]
        end = points[segment_index]
        segment_length_m = distance_between(start, end)
        if segment_length_m <= 1e-6:
            continue

        sample_count = max(1, int(math.ceil(segment_length_m / spacing_m)))
        for sample_index in range(0, sample_count + 1):
            along_m = min(segment_length_m, sample_index * spacing_m)
            if sample_index == sample_count:
                along_m = segment_length_m
            if segment_index == 1 and along_m < ignored_first_m:
                continue

            sample = interpolate_point(start, end, along_m / segment_length_m)
            if not is_route_corridor_clear(planner, sample, args):
                return sample, travelled_m + along_m

        travelled_m += segment_length_m

    return None


def build_lateral_bypass_points(
    planner,
    args,
    start: Sequence[float],
    target: Sequence[float],
    obstacle_point: Sequence[float],
    label: str,
) -> List[List[float]]:
    segment = [float(target[index]) - float(start[index]) for index in range(3)]
    segment_length_sq = sum(component * component for component in segment)
    if segment_length_sq <= 1e-9:
        return [route_to_ned(target)]

    offset = [float(obstacle_point[index]) - float(start[index]) for index in range(3)]
    obstacle_fraction = clamp(
        sum(offset[index] * segment[index] for index in range(3)) / segment_length_sq,
        0.0,
        1.0,
    )
    segment_length_m = math.sqrt(segment_length_sq)
    along_fraction = max(0.0, float(args.bypass_along_distance_m)) / segment_length_m
    before_fraction = clamp(obstacle_fraction - along_fraction, 0.0, 1.0)
    after_fraction = clamp(obstacle_fraction + along_fraction, 0.0, 1.0)

    dx = segment[0]
    dy = segment[1]
    horizontal_length = math.hypot(dx, dy)
    if horizontal_length <= 1e-6:
        return [route_to_ned(target)]

    left = [-dy / horizontal_length, dx / horizontal_length, 0.0]
    base_lateral_offset_m = max(
        float(args.bypass_lateral_offset_m),
        float(args.object_path_clearance_m) * 2.0,
    )
    max_lateral_offset_m = max(
        base_lateral_offset_m,
        float(args.bypass_max_lateral_offset_m),
    )
    lateral_step_m = max(
        float(args.bypass_lateral_step_m),
        float(args.object_scan_resolution_m),
    )

    def shifted(fraction: float, side: float, lateral_offset_m: float) -> List[float]:
        base = point_on_segment_by_fraction(start, target, fraction)
        return [
            base[0] + left[0] * side * lateral_offset_m,
            base[1] + left[1] * side * lateral_offset_m,
            base[2],
        ]

    best_side = None
    best_offset_m = base_lateral_offset_m
    best_score = -math.inf
    best_points = None

    offset_count = int(
        math.floor((max_lateral_offset_m - base_lateral_offset_m) / lateral_step_m)
    )
    lateral_offsets_m = [
        base_lateral_offset_m + offset_index * lateral_step_m
        for offset_index in range(0, offset_count + 1)
    ]
    if lateral_offsets_m[-1] < max_lateral_offset_m - 1e-6:
        lateral_offsets_m.append(max_lateral_offset_m)

    for lateral_offset_m in lateral_offsets_m:
        for side in (1.0, -1.0):
            raw_points = [
                shifted(before_fraction, side, lateral_offset_m),
                shifted(obstacle_fraction, side, lateral_offset_m),
                shifted(after_fraction, side, lateral_offset_m),
            ]
            score = 0.0
            snapped_points = []
            snap_distance_m = 0.0
            for point_index, point in enumerate(raw_points, start=1):
                if is_route_corridor_clear(planner, point, args):
                    score += 10.0
                    snapped_points.append(point)
                else:
                    snapped = nearest_valid_planner_point(
                        planner,
                        args,
                        point,
                        f"{label} lateral bypass {point_index}",
                    )
                    snapped = clamp_leg_altitude(
                        snapped,
                        start,
                        target,
                        args.bypass_vertical_margin_m,
                    )
                    snap_distance_m += distance_between(point, snapped)
                    snapped_points.append(snapped)
                    if is_route_corridor_clear(planner, snapped, args):
                        score += 5.0
                    else:
                        score -= 20.0

            corridor_points = [route_to_ned(start), *snapped_points, route_to_ned(target)]
            blocked = first_blocked_polyline_sample(
                planner,
                args,
                corridor_points,
                first_segment_start_ignore_m=args.object_scan_start_ignore_m,
            )

            side_name = "left" if side > 0 else "right"
            if blocked is None:
                score += 10000.0 - lateral_offset_m - snap_distance_m
                projectairsim_log().info(
                    "%s lateral bypass candidate %s offset=%.1fm clear "
                    "score=%.1f first=%s",
                    label,
                    side_name,
                    lateral_offset_m,
                    score,
                    format_vector3(snapped_points[0]),
                )
            else:
                blocked_sample, blocked_distance_m = blocked
                score += blocked_distance_m - lateral_offset_m - snap_distance_m
                projectairsim_log().info(
                    "%s lateral bypass candidate %s offset=%.1fm blocked at %s "
                    "after %.1fm score=%.1f first=%s",
                    label,
                    side_name,
                    lateral_offset_m,
                    format_vector3(blocked_sample),
                    blocked_distance_m,
                    score,
                    format_vector3(snapped_points[0]),
                )

            if score > best_score:
                best_score = score
                best_side = side_name
                best_offset_m = lateral_offset_m
                best_points = snapped_points
            if blocked is None:
                break
        if best_points is not None and best_score >= 9000.0:
            break

    projectairsim_log().warning(
        "Using %s lateral bypass for %s with %.1fm offset around obstacle %s.",
        best_side,
        label,
        best_offset_m,
        format_vector3(obstacle_point),
    )
    result = []
    for point in best_points or []:
        append_unique_point(result, point)
    append_unique_point(result, target)
    return result


def plan_obstacle_aware_route(
    world: World,
    args,
    route: Sequence[RoutePoint],
) -> List[RoutePoint]:
    if not args.replan_on_object or len(route) < 2:
        return list(route)

    planned_path = [route_to_ned(route[0].position)]
    bypass_legs = 0

    for index in range(1, len(route)):
        start = planned_path[-1]
        target = route_to_ned(route[index].position)
        leg_path = [start, target]
        label = f"{route[index - 1].label} -> {route[index].label}"

        projectairsim_log().info(
            "Checking leg %d/%d for obstacles: %s",
            index,
            len(route) - 1,
            label,
        )
        scan_margin_m = float(args.object_scan_margin_m)
        if args.obstacle_planner == "lateral":
            scan_margin_m = max(
                scan_margin_m,
                max(
                    float(args.bypass_lateral_offset_m),
                    float(args.bypass_max_lateral_offset_m),
                )
                + float(args.object_path_clearance_m)
                + 2.0 * float(args.object_scan_resolution_m),
            )
        scan = create_static_route_scan(
            world,
            args,
            leg_path,
            log_scan=args.log_obstacle_scans,
            margin_override_m=scan_margin_m,
        )
        obstacle = find_static_route_obstacle(
            scan,
            args,
            leg_path,
            log_clear=args.log_obstacle_scans,
        )
        if obstacle is None:
            append_unique_point(planned_path, target)
            continue

        bypass_legs += 1
        projectairsim_log().warning(
            "Obstacle detected on %s at NED %s. Planning %s bypass to %s.",
            label,
            format_vector3(obstacle.obstacle_point),
            args.obstacle_planner,
            route[index].label,
        )
        if args.obstacle_planner == "lateral":
            bypass_path = build_lateral_bypass_points(
                scan.planner,
                args,
                start,
                target,
                obstacle.obstacle_point,
                label,
            )
            projectairsim_log().warning(
                "Lateral bypass for %s: inserted %d route point(s)",
                label,
                len(bypass_path),
            )
            for point in bypass_path:
                append_unique_point(planned_path, point)
            continue

        plan_start = nearest_valid_planner_point(scan.planner, args, start, f"{label} start")
        plan_target = nearest_valid_planner_point(scan.planner, args, target, f"{label} target")
        dense_path = scan.planner.generate_plan(plan_start, plan_target)
        if not dense_path:
            raise RuntimeError(f"A* did not find an obstacle bypass for {label}")

        bypass_path = sparsify_path(
            [[float(point[0]), float(point[1]), float(point[2])] for point in dense_path],
            args.replan_waypoint_spacing_m,
        )
        projectairsim_log().warning(
            "A* bypass for %s: %d dense points -> %d route points",
            label,
            len(dense_path),
            len(bypass_path),
        )
        for point in bypass_path[1:]:
            append_unique_point(
                planned_path,
                clamp_leg_altitude(
                    point,
                    start,
                    target,
                    args.bypass_vertical_margin_m,
                ),
            )
        append_unique_point(planned_path, target)

    if bypass_legs == 0:
        projectairsim_log().info("Obstacle planner: original route corridor is clear")
        return list(route)

    planned_route = route_points_from_path(planned_path, route)
    projectairsim_log().warning(
        "Obstacle planner expanded route from %d to %d points across %d bypass leg(s)",
        len(route),
        len(planned_route),
        bypass_legs,
    )
    return planned_route


def drain_teleport_request(state: DemoState) -> Optional[int]:
    delta = None
    while not state.teleport_requests.empty():
        if delta is None:
            delta = state.teleport_requests.get()
        else:
            state.teleport_requests.get()
    return delta


def nearest_route_index(route: Sequence[RoutePoint], position: Sequence[float]) -> int:
    best_index = 0
    best_distance = float("inf")
    for index, point in enumerate(route):
        distance = distance_between(position, point.position)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def update_pose_topic_diagnostic(state: DemoState, pose_msg):
    state.update_diagnostic(
        "Actual pose topic",
        extract_pose_position(pose_msg),
        extract_pose_yaw_deg(pose_msg),
        "topic",
    )


def safe_call(label: str, fn):
    try:
        return fn()
    except Exception as exc:
        if label not in WARNED_UNAVAILABLE:
            projectairsim_log().warning("%s unavailable: %s", label, exc)
            WARNED_UNAVAILABLE.add(label)
        return None


def update_inspection_geometry(
    world: Optional[World],
    state: DemoState,
    args,
    camera_position: Sequence[float],
    drone_position: Sequence[float],
):
    if world is None:
        state.clear_inspection_geometry("no world")
        return

    object_names = [
        INSPECTION_TARGET_OBJECT,
        INSPECTION_NORMAL_OBJECT,
        INSPECTION_ROOT_OBJECT,
        INSPECTION_TIP_OBJECT,
    ]
    object_poses = safe_call(
        "Inspection object poses",
        lambda: world.get_object_poses(object_names),
    )
    if not object_poses or len(object_poses) != len(object_names):
        state.clear_inspection_geometry("objects unavailable")
        return

    positions = [extract_pose_position(pose) for pose in object_poses]
    if any(position is None for position in positions):
        state.clear_inspection_geometry("objects unavailable")
        return

    target_position, normal_position, root_position, tip_position = positions
    target_scale = safe_call(
        "Inspection target scale",
        lambda: world.get_object_scale(INSPECTION_TARGET_OBJECT),
    )
    object_diameter_m, object_area_m2 = sphere_size_from_scale(target_scale)
    radius_m = max(0.1, float(args.inspection_radius_m))
    camera_distance_m = distance_between(camera_position, target_position)
    drone_distance_m = distance_between(drone_position, target_position)
    active = camera_distance_m <= radius_m

    surface_normal = vector_subtract(normal_position, target_position)
    view_direction = vector_subtract(camera_position, target_position)
    angle_deg = angle_between_vectors_deg(surface_normal, view_direction)
    if angle_deg is None:
        viewing_angle_score = 0.0
        angle_deg = 180.0
    else:
        viewing_angle_score = clamp(1.0 - angle_deg / 90.0, 0.0, 1.0)

    distance_score = clamp(1.0 - camera_distance_m / radius_m, 0.0, 1.0)
    state.update_inspection_geometry(
        {
            "active": active,
            "status": "inside" if active else "outside",
            "target_name": INSPECTION_TARGET_OBJECT,
            "target_position": list(target_position),
            "drone_position": list(drone_position),
            "camera_position": list(camera_position),
            "object_scale": finite_vector(target_scale),
            "object_diameter_m": object_diameter_m,
            "object_area_m2": object_area_m2,
            "distance_m": float(camera_distance_m),
            "camera_distance_m": float(camera_distance_m),
            "drone_distance_m": float(drone_distance_m),
            "angle_deg": float(angle_deg),
            "distance_score": float(distance_score),
            "viewing_angle_score": float(viewing_angle_score),
            "span_percent": span_percent_along_blade(
                target_position,
                root_position,
                tip_position,
            ),
            "time": time.time(),
        }
    )


def update_drone_diagnostics(
    drone: Drone,
    state: DemoState,
    world: Optional[World] = None,
    object_name: Optional[str] = None,
):
    gt_kin = safe_call("AirSim ground-truth kinematics", drone.get_ground_truth_kinematics)
    if gt_kin is not None:
        state.update_diagnostic(
            "AirSim GT kin",
            extract_pose_position(gt_kin),
            extract_pose_yaw_deg(gt_kin),
            "GetGroundTruthKinematics",
        )

    gt_pose = safe_call("AirSim ground-truth pose", drone.get_ground_truth_pose)
    if gt_pose is not None:
        state.update_diagnostic(
            "AirSim GT pose",
            extract_pose_position(gt_pose),
            extract_pose_yaw_deg(gt_pose),
            "GetGroundTruthPose",
        )

    if world is not None and object_name:
        object_poses = safe_call(
            f"World get_object_poses {object_name}",
            lambda: world.get_object_poses([object_name]),
        )
        if object_poses:
            state.update_diagnostic(
                "World object pose",
                extract_pose_position(object_poses[0]),
                extract_pose_yaw_deg(object_poses[0]),
                f"GetObjectPoses {object_name}",
            )

    est_kin = safe_call("PX4 estimated kinematics", drone.get_estimated_kinematics)
    if est_kin is not None:
        state.update_diagnostic(
            "PX4 estimated kin",
            extract_pose_position(est_kin),
            extract_pose_yaw_deg(est_kin),
            "GetEstimatedKinematics",
        )

    gt_geo = safe_call("AirSim ground-truth geo", drone.get_ground_truth_geo_location)
    if gt_geo is not None:
        state.update_diagnostic("AirSim GT geo", None, None, geo_summary(gt_geo))

    est_geo = safe_call("PX4 estimated geo", drone.get_estimated_geo_location)
    if est_geo is not None:
        state.update_diagnostic("PX4 estimated geo", None, None, geo_summary(est_geo))

    if "GPS" in drone.sensors and "gps" in drone.sensors["GPS"]:
        gps = safe_call("GPS sensor", lambda: drone.get_gps_data("GPS"))
        if gps is not None:
            state.update_diagnostic("GPS sensor", None, None, geo_summary(gps))


def update_requested_camera_diagnostics(drone: Drone, state: DemoState, args):
    for camera_id, label in (
        (args.fpv_camera, "FPV get_images"),
        (args.chase_camera, "Chase get_images"),
    ):
        images = safe_call(
            f"{camera_id} GetImages",
            lambda camera_id=camera_id: drone.get_images(
                camera_id,
                [ImageType.SCENE],
            ),
        )
        if not images:
            continue

        image = images.get(ImageType.SCENE) or images.get(int(ImageType.SCENE))
        if image is None:
            continue

        position = image_pose_position(image)
        yaw_deg = image_pose_yaw_deg(image)
        extra = "GetImages"
        if position is not None:
            offset_m = distance_between(position, state.snapshot()["position"])
            extra = f"GetImages offset={offset_m:.1f}m"
        state.update_diagnostic(label, position, yaw_deg, extra)


def reset_camera_renderers(drone: Drone, camera_ids: Sequence[str]):
    for camera_id in camera_ids:
        safe_call(
            f"Reset camera {camera_id}",
            lambda camera_id=camera_id: drone.reset_camera_pose(
                camera_id,
                wait_for_pose_update=True,
            ),
        )


def diagnostics_lines(snapshot: Dict) -> List[str]:
    diagnostics = snapshot["diagnostics"]
    rows = [
        (
            "Requested",
            snapshot["last_requested_position"],
            snapshot["last_requested_yaw_deg"],
            snapshot["last_requested_label"],
        )
    ]
    for name in (
        "AirSim GT kin",
        "AirSim GT pose",
        "World object pose",
        "PX4 estimated kin",
        "Actual pose topic",
        "FPV camera msg",
        "Chase camera msg",
        "FPV get_images",
        "Chase get_images",
    ):
        item = diagnostics.get(name, {})
        rows.append((name, item.get("position"), item.get("yaw_deg"), item.get("extra", "")))

    lines = [
        "",
        "Coordinate diagnostics",
        "source                x        y        z      yaw    note",
        "---------------------------------------------------------------",
    ]
    for name, position, yaw_deg, extra in rows:
        yaw_text = "   n/a" if yaw_deg is None else f"{yaw_deg:6.1f}"
        lines.append(
            f"{name[:18]:18s} {vector_or_none(position)} {yaw_text}  {extra}"
        )
    for name in ("AirSim GT geo", "PX4 estimated geo", "GPS sensor"):
        extra = diagnostics.get(name, {}).get("extra")
        if extra:
            lines.append(f"{name[:18]:18s} {extra}")
    return lines


def log_diagnostics(snapshot: Dict):
    for line in diagnostics_lines(snapshot):
        projectairsim_log().info(line)


def log_mission_snapshot(label: str, index: int, snapshot: Dict):
    projectairsim_log().info("")
    projectairsim_log().info("=" * 72)
    projectairsim_log().info("Mission snapshot %02d: %s", index, label)
    projectairsim_log().info("=" * 72)
    log_diagnostics(snapshot)


def log_battery_start_if_needed(started: bool):
    if started:
        projectairsim_log().info(
            "Battery countdown started at %s: 100%%, depleting 1%% every %.1f seconds",
            BATTERY_START_LABEL,
            BATTERY_SECONDS_PER_PERCENT,
        )


def log_coverage_unfeasible_if_needed(state: DemoState):
    if state.take_coverage_unfeasible_log_event():
        projectairsim_log().warning(
            "Coverage Unfeasible: battery reached 0%%. Stopping drone and simulation."
        )


async def hold_then_request_stop(state: DemoState):
    await asyncio.sleep(COVERAGE_UNFEASIBLE_HOLD_SEC)
    state.request_stop()


def resolve_wrf_path(args) -> Path:
    requested = Path(args.wrf_file)
    if not requested.is_absolute():
        requested = REPO_ROOT / requested
    if requested.exists():
        return requested

    if str(args.wrf_file) == str(DEFAULT_WRF_FILE):
        nc_files = sorted((REPO_ROOT / "wind_data").glob("*.nc"))
        if len(nc_files) == 1:
            projectairsim_log().warning(
                "Default WRF file %s was not found; using only available NetCDF file %s",
                requested,
                nc_files[0],
            )
            return nc_files[0]

    raise FileNotFoundError(f"WRF file not found: {requested}")


def resolve_wrf_origin(args) -> Tuple[Optional[float], Optional[float]]:
    if args.wrf_origin_lat is not None or args.wrf_origin_lon is not None:
        if args.wrf_origin_lat is None or args.wrf_origin_lon is None:
            raise ValueError("--wrf-origin-lat and --wrf-origin-lon must be provided together")
        return float(args.wrf_origin_lat), float(args.wrf_origin_lon)

    return None, None


def create_wrf_wind_field(_world: World, args) -> WRFWindField:
    wrf_path = resolve_wrf_path(args)
    origin_lat, origin_lon = resolve_wrf_origin(args)
    try:
        field = WRFWindField(
            wrf_path,
            origin_lat,
            origin_lon,
            time_index=args.wrf_time_index,
            region_half_size_m=args.wrf_region_half_size_m,
            altitude_min_agl_m=args.wrf_altitude_min_agl_m,
            altitude_max_agl_m=args.wrf_altitude_max_agl_m,
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc}. The Unreal scene is synthetic and not georeferenced; use "
            "--wrf-origin-lat and --wrf-origin-lon only to choose the WRF cell "
            "mapped to simulation NED [0,0,0]."
        ) from exc

    for line in field.startup_summary_lines():
        projectairsim_log().info(line)
    if field.origin_hgt_m > 25.0:
        projectairsim_log().warning(
            "WRF origin HGT %.2f m is unexpectedly high for an offshore demo.",
            field.origin_hgt_m,
        )
    if field.origin_landmask >= 0.5:
        projectairsim_log().warning(
            "WRF origin LANDMASK=%.0f indicates land, not water.",
            field.origin_landmask,
        )
    return field


def log_wrf_debug(sample: WRFWindSample):
    projectairsim_log().info("WRF WIND")
    projectairsim_log().info(
        "Sim position NED:  N=%.1f E=%.1f D=%.1f",
        sample.sim_north_m,
        sample.sim_east_m,
        sample.sim_down_m,
    )
    projectairsim_log().info("WRF location:      lat=%.7f lon=%.7f", sample.lat, sample.lon)
    projectairsim_log().info("Altitude AGL:      %.1f m", sample.altitude_agl_m)
    projectairsim_log().info("U East:            %.3f m/s", sample.u_east_mps)
    projectairsim_log().info("V North:           %.3f m/s", sample.v_north_mps)
    projectairsim_log().info("W Up:              %.3f m/s", sample.w_up_mps)
    projectairsim_log().info("AirSim North:      %.3f m/s", sample.north_mps)
    projectairsim_log().info("AirSim East:       %.3f m/s", sample.east_mps)
    projectairsim_log().info("AirSim Down:       %.3f m/s", sample.down_mps)
    projectairsim_log().info(
        "Speed:             %.3f m/s direction TO %.1f deg",
        sample.horizontal_speed_mps,
        sample.direction_to_deg,
    )
    if sample.outside_region:
        projectairsim_log().warning(
            "WRF wind sample is outside configured wind-farm region; continuing interpolation."
        )
    if sample.outside_dataset:
        projectairsim_log().warning(
            "WRF wind sample is outside the WRF grid; using nearest grid edge."
        )
    if sample.w_vertical_note:
        projectairsim_log().info("Vertical W note:   %s", sample.w_vertical_note)


async def run_wrf_wind_updater(
    drone: Drone,
    world: World,
    state: DemoState,
    wrf_field: WRFWindField,
    args,
):
    update_interval_sec = 1.0 / max(0.1, float(args.wind_update_hz))
    last_debug_at = 0.0
    try:
        while not state.snapshot()["stop_requested"]:
            position_ned = get_pose_position_ned(drone)
            sample = wrf_field.sample_ned(position_ned)
            world.set_wind_velocity(sample.north_mps, sample.east_mps, sample.down_mps)
            state.update_wind_sample(sample)

            now = time.time()
            if args.wrf_debug and now - last_debug_at >= 1.0:
                log_wrf_debug(sample)
                last_debug_at = now

            await asyncio.sleep(update_interval_sec)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        projectairsim_log().warning("WRF wind updater failed: %s", exc)
        projectairsim_log().warning("WRF wind updater traceback:\n%s", traceback.format_exc())
        state.set_wind_enabled(False, "failed")
        state.request_stop()


def drain_manual_teleport_request(state: DemoState) -> bool:
    requested = False
    while not state.manual_teleport_requests.empty():
        state.manual_teleport_requests.get()
        requested = True
    return requested


def parse_manual_teleport(value: str) -> RoutePoint:
    cleaned = value.strip().strip("[]()")
    parts = cleaned.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("expected x,y,z,angle")

    numbers = [float(part) for part in parts]
    return RoutePoint("Custom", numbers[:3], numbers[3])


def sync_world_object_pose(world: Optional[World], object_name: str, pose: Pose):
    if world is None:
        return
    safe_call(
        f"World set_object_pose {object_name}",
        lambda: world.set_object_pose(object_name, pose, teleport=True),
    )


def cancel_last_task_safely(drone: Drone):
    safe_call("Cancel last drone task", drone.cancel_last_task)


async def apply_teleport(
    drone: Drone,
    state: DemoState,
    index: int,
    args,
    world: Optional[World] = None,
):
    point = state.route[index]
    cancel_last_task_safely(drone)
    target_ned = route_to_ned(point.position)
    state.update_requested(point.label, target_ned, point.yaw_deg)
    state.bump_camera_epoch()
    pose = make_pose_ned_yaw(target_ned, point.yaw_deg)
    drone.set_pose(pose, reset_kinematics=True)
    sync_world_object_pose(world, args.drone_name, pose)
    reset_camera_renderers(drone, [args.fpv_camera, args.chase_camera])
    await asyncio.sleep(0.1)
    update_drone_diagnostics(drone, state, world, args.drone_name)
    actual_ned = get_pose_position_ned(drone)
    actual_route = ned_to_route(actual_ned)
    state.update_pose(actual_route, point.yaw_deg)
    error_m = distance_between(actual_route, point.position)
    battery_started = state.mark_reached(index)
    log_battery_start_if_needed(battery_started)
    projectairsim_log().info(
        "Teleported to %s index=%d NED requested=%s actual=%s error=%.2fm heading=%.1f deg",
        point.label,
        index,
        format_vector3(target_ned),
        format_vector3(actual_ned),
        error_m,
        point.yaw_deg % 360.0,
    )
    if error_m > TELEPORT_WARN_ERROR_M:
        projectairsim_log().warning(
            "Teleport landed %.2fm from requested %s. This usually means the "
            "sim adjusted the pose because of collision/terrain, or PX4/physics "
            "moved the vehicle immediately after SetPose.",
            error_m,
            point.label,
        )


async def prompt_manual_teleport(
    drone: Drone,
    state: DemoState,
    args,
    world: Optional[World] = None,
):
    value = (
        await asyncio.to_thread(
            input,
            "\nTeleport NED [x,y,z,angle] (blank/cancel to skip): ",
        )
    ).strip()
    if not value or value.lower() in {"c", "cancel", "q", "quit"}:
        projectairsim_log().info("Custom teleport cancelled")
        return

    try:
        point = parse_manual_teleport(value)
    except ValueError as exc:
        projectairsim_log().warning("Invalid custom teleport '%s': %s", value, exc)
        return

    target_ned = route_to_ned(point.position)
    cancel_last_task_safely(drone)
    state.update_requested(point.label, target_ned, point.yaw_deg)
    state.bump_camera_epoch()
    pose = make_pose_ned_yaw(target_ned, point.yaw_deg)
    drone.set_pose(pose, reset_kinematics=True)
    sync_world_object_pose(world, args.drone_name, pose)
    reset_camera_renderers(drone, [args.fpv_camera, args.chase_camera])
    await asyncio.sleep(0.1)
    update_drone_diagnostics(drone, state, world, args.drone_name)
    actual_ned = get_pose_position_ned(drone)
    actual_route = ned_to_route(actual_ned)
    state.update_pose(actual_route, point.yaw_deg)
    projectairsim_log().info(
        "Teleported to custom NED requested=%s actual=%s heading=%.1f deg",
        format_vector3(target_ned),
        format_vector3(actual_ned),
        point.yaw_deg % 360.0,
    )


async def settle_and_sample(
    drone: Drone,
    state: DemoState,
    duration_sec: float,
    world: Optional[World] = None,
    object_name: Optional[str] = None,
):
    settle_until = time.time() + max(0.0, duration_sec)
    while time.time() < settle_until and not state.snapshot()["stop_requested"]:
        if state.check_battery_depleted() or state.snapshot()["coverage_unfeasible"]:
            log_coverage_unfeasible_if_needed(state)
            await hold_then_request_stop(state)
            break
        current_ned = get_pose_position_ned(drone)
        current = ned_to_route(current_ned)
        current_heading = heading_deg_360(get_pose_yaw_ned(drone))
        state.update_pose(current, current_heading)
        state.set_route_index(nearest_route_index(state.route, current))
        update_drone_diagnostics(drone, state, world, object_name)
        await asyncio.sleep(0.25)

    update_drone_diagnostics(drone, state, world, object_name)


async def fly_route_segment_smooth(
    drone: Drone,
    state: DemoState,
    args,
    previous_target: Sequence[float],
    target: Sequence[float],
    label: str,
    waypoint_index: int,
    stop_at_target: bool,
    commanded_velocity: Sequence[float],
    timeout_sec: float,
    world: Optional[World] = None,
) -> List[float]:
    command_duration_sec = max(0.05, args.velocity_command_duration_sec)
    acceleration_limit_mps2 = max(0.1, float(args.velocity_acceleration_limit_mps2))
    max_velocity_delta = acceleration_limit_mps2 * command_duration_sec
    max_yaw_rate_radps = math.radians(max(0.0, args.path_yaw_rate_dps))
    yaw_deadband_rad = math.radians(max(0.0, args.path_yaw_deadband_deg))
    yaw_response_sec = max(command_duration_sec, args.path_yaw_response_sec)
    slowdown_distance_m = max(args.route_acceptance_m, args.slowdown_distance_m)
    if stop_at_target:
        braking_distance_m = (
            float(args.route_speed_mps) * float(args.route_speed_mps)
            / (2.0 * acceleration_limit_mps2)
        )
        slowdown_distance_m = max(slowdown_distance_m, braking_distance_m)
    velocity_lookahead_m = max(0.0, args.velocity_lookahead_m)
    segment_delta = [
        float(target[0]) - float(previous_target[0]),
        float(target[1]) - float(previous_target[1]),
    ]
    segment_has_heading = math.hypot(segment_delta[0], segment_delta[1]) > 0.1
    velocity = [
        float(commanded_velocity[0]),
        float(commanded_velocity[1]),
        float(commanded_velocity[2]),
    ]
    start_pose = get_pose_position_ned(drone)
    started_at = time.time()
    last_report_at = 0.0

    while not state.snapshot()["stop_requested"]:
        current = get_pose_position_ned(drone)
        distance = distance_between(current, target)
        state.update_pose(ned_to_route(current), heading_deg_360(get_pose_yaw_ned(drone)))
        if state.check_battery_depleted() or state.snapshot()["coverage_unfeasible"]:
            log_coverage_unfeasible_if_needed(state)
            velocity = await brake_to_stop_by_velocity(
                drone,
                velocity,
                command_duration_sec,
                max_velocity_delta,
            )
            cancel_last_task_safely(drone)
            await hold_then_request_stop(state)
            return velocity

        if distance <= args.route_acceptance_m:
            if stop_at_target:
                velocity = await brake_to_stop_by_velocity(
                    drone,
                    velocity,
                    command_duration_sec,
                    max_velocity_delta,
                )
            projectairsim_log().info(
                "%s reached target %s; pose NED %s; error %.2f m",
                label,
                format_vector3(target),
                format_vector3(current),
                distance,
            )
            return velocity

        elapsed = time.time() - started_at
        if timeout_sec > 0.0 and elapsed > timeout_sec:
            cancel_last_task_safely(drone)
            raise RuntimeError(
                f"{label} timed out after {timeout_sec:.1f}s; "
                f"target {format_vector3(target)}, pose NED {format_vector3(current)}, "
                f"remaining {distance:.2f} m"
            )

        if (
            args.route_stuck_timeout_sec > 0.0
            and elapsed >= args.route_stuck_timeout_sec
            and distance_between(start_pose, current) < args.route_stuck_distance_m
            and distance_between(start_pose, target) > args.route_acceptance_m
        ):
            cancel_last_task_safely(drone)
            raise RuntimeError(
                f"{label} did not move in AirSim ground truth after "
                f"{elapsed:.1f}s; start {format_vector3(start_pose)}, "
                f"pose NED {format_vector3(current)}, target {format_vector3(target)}. "
                "PX4 may be armed/offboard, but the simulated actor is not coupled."
            )

        if elapsed - last_report_at >= args.pose_report_interval_sec:
            actual_velocity = get_ground_truth_velocity_ned(drone)
            projectairsim_log().info(
                "%s following segment to %s; pose NED %s; remaining %.2f m; "
                "commanded %.2f m/s actual %.2f m/s",
                label,
                format_vector3(target),
                format_vector3(current),
                distance,
                vector_length(velocity),
                vector_length(actual_velocity),
            )
            update_drone_diagnostics(drone, state, world, args.drone_name)
            last_report_at = elapsed

        max_route_deviation_m = max(0.0, float(args.max_route_deviation_m))
        if max_route_deviation_m > 0.0:
            closest_on_segment = closest_point_on_segment(previous_target, target, current)
            route_deviation_m = distance_between(current, closest_on_segment)
            if route_deviation_m > max_route_deviation_m:
                velocity = await brake_to_stop_by_velocity(
                    drone,
                    velocity,
                    command_duration_sec,
                    max_velocity_delta,
                )
                cancel_last_task_safely(drone)
                raise RuntimeError(
                    f"{label} drifted {route_deviation_m:.1f}m from active route "
                    f"segment; target {format_vector3(target)}, pose NED "
                    f"{format_vector3(current)}"
                )

        max_target_overshoot_m = max(0.0, float(args.max_target_overshoot_m))
        if max_target_overshoot_m > 0.0:
            segment_length_m = distance_between(previous_target, target)
            if segment_length_m > 1e-6:
                projection_fraction = segment_projection_fraction(previous_target, target, current)
                overshoot_m = max(0.0, (projection_fraction - 1.0) * segment_length_m)
                if overshoot_m > max_target_overshoot_m and distance > args.route_acceptance_m:
                    velocity = await brake_to_stop_by_velocity(
                        drone,
                        velocity,
                        command_duration_sec,
                        max_velocity_delta,
                    )
                    cancel_last_task_safely(drone)
                    raise RuntimeError(
                        f"{label} overshot active target by {overshoot_m:.1f}m; "
                        f"target {format_vector3(target)}, pose NED {format_vector3(current)}"
                    )

        steering_target = segment_lookahead_point(
            previous_target,
            target,
            current,
            velocity_lookahead_m,
        )
        if stop_at_target and distance <= slowdown_distance_m:
            steering_target = [float(target[0]), float(target[1]), float(target[2])]

        steering_distance = distance_between(current, steering_target)
        if steering_distance <= 1e-6:
            steering_target = [float(target[0]), float(target[1]), float(target[2])]
            steering_distance = distance

        delta = [steering_target[index] - current[index] for index in range(3)]
        speed_scale = 1.0
        if stop_at_target:
            approach_ratio = clamp(distance / max(slowdown_distance_m, 1e-6), 0.0, 1.0)
            speed_scale = max(0.08, approach_ratio * approach_ratio)
        desired_speed = min(
            args.route_speed_mps,
            steering_distance / command_duration_sec,
        ) * speed_scale
        desired_velocity = [
            (component / steering_distance) * desired_speed
            for component in delta
        ]
        max_vertical_speed_mps = max(0.0, float(args.route_vertical_speed_mps))
        if max_vertical_speed_mps > 0.0:
            desired_velocity[2] = clamp(
                desired_velocity[2],
                -max_vertical_speed_mps,
                max_vertical_speed_mps,
            )
        velocity = limit_vector_delta(velocity, desired_velocity, max_velocity_delta)

        yaw_rate_radps = 0.0
        horizontal_speed = math.hypot(velocity[0], velocity[1])
        if (
            args.face_travel_direction
            and segment_has_heading
            and horizontal_speed > 0.05
            and max_yaw_rate_radps > 0.0
        ):
            desired_yaw = math.atan2(segment_delta[1], segment_delta[0])
            yaw_error = wrap_angle_rad(desired_yaw - get_pose_yaw_ned(drone))
            if abs(yaw_error) > yaw_deadband_rad:
                yaw_error_to_close = math.copysign(
                    abs(yaw_error) - yaw_deadband_rad,
                    yaw_error,
                )
                yaw_rate_radps = clamp(
                    yaw_error_to_close / yaw_response_sec,
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

    return velocity


async def align_to_waypoint_yaw(
    drone: Drone,
    state: DemoState,
    args,
    target_yaw_deg: float,
    label: str,
    world: Optional[World] = None,
):
    if not args.align_yaw_at_waypoints:
        return

    await align_to_heading(
        drone,
        state,
        args,
        target_yaw_deg,
        label,
        world,
        args.waypoint_yaw_rate_dps,
        args.waypoint_yaw_acceptance_deg,
        args.waypoint_yaw_timeout_sec,
    )


async def align_to_heading(
    drone: Drone,
    state: DemoState,
    args,
    target_yaw_deg: float,
    label: str,
    world: Optional[World],
    yaw_rate_dps: float,
    yaw_acceptance_deg: float,
    yaw_timeout_sec: float,
):
    command_duration_sec = max(0.05, args.velocity_command_duration_sec)
    max_yaw_rate_radps = math.radians(max(0.0, yaw_rate_dps))
    if max_yaw_rate_radps <= 0.0:
        return

    target_yaw_rad = math.radians(float(target_yaw_deg) % 360.0)
    yaw_acceptance_rad = math.radians(max(0.0, yaw_acceptance_deg))
    yaw_response_sec = max(command_duration_sec, args.path_yaw_response_sec)
    started_at = time.time()
    last_report_at = 0.0

    while not state.snapshot()["stop_requested"]:
        current_ned = get_pose_position_ned(drone)
        current_yaw_rad = get_pose_yaw_ned(drone)
        yaw_error = wrap_angle_rad(target_yaw_rad - current_yaw_rad)
        state.update_pose(ned_to_route(current_ned), heading_deg_360(current_yaw_rad))
        if state.check_battery_depleted() or state.snapshot()["coverage_unfeasible"]:
            log_coverage_unfeasible_if_needed(state)
            return

        if abs(yaw_error) <= yaw_acceptance_rad:
            projectairsim_log().info(
                "%s yaw aligned to %.1f deg; current %.1f deg; error %.1f deg",
                label,
                float(target_yaw_deg) % 360.0,
                heading_deg_360(current_yaw_rad),
                math.degrees(abs(yaw_error)),
            )
            return

        elapsed = time.time() - started_at
        if yaw_timeout_sec > 0.0 and elapsed > yaw_timeout_sec:
            projectairsim_log().warning(
                "%s yaw alignment timed out at %.1f deg; target %.1f deg; error %.1f deg",
                label,
                heading_deg_360(current_yaw_rad),
                float(target_yaw_deg) % 360.0,
                math.degrees(abs(yaw_error)),
            )
            return

        if elapsed - last_report_at >= args.pose_report_interval_sec:
            projectairsim_log().info(
                "%s aligning yaw: current %.1f deg target %.1f deg error %.1f deg",
                label,
                heading_deg_360(current_yaw_rad),
                float(target_yaw_deg) % 360.0,
                math.degrees(abs(yaw_error)),
            )
            update_drone_diagnostics(drone, state, world, args.drone_name)
            last_report_at = elapsed

        yaw_rate_radps = clamp(
            yaw_error / yaw_response_sec,
            -max_yaw_rate_radps,
            max_yaw_rate_radps,
        )
        await drone.move_by_velocity_async(
            v_north=0.0,
            v_east=0.0,
            v_down=0.0,
            duration=command_duration_sec,
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
            yaw_is_rate=True,
            yaw=yaw_rate_radps,
        )
        await asyncio.sleep(command_duration_sec)


async def run_route_auto_flight(
    drone: Drone,
    state: DemoState,
    args,
    world: Optional[World] = None,
):
    if state.snapshot()["auto_flight_running"]:
        projectairsim_log().info("Auto flight is already running")
        return

    state.set_auto_flight_running(True)
    state.set_mode("auto flight", f"speed {args.route_speed_mps:.1f} m/s")
    commanded_velocity = [0.0, 0.0, 0.0]
    failed = False
    try:
        if not args.face_travel_direction:
            projectairsim_log().warning(
                "--no-face-travel-direction is ignored in offshore auto-flight; "
                "the drone always faces the active route segment."
            )
            args.face_travel_direction = True
        projectairsim_log().info(
            "Auto flight requested: teleporting to Origin, then flying %d route points at %.1f m/s",
            len(state.route),
            args.route_speed_mps,
        )
        await apply_teleport(drone, state, 0, args, world)
        await settle_and_sample(
            drone,
            state,
            args.post_teleport_flight_delay_sec,
            world,
            args.drone_name,
        )

        if not drone.enable_api_control():
            raise RuntimeError("Project AirSim rejected EnableApiControl for PX4 auto flight")
        await arm_with_retry(drone, args.arm_timeout_sec)

        if not args.skip_takeoff_before_route:
            projectairsim_log().info("Taking off before route flight")
            takeoff_task = await drone.takeoff_async(timeout_sec=args.takeoff_timeout_sec)
            await await_drone_task(
                drone,
                takeoff_task,
                "Takeoff",
                args.takeoff_timeout_sec + 5.0,
                args.pose_report_interval_sec,
            )
            projectairsim_log().info(
                "Takeoff completed; resetting back to exact Origin before route flight"
            )
            await apply_teleport(drone, state, 0, args, world)
            await settle_and_sample(
                drone,
                state,
                args.post_takeoff_origin_reset_delay_sec,
                world,
                args.drone_name,
            )
        else:
            projectairsim_log().info(
                "Skipping PX4 takeoff; route flight starts from teleported Origin pose"
            )

        await request_px4_control(drone)
        await asyncio.sleep(max(0.0, args.request_control_settle_sec))

        for index, point in enumerate(state.route[1:], start=1):
            if state.snapshot()["stop_requested"]:
                break

            target_ned = route_to_ned(point.position)
            current_ned = get_pose_position_ned(drone)
            leg_distance_m = distance_between(current_ned, target_ned)
            timeout_sec = args.route_move_timeout_sec
            if timeout_sec <= 0.0:
                timeout_sec = (
                    leg_distance_m
                    / max(args.route_speed_mps, 0.1)
                    * args.route_timeout_multiplier
                )

            with state.lock:
                state.target_index = index

            named_waypoint = is_named_waypoint(point)
            stop_at_target = args.stop_at_bypass_points or named_waypoint
            leg_behavior = (
                "stop+yaw"
                if named_waypoint
                else "stop"
                if stop_at_target
                else "pass-through"
            )
            state.update_requested(point.label, target_ned, point.yaw_deg)
            projectairsim_log().info(
                "Auto flight leg %d/%d to %s NED %s distance %.1fm speed %.1fm/s timeout %.1fs %s",
                index,
                len(state.route) - 1,
                point.label,
                format_vector3(target_ned),
                leg_distance_m,
                args.route_speed_mps,
                timeout_sec,
                leg_behavior,
            )
            previous_ned = route_to_ned(state.route[index - 1].position)
            if args.face_travel_direction and leg_distance_m > args.route_acceptance_m:
                travel_yaw_deg = segment_yaw_deg(previous_ned, target_ned, point.yaw_deg)
                await align_to_heading(
                    drone,
                    state,
                    args,
                    travel_yaw_deg,
                    f"{point.label} travel heading",
                    world,
                    args.path_yaw_rate_dps,
                    args.path_yaw_deadband_deg,
                    args.travel_yaw_timeout_sec,
                )
                if state.snapshot()["stop_requested"] or state.snapshot()["coverage_unfeasible"]:
                    break
            commanded_velocity = await fly_route_segment_smooth(
                drone,
                state,
                args,
                previous_ned,
                target_ned,
                point.label,
                index,
                stop_at_target,
                commanded_velocity,
                timeout_sec,
                world,
            )
            if state.snapshot()["stop_requested"] or state.snapshot()["coverage_unfeasible"]:
                break
            current_ned = get_pose_position_ned(drone)
            state.update_pose(ned_to_route(current_ned), heading_deg_360(get_pose_yaw_ned(drone)))
            battery_started = state.mark_reached(index)
            log_battery_start_if_needed(battery_started)
            update_drone_diagnostics(drone, state, world, args.drone_name)
            if named_waypoint:
                await align_to_waypoint_yaw(
                    drone,
                    state,
                    args,
                    point.yaw_deg,
                    point.label,
                    world,
                )
                if state.snapshot()["stop_requested"] or state.snapshot()["coverage_unfeasible"]:
                    break

        final_snapshot = state.snapshot()
        if final_snapshot["coverage_unfeasible"]:
            log_coverage_unfeasible_if_needed(state)
            if not final_snapshot["stop_requested"]:
                await hold_then_request_stop(state)
        elif final_snapshot["stop_requested"]:
            projectairsim_log().info("Auto flight stopped")
        else:
            projectairsim_log().info("Auto flight complete")
            state.set_mode("auto flight complete", "mission complete")
            state.request_stop()
    except asyncio.CancelledError:
        projectairsim_log().info("Auto flight cancelled")
        cancel_last_task_safely(drone)
        await brake_to_stop_by_velocity(
            drone,
            commanded_velocity,
            max(0.05, args.velocity_command_duration_sec),
            max(0.1, args.velocity_acceleration_limit_mps2)
            * max(0.05, args.velocity_command_duration_sec),
        )
        raise
    except Exception as exc:
        failed = True
        projectairsim_log().warning("Auto flight failed: %s", exc)
        projectairsim_log().warning("Auto flight traceback:\n%s", traceback.format_exc())
        cancel_last_task_safely(drone)
        commanded_velocity = await brake_to_stop_by_velocity(
            drone,
            commanded_velocity,
            max(0.05, args.velocity_command_duration_sec),
            max(0.1, args.velocity_acceleration_limit_mps2)
            * max(0.05, args.velocity_command_duration_sec),
        )
        state.set_mode("auto flight failed", str(exc))
    finally:
        state.set_auto_flight_running(False)
        if not state.snapshot()["stop_requested"] and not failed:
            state.set_mode("teleport", "manual teleport mode")


async def run_auto_diagnostic_mission(
    drone: Drone,
    state: DemoState,
    args,
    world: Optional[World] = None,
):
    projectairsim_log().info(
        "Automatic diagnostic mission: %d route points, %.1fs wait at each point",
        len(state.route),
        args.auto_wait_sec,
    )

    for index, point in enumerate(state.route):
        if state.snapshot()["stop_requested"]:
            break

        await apply_teleport(drone, state, index, args, world)
        await settle_and_sample(drone, state, args.auto_wait_sec, world, args.drone_name)
        update_requested_camera_diagnostics(drone, state, args)
        log_mission_snapshot(point.label, index, state.snapshot())

    projectairsim_log().info("")
    projectairsim_log().info("Automatic diagnostic mission complete.")

    if args.hold_open_after_mission:
        projectairsim_log().info("Holding preview open. Press q/esc in preview or Ctrl+C.")
        while not state.snapshot()["stop_requested"]:
            await settle_and_sample(drone, state, 0.5, world, args.drone_name)
    else:
        state.request_stop()


def manual_initial_speed_mps(args) -> float:
    configured = args.manual_speed_mps
    if configured is None:
        configured = args.route_speed_mps
    return max(float(args.manual_min_speed_mps), float(configured))


def print_manual_controls(
    mode_label: str,
    current_speed_mps: float,
    speed_step_mps: float,
):
    print(f"\n--- {mode_label} ---")
    print("W/S: forward/backward")
    print("A/D: left/right")
    print("Up/Down Arrows: up/down altitude")
    print("Left/Right Arrows: yaw left/right")
    print("K/L: increase/decrease speed")
    print("E/R/T: route/custom teleport in the preview window")
    print("Q/Esc: exit")
    print("Manual PX4 runs PX4 takeoff by default; use --no-manual-px4-takeoff to skip")
    print(f"Speed: {current_speed_mps:.1f} m/s")
    print(f"Speed step: {speed_step_mps:.1f} m/s")
    print("Speed cap: unbounded")
    print("----------------------------")


async def run_manual_direct_flight(
    drone: Drone,
    state: DemoState,
    args,
    world: Optional[World] = None,
):
    keyboard_module = load_keyboard_module()
    current_speed_mps = manual_initial_speed_mps(args)
    speed_step_mps = max(0.1, float(args.manual_speed_step_mps))
    min_speed_mps = max(0.0, float(args.manual_min_speed_mps))
    yaw_rate_dps = max(0.0, float(args.manual_yaw_rate_dps))
    speed_adjust_debounce_sec = 0.25
    last_speed_adjust_at = 0.0
    last_report_at = 0.0
    last_tick_at = time.time()

    state.set_mode("manual direct", f"speed {current_speed_mps:.1f} m/s")
    print_manual_controls("Manual Direct Flight", current_speed_mps, speed_step_mps)
    projectairsim_log().info(
        "Manual direct flight active: WASD/arrows fly, K/L speed, q/esc quit"
    )

    while not state.snapshot()["stop_requested"]:
        now = time.time()
        dt = min(max(now - last_tick_at, 0.0), 0.1)
        last_tick_at = now

        if state.check_battery_depleted() or state.snapshot()["coverage_unfeasible"]:
            log_coverage_unfeasible_if_needed(state)
            await hold_then_request_stop(state)
            break

        if keyboard_module.is_pressed("q") or keyboard_module.is_pressed("esc"):
            projectairsim_log().info("Manual direct flight requested exit")
            state.request_stop()
            break

        if keyboard_module.is_pressed("k") and (
            now - last_speed_adjust_at >= speed_adjust_debounce_sec
        ):
            current_speed_mps += speed_step_mps
            state.set_mode("manual direct", f"speed {current_speed_mps:.1f} m/s")
            projectairsim_log().info("Manual speed: %.1f m/s", current_speed_mps)
            last_speed_adjust_at = now
        elif keyboard_module.is_pressed("l") and (
            now - last_speed_adjust_at >= speed_adjust_debounce_sec
        ):
            current_speed_mps = max(min_speed_mps, current_speed_mps - speed_step_mps)
            state.set_mode("manual direct", f"speed {current_speed_mps:.1f} m/s")
            projectairsim_log().info("Manual speed: %.1f m/s", current_speed_mps)
            last_speed_adjust_at = now

        current_ned = get_pose_position_ned(drone)
        yaw_rad = get_pose_yaw_ned(drone)
        target_yaw_rad = yaw_rad
        if keyboard_module.is_pressed("left"):
            target_yaw_rad -= math.radians(yaw_rate_dps) * dt
        elif keyboard_module.is_pressed("right"):
            target_yaw_rad += math.radians(yaw_rate_dps) * dt

        body_velocity = [0.0, 0.0, 0.0]
        if keyboard_module.is_pressed("w"):
            body_velocity[0] = current_speed_mps
        elif keyboard_module.is_pressed("s"):
            body_velocity[0] = -current_speed_mps

        if keyboard_module.is_pressed("a"):
            body_velocity[1] = -current_speed_mps
        elif keyboard_module.is_pressed("d"):
            body_velocity[1] = current_speed_mps

        if keyboard_module.is_pressed("up"):
            body_velocity[2] = -current_speed_mps
        elif keyboard_module.is_pressed("down"):
            body_velocity[2] = current_speed_mps

        world_velocity = [
            body_velocity[0] * math.cos(target_yaw_rad)
            - body_velocity[1] * math.sin(target_yaw_rad),
            body_velocity[0] * math.sin(target_yaw_rad)
            + body_velocity[1] * math.cos(target_yaw_rad),
            body_velocity[2],
        ]
        moved = vector_length(world_velocity) > 1e-6
        yaw_changed = abs(wrap_angle_rad(target_yaw_rad - yaw_rad)) > 1e-6
        if moved or yaw_changed:
            current_ned = [
                current_ned[index] + world_velocity[index] * dt
                for index in range(3)
            ]
            drone.set_pose(
                make_pose_ned_yaw(current_ned, heading_deg_360(target_yaw_rad)),
                reset_kinematics=True,
            )

        state.update_pose(ned_to_route(current_ned), heading_deg_360(target_yaw_rad))

        if now - last_report_at >= args.pose_report_interval_sec:
            projectairsim_log().info(
                "Manual direct NED %s heading %.1f deg speed %.1f m/s command %.1f m/s",
                format_vector3(current_ned),
                heading_deg_360(target_yaw_rad),
                current_speed_mps,
                vector_length(world_velocity),
            )
            last_report_at = now

        await asyncio.sleep(0.02)


async def run_manual_px4_flight(
    drone: Drone,
    state: DemoState,
    args,
    world: Optional[World] = None,
):
    keyboard_module = load_keyboard_module()
    current_speed_mps = manual_initial_speed_mps(args)
    speed_step_mps = max(0.1, float(args.manual_speed_step_mps))
    min_speed_mps = max(0.0, float(args.manual_min_speed_mps))
    command_duration_sec = max(0.05, float(args.manual_command_duration_sec))
    max_velocity_delta = (
        max(0.0, float(args.manual_acceleration_limit_mps2)) * command_duration_sec
    )
    max_yaw_delta = (
        max(0.0, float(args.manual_yaw_acceleration_dps2)) * command_duration_sec
    )
    yaw_rate_dps = max(0.0, float(args.manual_yaw_rate_dps))
    speed_adjust_debounce_sec = 0.25
    last_speed_adjust_at = 0.0
    last_report_at = 0.0
    last_camera_epoch = state.snapshot()["camera_epoch"]
    commanded_velocity = [0.0, 0.0, 0.0]
    commanded_yaw_rate_dps = 0.0
    api_control_enabled = False

    state.set_mode("manual px4", "arming PX4")
    print_manual_controls("Manual PX4 Flight", current_speed_mps, speed_step_mps)
    try:
        if not drone.enable_api_control():
            raise RuntimeError("Project AirSim rejected EnableApiControl for manual PX4 flight")
        api_control_enabled = True
        await arm_with_retry(drone, args.arm_timeout_sec)

        if args.manual_px4_takeoff:
            state.set_mode("manual px4", "taking off")
            projectairsim_log().info("Manual PX4 takeoff requested")
            takeoff_task = await drone.takeoff_async(timeout_sec=args.takeoff_timeout_sec)
            await await_drone_task(
                drone,
                takeoff_task,
                "Manual PX4 takeoff",
                args.takeoff_timeout_sec + 5.0,
                args.pose_report_interval_sec,
            )
        else:
            projectairsim_log().info(
                "Manual PX4 takeoff skipped; velocity commands may not move a landed PX4 vehicle"
            )

        await request_px4_control(drone)
        await asyncio.sleep(max(0.0, args.request_control_settle_sec))
        state.set_mode("manual px4", f"speed {current_speed_mps:.1f} m/s")
        projectairsim_log().info(
            "Manual PX4 flight active: WASD/arrows fly, K/L speed, q/esc quit"
        )

        while not state.snapshot()["stop_requested"]:
            now = time.time()
            snapshot = state.snapshot()
            if snapshot["camera_epoch"] != last_camera_epoch:
                commanded_velocity = [0.0, 0.0, 0.0]
                commanded_yaw_rate_dps = 0.0
                last_camera_epoch = snapshot["camera_epoch"]

            if state.check_battery_depleted() or state.snapshot()["coverage_unfeasible"]:
                log_coverage_unfeasible_if_needed(state)
                await hold_then_request_stop(state)
                break

            if keyboard_module.is_pressed("q") or keyboard_module.is_pressed("esc"):
                projectairsim_log().info("Manual PX4 flight requested exit")
                state.request_stop()
                break

            if keyboard_module.is_pressed("k") and (
                now - last_speed_adjust_at >= speed_adjust_debounce_sec
            ):
                current_speed_mps += speed_step_mps
                state.set_mode("manual px4", f"speed {current_speed_mps:.1f} m/s")
                projectairsim_log().info("Manual speed: %.1f m/s", current_speed_mps)
                last_speed_adjust_at = now
            elif keyboard_module.is_pressed("l") and (
                now - last_speed_adjust_at >= speed_adjust_debounce_sec
            ):
                current_speed_mps = max(min_speed_mps, current_speed_mps - speed_step_mps)
                state.set_mode("manual px4", f"speed {current_speed_mps:.1f} m/s")
                projectairsim_log().info("Manual speed: %.1f m/s", current_speed_mps)
                last_speed_adjust_at = now

            target_velocity = [0.0, 0.0, 0.0]
            target_yaw_rate_dps = 0.0
            if keyboard_module.is_pressed("w"):
                target_velocity[0] = current_speed_mps
            elif keyboard_module.is_pressed("s"):
                target_velocity[0] = -current_speed_mps

            if keyboard_module.is_pressed("a"):
                target_velocity[1] = -current_speed_mps
            elif keyboard_module.is_pressed("d"):
                target_velocity[1] = current_speed_mps

            if keyboard_module.is_pressed("up"):
                target_velocity[2] = -current_speed_mps
            elif keyboard_module.is_pressed("down"):
                target_velocity[2] = current_speed_mps

            if keyboard_module.is_pressed("left"):
                target_yaw_rate_dps = -yaw_rate_dps
            elif keyboard_module.is_pressed("right"):
                target_yaw_rate_dps = yaw_rate_dps

            commanded_velocity = limit_vector_delta(
                commanded_velocity,
                target_velocity,
                max_velocity_delta,
            )
            commanded_yaw_rate_dps = move_scalar_toward(
                commanded_yaw_rate_dps,
                target_yaw_rate_dps,
                max_yaw_delta,
            )

            should_send = (
                vector_length(commanded_velocity) > 0.05
                or vector_length(target_velocity) > 0.0
                or abs(commanded_yaw_rate_dps) > 0.1
                or abs(target_yaw_rate_dps) > 0.0
            )
            if should_send:
                await drone.move_by_velocity_body_frame_async(
                    commanded_velocity[0],
                    commanded_velocity[1],
                    commanded_velocity[2],
                    command_duration_sec,
                    yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
                    yaw_is_rate=True,
                    yaw=math.radians(commanded_yaw_rate_dps),
                )

            if now - last_report_at >= args.pose_report_interval_sec:
                current_ned = get_pose_position_ned(drone)
                actual_velocity = get_ground_truth_velocity_ned(drone)
                state.update_pose(
                    ned_to_route(current_ned),
                    heading_deg_360(get_pose_yaw_ned(drone)),
                )
                projectairsim_log().info(
                    "Manual PX4 NED %s heading %.1f deg speed %.1f m/s commanded %.1f actual %.1f m/s",
                    format_vector3(current_ned),
                    heading_deg_360(get_pose_yaw_ned(drone)),
                    current_speed_mps,
                    vector_length(commanded_velocity),
                    vector_length(actual_velocity),
                )
                last_report_at = now

            await asyncio.sleep(0.02)
    finally:
        cancel_last_task_safely(drone)
        if api_control_enabled:
            with suppress(Exception):
                await brake_to_stop_by_velocity(
                    drone,
                    commanded_velocity,
                    command_duration_sec,
                    max_velocity_delta,
                )
            if not args.manual_keep_armed:
                safe_call("Manual PX4 disarm", drone.disarm)
                safe_call("Manual PX4 disable API control", drone.disable_api_control)


async def run_teleport_viewer(
    drone: Drone,
    state: DemoState,
    args,
    world: Optional[World] = None,
    start_auto_flight: bool = False,
    manual_flight_mode: Optional[str] = None,
):
    initial_ned = get_pose_position_ned(drone)
    initial_route = ned_to_route(initial_ned)
    initial_index = nearest_route_index(state.route, initial_route)
    state.set_route_index(initial_index)
    state.update_pose(initial_route, heading_deg_360(get_pose_yaw_ned(drone)))
    update_drone_diagnostics(drone, state, world, args.drone_name)
    projectairsim_log().info("Interactive mode active. Press e/r/t for teleport controls.")
    last_report_at = 0.0
    last_diagnostics_at = 0.0
    last_inspection_at = 0.0
    auto_flight_task = (
        asyncio.create_task(run_route_auto_flight(drone, state, args, world))
        if start_auto_flight
        else None
    )
    manual_flight_task = None
    if manual_flight_mode == "direct":
        manual_flight_task = asyncio.create_task(
            run_manual_direct_flight(drone, state, args, world)
        )
    elif manual_flight_mode == "px4":
        manual_flight_task = asyncio.create_task(
            run_manual_px4_flight(drone, state, args, world)
        )

    try:
        while not state.snapshot()["stop_requested"]:
            current_ned = get_pose_position_ned(drone)
            current = ned_to_route(current_ned)
            current_heading = heading_deg_360(get_pose_yaw_ned(drone))
            state.update_pose(current, current_heading)
            if state.check_battery_depleted() or state.snapshot()["coverage_unfeasible"]:
                log_coverage_unfeasible_if_needed(state)
                if not state.snapshot()["auto_flight_running"]:
                    await hold_then_request_stop(state)
                    break
            if not state.snapshot()["auto_flight_running"]:
                state.set_route_index(nearest_route_index(state.route, current))

            now = time.time()
            if now - last_inspection_at >= 1.0 / max(0.1, float(args.inspection_update_hz)):
                snapshot = state.snapshot()
                camera_position = snapshot.get("fpv_camera_position") or current_ned
                update_inspection_geometry(world, state, args, camera_position, current_ned)
                last_inspection_at = now

            if now - last_diagnostics_at >= 0.5:
                update_drone_diagnostics(drone, state, world, args.drone_name)
                last_diagnostics_at = now

            if auto_flight_task is not None and auto_flight_task.done():
                await auto_flight_task
                auto_flight_task = None
            if manual_flight_task is not None and manual_flight_task.done():
                await manual_flight_task
                manual_flight_task = None
                if not state.snapshot()["stop_requested"]:
                    state.request_stop()

            direction = drain_teleport_request(state)
            if direction is not None:
                if state.snapshot()["auto_flight_running"]:
                    projectairsim_log().info("Ignoring teleport key while auto flight is running")
                else:
                    base_index = nearest_route_index(state.route, current)
                    target_index = (base_index + direction) % len(state.route)
                    projectairsim_log().info(
                        "Route key %s from nearest %s index=%d to %s index=%d",
                        "next" if direction > 0 else "previous",
                        state.route[base_index].label,
                        base_index,
                        state.route[target_index].label,
                        target_index,
                    )
                    await apply_teleport(drone, state, target_index, args, world)
                    current_ned = get_pose_position_ned(drone)
                    current = ned_to_route(current_ned)
                    current_heading = heading_deg_360(get_pose_yaw_ned(drone))
                    state.update_pose(current, current_heading)
            if drain_manual_teleport_request(state):
                if state.snapshot()["auto_flight_running"]:
                    projectairsim_log().info("Ignoring custom teleport while auto flight is running")
                else:
                    await prompt_manual_teleport(drone, state, args, world)

            if now - last_report_at >= args.pose_report_interval_sec:
                projectairsim_log().info(
                    "Viewer NED %s heading %.1f deg mode=%s",
                    format_vector3(current),
                    current_heading,
                    state.snapshot()["mode"],
                )
                log_diagnostics(state.snapshot())
                last_report_at = now

            await asyncio.sleep(0.05)
    finally:
        if auto_flight_task is not None and not auto_flight_task.done():
            cancel_last_task_safely(drone)
            auto_flight_task.cancel()
            with suppress(asyncio.CancelledError):
                await auto_flight_task
        if manual_flight_task is not None and not manual_flight_task.done():
            cancel_last_task_safely(drone)
            manual_flight_task.cancel()
            with suppress(asyncio.CancelledError):
                await manual_flight_task


async def run_demo(args):
    route = route_from_constants()
    state = None
    temp_config_dir = None
    world = None
    drone = None
    preview = None
    wind_field = None
    wind_task = None
    client = ProjectAirSimClient(
        address=args.server_ip,
        port_topics=args.topics_port,
        port_services=args.services_port,
    )

    try:
        mode_flight = "auto-diagnostic" if args.auto_diagnostic else args.mode_flight
        mode_flight = {
            "direct": "manual-direct",
            "px4": "manual-px4",
        }.get(mode_flight, mode_flight)
        temp_config_dir, scene_name, sim_config_path = make_runtime_scene_config(args, route)

        projectairsim_log().info("Connecting to Project AirSim")
        client.connect()
        world = World(
            client,
            scene_name,
            delay_after_load_sec=args.load_delay_sec,
            sim_config_path=sim_config_path,
        )
        projectairsim_log().info(
            "Loaded %s. If PX4 was already waiting on TCP port 4560 before this "
            "scene loaded, restart PX4 now and let this script keep waiting.",
            scene_name,
        )

        drone = Drone(client, world, args.drone_name)
        fpv_topic = drone.sensors.get(args.fpv_camera, {}).get("scene_camera")
        chase_topic = drone.sensors.get(args.chase_camera, {}).get("scene_camera")
        if not fpv_topic or not chase_topic:
            available_topics = {
                sensor: sorted(topics.keys())
                for sensor, topics in drone.sensors.items()
                if topics
            }
            raise RuntimeError(
                "FPV or chase camera topic is not available. Available sensor topics: "
                f"{available_topics}"
            )

        if mode_flight == "auto-flight":
            if args.replan_on_object:
                projectairsim_log().warning(
                    "--replan-on-object is ignored in offshore_demo; "
                    "auto-flight uses OFFSHORE_ROUTE exactly as listed."
                )
            projectairsim_log().info(
                "Obstacle avoidance disabled; using %d OFFSHORE_ROUTE points unchanged",
                len(route),
            )

        state = DemoState(route, battery_start_percent=args.battery_start_percent)
        if args.wind:
            state.set_wind_enabled(True, "initializing")
            wind_field = create_wrf_wind_field(world, args)
            wind_task = asyncio.create_task(
                run_wrf_wind_updater(drone, world, state, wind_field, args)
            )
        preview = OffshorePreview(
            state,
            args.window_name,
            args.preview_width,
            args.preview_height,
            args.pip_scale,
            args.preview_fps,
            record_video=args.video,
            video_dir=Path(args.video_dir),
            video_fps=args.video_fps,
            coordinate_diagnostics=args.coordinate_diagnostics,
            front_camera_stabilized=args.gimbal,
            route_overlay=full_route_from_constants(),
            camera_fov_degrees=args.camera_fov_degrees,
            inspection_region_width_m=args.inspection_region_width_m,
            inspection_region_height_m=args.inspection_region_height_m,
            inspection_good_footprint_px=args.inspection_good_footprint_px,
            inspection_frame_margin_fraction=args.inspection_frame_margin_fraction,
            inspection_sharpness_min_variance=args.inspection_sharpness_min_variance,
            inspection_sharpness_good_variance=args.inspection_sharpness_good_variance,
            inspection_weights=args.inspection_confidence_weights,
            inspection_pass_threshold=args.inspection_pass_threshold,
        )
        preview.start()
        client.subscribe(fpv_topic, preview.receive_fpv)
        client.subscribe(chase_topic, preview.receive_chase)
        client.subscribe(
            drone.robot_info["actual_pose"],
            lambda _, pose_msg: update_pose_topic_diagnostic(state, pose_msg),
        )
        projectairsim_log().info(
            "Preview opened. Manual modes use WASD/arrows and K/L; e/r/t teleport; q/esc quits."
        )

        if mode_flight == "manual-direct":
            projectairsim_log().info("Manual direct mode: skipping PX4 readiness wait")
        else:
            await wait_for_px4_ready(drone, args.px4_ready_timeout_sec)
        initial_pose_ned = get_pose_position_ned(drone)
        initial_pose_route = ned_to_route(initial_pose_ned)
        state.update_pose(initial_pose_route, heading_deg_360(get_pose_yaw_ned(drone)))
        projectairsim_log().info(
            "Initial NED %s",
            format_vector3(initial_pose_route),
        )

        if mode_flight == "auto-diagnostic":
            state.set_mode("auto diagnostic", "automatic teleport snapshots")
            await run_auto_diagnostic_mission(drone, state, args, world)
        elif mode_flight == "auto-flight":
            state.set_mode("auto flight", f"speed {args.route_speed_mps:.1f} m/s")
            await run_teleport_viewer(
                drone,
                state,
                args,
                world,
                start_auto_flight=True,
            )
        elif mode_flight == "manual-direct":
            state.set_mode(
                "manual direct",
                f"speed {manual_initial_speed_mps(args):.1f} m/s",
            )
            await run_teleport_viewer(
                drone,
                state,
                args,
                world,
                manual_flight_mode="direct",
            )
        elif mode_flight == "manual-px4":
            state.set_mode(
                "manual px4",
                f"speed {manual_initial_speed_mps(args):.1f} m/s",
            )
            await run_teleport_viewer(
                drone,
                state,
                args,
                world,
                manual_flight_mode="px4",
            )
        else:
            state.set_mode("teleport", "manual teleport mode")
            await run_teleport_viewer(drone, state, args, world)
    finally:
        if state is not None:
            state.request_stop()
        if wind_task is not None:
            wind_task.cancel()
            with suppress(asyncio.CancelledError):
                await wind_task
        if args.wind and world is not None:
            try:
                world.set_wind_velocity(0.0, 0.0, 0.0)
            except Exception:
                pass
        if wind_field is not None:
            wind_field.close()
        if preview is not None:
            preview.stop()
        if preview is not None and preview.error is not None:
            projectairsim_log().warning("Preview stopped with error: %s", preview.error)
        try:
            client.unsubscribe_all()
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass
        if temp_config_dir is not None:
            temp_config_dir.cleanup()


def build_parser():
    parser = argparse.ArgumentParser(description="PX4 offshore wind farm route demo.")
    parser.add_argument("--scene", default="scene_px4_sitl.jsonc")
    parser.add_argument(
        "--sim-config-path",
        default=str(EXAMPLE_SCRIPTS_DIR / "sim_config"),
        help="Directory containing Project AirSim scene/robot configs.",
    )
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--topics-port", type=int, default=8989)
    parser.add_argument("--services-port", type=int, default=8990)
    parser.add_argument("--load-delay-sec", type=float, default=2.0)
    parser.add_argument(
        "--mode-flight",
        choices=[
            "teleport",
            "auto-flight",
            "auto-diagnostic",
            "manual-direct",
            "manual-px4",
            "direct",
            "px4",
        ],
        default="teleport",
        help=(
            "Startup mode: manual teleport viewer, immediate PX4 route flight, "
            "teleport diagnostics, direct manual flight, or PX4 manual flight."
        ),
    )
    parser.set_defaults(replan_on_object=False)
    parser.add_argument(
        "--replan-on-object",
        dest="replan_on_object",
        action="store_true",
        help="Legacy no-op. Auto-flight now uses OFFSHORE_ROUTE exactly as listed.",
    )
    parser.add_argument(
        "--no-replan-on-object",
        dest="replan_on_object",
        action="store_false",
        help="Legacy no-op. Auto-flight already uses OFFSHORE_ROUTE unchanged.",
    )
    parser.add_argument(
        "--log-obstacle-scans",
        action="store_true",
        help="Print detailed voxel scan messages for each route leg.",
    )
    parser.add_argument(
        "--obstacle-planner",
        choices=["lateral", "astar"],
        default="lateral",
        help="Obstacle bypass planner: compact lateral detours or full A* leg expansion.",
    )
    parser.add_argument(
        "--ignore-actor",
        action="append",
        default=[],
        help="Actor to ignore in obstacle voxel grids. Repeatable.",
    )
    parser.add_argument(
        "--ground-z-ned",
        type=float,
        default=200.0,
        help="Ground/down value used by A* validity checks for offshore planning.",
    )
    parser.add_argument(
        "--object-path-clearance-m",
        type=float,
        default=15.0,
        help="Horizontal route corridor radius that counts as blocked by objects.",
    )
    parser.add_argument(
        "--object-scan-resolution-m",
        type=float,
        default=5.0,
        help="Voxel-grid resolution for obstacle route scanning and A* bypasses.",
    )
    parser.add_argument(
        "--object-scan-sample-spacing-m",
        type=float,
        default=5.0,
        help="Spacing between route samples checked for obstacle occupancy.",
    )
    parser.add_argument(
        "--object-scan-margin-m",
        type=float,
        default=80.0,
        help="Extra meters around each route leg when creating the obstacle scan grid.",
    )
    parser.add_argument(
        "--object-scan-min-size-m",
        type=float,
        default=40.0,
        help="Minimum x/y/z size for each obstacle scan grid.",
    )
    parser.add_argument(
        "--object-scan-start-ignore-m",
        type=float,
        default=5.0,
        help="Ignore occupied route samples this close to the start of a leg.",
    )
    parser.add_argument(
        "--object-stop-distance-m",
        type=float,
        default=20.0,
        help="Distance before a blocked sample used for diagnostic obstacle reporting.",
    )
    parser.add_argument(
        "--replan-waypoint-spacing-m",
        type=float,
        default=15.0,
        help="Minimum spacing between generated A* bypass waypoints.",
    )
    parser.add_argument(
        "--stop-at-bypass-points",
        dest="stop_at_bypass_points",
        action="store_true",
        default=True,
        help="Brake at generated obstacle bypass points before continuing. This is the default.",
    )
    parser.add_argument(
        "--pass-through-bypass-points",
        dest="stop_at_bypass_points",
        action="store_false",
        help="Treat generated obstacle bypass points as pass-through route points.",
    )
    parser.add_argument(
        "--bypass-lateral-offset-m",
        type=float,
        default=180.0,
        help="First side offset tried by the compact lateral obstacle bypass planner.",
    )
    parser.add_argument(
        "--bypass-max-lateral-offset-m",
        type=float,
        default=300.0,
        help="Largest side offset tried by the compact lateral obstacle bypass planner.",
    )
    parser.add_argument(
        "--bypass-lateral-step-m",
        type=float,
        default=40.0,
        help="Step between lateral bypass offsets while searching for a clear corridor.",
    )
    parser.add_argument(
        "--bypass-along-distance-m",
        type=float,
        default=70.0,
        help="Distance before/after the detected obstacle used by the lateral bypass.",
    )
    parser.add_argument(
        "--bypass-vertical-margin-m",
        type=float,
        default=5.0,
        help="Maximum extra NED-z margin allowed for generated obstacle bypass points.",
    )
    parser.add_argument(
        "--replan-endpoint-search-radius-m",
        type=float,
        default=80.0,
        help="Horizontal radius used to snap an occupied A* start/target to nearby free space.",
    )
    parser.add_argument(
        "--replan-endpoint-vertical-search-m",
        type=float,
        default=40.0,
        help="Vertical radius used to snap an occupied A* start/target to nearby free space.",
    )
    parser.add_argument(
        "--auto-diagnostic",
        action="store_true",
        help="Compatibility alias for --mode-flight auto-diagnostic.",
    )
    parser.add_argument(
        "--auto-wait-sec",
        type=float,
        default=5.0,
        help="Seconds to wait after each automatic teleport before logging diagnostics.",
    )
    parser.add_argument(
        "--hold-open-after-mission",
        action="store_true",
        help="Keep the preview open after the automatic diagnostic route completes.",
    )
    parser.add_argument("--px4-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument("--arm-timeout-sec", type=float, default=60.0)
    parser.add_argument(
        "--route-speed-mps",
        type=float,
        default=12.0,
        help="PX4 auto-flight speed.",
    )
    parser.add_argument(
        "--manual-speed-mps",
        type=float,
        default=None,
        help="Initial manual flight speed. Defaults to --route-speed-mps.",
    )
    parser.add_argument(
        "--manual-speed-step-mps",
        type=float,
        default=5.0,
        help="Manual speed increment/decrement for K/L.",
    )
    parser.add_argument(
        "--manual-min-speed-mps",
        type=float,
        default=0.0,
        help="Minimum manual speed after pressing L.",
    )
    parser.add_argument(
        "--manual-yaw-rate-dps",
        type=float,
        default=60.0,
        help="Manual yaw rate for left/right arrow keys.",
    )
    parser.add_argument(
        "--manual-command-duration-sec",
        type=float,
        default=0.1,
        help="Duration of each manual PX4 velocity command.",
    )
    parser.add_argument(
        "--manual-acceleration-limit-mps2",
        type=float,
        default=20.0,
        help="Manual PX4 velocity change rate. Higher values make K/L speed changes take effect faster.",
    )
    parser.add_argument(
        "--manual-yaw-acceleration-dps2",
        type=float,
        default=180.0,
        help="Manual PX4 yaw-rate change rate.",
    )
    parser.add_argument(
        "--manual-px4-takeoff",
        dest="manual_px4_takeoff",
        action="store_true",
        default=True,
        help="Run a PX4 takeoff before manual PX4 control.",
    )
    parser.add_argument(
        "--no-manual-px4-takeoff",
        dest="manual_px4_takeoff",
        action="store_false",
        help="Skip PX4 takeoff before manual PX4 control.",
    )
    parser.add_argument(
        "--manual-keep-armed",
        action="store_true",
        help="Leave PX4 armed/API-control enabled after manual PX4 exits.",
    )
    parser.add_argument(
        "--manual-px4-param-speed-limit-mps",
        type=float,
        default=250.0,
        help="Runtime PX4 horizontal speed parameter used by manual-px4.",
    )
    parser.add_argument(
        "--manual-px4-param-vertical-speed-limit-mps",
        type=float,
        default=80.0,
        help="Runtime PX4 vertical speed parameter used by manual-px4.",
    )
    parser.add_argument(
        "--route-acceptance-m",
        type=float,
        default=3.0,
        help="Distance from a waypoint that counts as reached during velocity route flight.",
    )
    parser.add_argument(
        "--route-move-timeout-sec",
        type=float,
        default=0.0,
        help="Per-leg timeout. Use 0 to infer from leg distance and speed.",
    )
    parser.add_argument(
        "--route-timeout-multiplier",
        type=float,
        default=10.0,
        help="Multiplier for inferred per-leg timeout when route move timeout is 0.",
    )
    parser.add_argument(
        "--post-teleport-flight-delay-sec",
        type=float,
        default=1.0,
        help="Seconds to wait after teleporting to Origin before starting PX4 flight.",
    )
    parser.add_argument("--takeoff-timeout-sec", type=float, default=20.0)
    parser.add_argument(
        "--skip-takeoff-before-route",
        action="store_true",
        default=False,
        help="Skip PX4 takeoff before auto-flight route movement.",
    )
    parser.add_argument(
        "--takeoff-before-route",
        dest="skip_takeoff_before_route",
        action="store_false",
        help="Run PX4 takeoff before auto-flight route movement. This is the default.",
    )
    parser.add_argument(
        "--post-takeoff-origin-reset-delay-sec",
        type=float,
        default=0.75,
        help="Seconds to settle after takeoff and the automatic reset back to Origin.",
    )
    parser.add_argument(
        "--request-control-settle-sec",
        type=float,
        default=0.5,
        help="Small delay after PX4 RequestControl before sending route velocity commands.",
    )
    parser.add_argument(
        "--pose-report-interval-sec",
        type=float,
        default=1.0,
        help="Seconds between terminal pose/flight progress reports.",
    )
    parser.add_argument(
        "--velocity-command-duration-sec",
        type=float,
        default=0.1,
        help="Duration of each PX4 velocity setpoint in velocity route mode.",
    )
    parser.add_argument(
        "--velocity-acceleration-limit-mps2",
        type=float,
        default=4.0,
        help="Maximum velocity change rate in velocity route mode.",
    )
    parser.add_argument(
        "--velocity-lookahead-m",
        type=float,
        default=8.0,
        help=(
            "Distance ahead on the current route segment used by velocity route "
            "mode. This prevents side-to-side waypoint-center chasing."
        ),
    )
    parser.add_argument(
        "--route-vertical-speed-mps",
        type=float,
        default=6.0,
        help="Maximum up/down velocity command during route flight. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-route-deviation-m",
        type=float,
        default=180.0,
        help="Abort auto-flight if ground-truth pose drifts this far from the active route segment. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-target-overshoot-m",
        type=float,
        default=35.0,
        help="Abort auto-flight if the drone passes this far beyond the active target. Use 0 to disable.",
    )
    parser.add_argument(
        "--battery-start-percent",
        type=float,
        default=100.0,
        help="Battery percentage shown when the mission battery timer starts.",
    )
    parser.add_argument(
        "--route-stuck-timeout-sec",
        type=float,
        default=6.0,
        help="Abort if AirSim ground truth does not move after this many seconds of route commands. Use 0 to disable.",
    )
    parser.add_argument(
        "--route-stuck-distance-m",
        type=float,
        default=1.0,
        help="Minimum AirSim ground-truth movement expected before the stuck timeout.",
    )
    parser.add_argument(
        "--slowdown-distance-m",
        type=float,
        default=30.0,
        help="Distance over which velocity route mode eases down near each waypoint.",
    )
    parser.add_argument(
        "--path-yaw-rate-dps",
        type=float,
        default=25.0,
        help="Maximum yaw rate used while facing the planned route segment.",
    )
    parser.add_argument(
        "--path-yaw-deadband-deg",
        type=float,
        default=2.0,
        help="Yaw error ignored while facing the planned route segment.",
    )
    parser.add_argument(
        "--path-yaw-response-sec",
        type=float,
        default=0.8,
        help="Seconds over which route-facing yaw tries to close heading error.",
    )
    parser.add_argument(
        "--travel-yaw-timeout-sec",
        type=float,
        default=8.0,
        help="Maximum time spent turning toward the next route segment before moving.",
    )
    parser.add_argument(
        "--align-yaw-at-waypoints",
        dest="align_yaw_at_waypoints",
        action="store_true",
        default=True,
        help="After each named waypoint is reached, rotate to that waypoint yaw. This is the default.",
    )
    parser.add_argument(
        "--no-align-yaw-at-waypoints",
        dest="align_yaw_at_waypoints",
        action="store_false",
        help="Do not stop to align yaw at named waypoints.",
    )
    parser.add_argument(
        "--waypoint-yaw-acceptance-deg",
        type=float,
        default=3.0,
        help="Yaw error accepted after arriving at a named waypoint.",
    )
    parser.add_argument(
        "--waypoint-yaw-rate-dps",
        type=float,
        default=5.0,
        help="Maximum yaw rate while stopped at a named waypoint.",
    )
    parser.add_argument(
        "--waypoint-yaw-timeout-sec",
        type=float,
        default=20.0,
        help="Maximum time spent aligning yaw at a named waypoint.",
    )
    parser.add_argument(
        "--face-travel-direction",
        dest="face_travel_direction",
        action="store_true",
        default=True,
        help="Yaw toward the current travel direction during auto flight.",
    )
    parser.add_argument(
        "--no-face-travel-direction",
        dest="face_travel_direction",
        action="store_false",
        help="Legacy accepted flag; offshore auto-flight still faces the active route segment.",
    )
    parser.add_argument("--fpv-camera", default="FrontCamera")
    parser.add_argument("--chase-camera", default="Chase")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov-degrees", type=float, default=90.0)
    parser.add_argument("--camera-capture-interval-sec", type=float, default=0.03)
    parser.add_argument(
        "--gimbal",
        action="store_true",
        help="Enable built-in roll/pitch stabilization for the runtime FrontCamera.",
    )
    parser.add_argument("--window-name", default="Offshore PX4 Demo")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=720)
    parser.add_argument("--preview-fps", type=float, default=30.0)
    parser.add_argument("--pip-scale", type=float, default=0.30)
    parser.add_argument(
        "--coordinate-diagnostics",
        action="store_true",
        help="Draw the coordinate diagnostics table in the FPV preview.",
    )
    parser.add_argument(
        "--inspection-radius-m",
        type=float,
        default=INSPECTION_DEFAULT_RADIUS_M,
        help="Distance from FrontCamera to Blade1_Object1 that enables the inspection panel.",
    )
    parser.add_argument(
        "--inspection-update-hz",
        type=float,
        default=10.0,
        help="How often to update inspection distance/angle geometry.",
    )
    parser.add_argument(
        "--inspection-pass-threshold",
        type=float,
        default=INSPECTION_DEFAULT_PASS_THRESHOLD,
        help="Rule-based inspection confidence score required for PASS.",
    )
    parser.add_argument(
        "--inspection-confidence-weights",
        type=parse_inspection_confidence_weights,
        default=INSPECTION_DEFAULT_WEIGHTS,
        help=(
            "Four weights for distance,angle,visual,stability. "
            "Example: 0.20,0.25,0.30,0.25"
        ),
    )
    parser.add_argument(
        "--inspection-region-width-m",
        type=float,
        default=INSPECTION_REGION_WIDTH_M,
        help="Physical width of the inspection target region used for visual footprint scoring.",
    )
    parser.add_argument(
        "--inspection-region-height-m",
        type=float,
        default=INSPECTION_REGION_HEIGHT_M,
        help="Physical height of the inspection target region used for visual footprint scoring.",
    )
    parser.add_argument(
        "--inspection-good-footprint-px",
        type=float,
        default=INSPECTION_GOOD_FOOTPRINT_PX,
        help="Projected target width/height in pixels that counts as a full apparent-size score.",
    )
    parser.add_argument(
        "--inspection-frame-margin-fraction",
        type=float,
        default=INSPECTION_FRAME_COMFORT_MARGIN_FRACTION,
        help="Fraction of the smaller image dimension used as the comfortable in-frame margin.",
    )
    parser.add_argument(
        "--inspection-sharpness-min-variance",
        type=float,
        default=INSPECTION_SHARPNESS_MIN_VARIANCE,
        help="Laplacian variance that maps to sharpness score 0.",
    )
    parser.add_argument(
        "--inspection-sharpness-good-variance",
        type=float,
        default=INSPECTION_SHARPNESS_GOOD_VARIANCE,
        help="Laplacian variance that maps to sharpness score 1.",
    )
    parser.add_argument(
        "--wind",
        action="store_true",
        help="Enable spatially varying WRF wind and show wind telemetry in the FPV overlay.",
    )
    parser.add_argument(
        "--wrf-file",
        default=str(DEFAULT_WRF_FILE),
        help="WRF NetCDF file used by --wind.",
    )
    parser.add_argument(
        "--wrf-origin-lat",
        type=float,
        default=None,
        help=(
            "WRF latitude mapped to simulation NED [0,0,0]. "
            "Defaults to an auto-selected offshore LANDMASK=0 WRF cell."
        ),
    )
    parser.add_argument(
        "--wrf-origin-lon",
        type=float,
        default=None,
        help=(
            "WRF longitude mapped to simulation NED [0,0,0]. "
            "Defaults to an auto-selected offshore LANDMASK=0 WRF cell."
        ),
    )
    parser.add_argument(
        "--wrf-time-index",
        type=int,
        default=0,
        help="Time index to read from the WRF file.",
    )
    parser.add_argument(
        "--wind-update-hz",
        type=float,
        default=5.0,
        help="How often to sample WRF wind and update Project AirSim wind.",
    )
    parser.add_argument(
        "--wrf-region-half-size-m",
        type=float,
        default=1000.0,
        help="Half-size of the intended local wind-farm region around NED origin.",
    )
    parser.add_argument(
        "--wrf-altitude-min-agl-m",
        type=float,
        default=0.0,
        help="Lower AGL bound for startup/debug wind-region reporting.",
    )
    parser.add_argument(
        "--wrf-altitude-max-agl-m",
        type=float,
        default=100.0,
        help="Upper AGL bound for startup/debug wind-region reporting.",
    )
    parser.add_argument(
        "--wrf-debug",
        action="store_true",
        help="Print detailed WRF wind samples about once per second.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record the OpenCV preview and save it when q/esc is pressed or auto-flight completes.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(VIDEO_DIR),
        help="Directory for --video recordings.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0.0,
        help="Video FPS for --video. Use 0 to match --preview-fps.",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    asyncio.run(run_demo(parsed_args))
