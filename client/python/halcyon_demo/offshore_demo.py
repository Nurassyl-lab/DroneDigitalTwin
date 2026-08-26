"""
Offshore wind farm PX4 demo.

Run this while the OffShoreWindFarm Unreal map is playing. The script loads a
PX4 scene, spawns Drone1 at OFFSHORE_ROUTE[0], and shows FPV with a
chase-camera picture-in-picture. E/R teleport between the route points. Use
--mode-flight auto-flight to teleport to Origin and start PX4 route flight.

Controls in the OpenCV preview:
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


OFFSHORE_ROUTE = [
    ("Origin", [0.0, 148.0, -2.0], 270.0),
    ("Waypoint 1", [1.28, -14.66, -115.16], 127.6),
    ("Waypoint 2", [893.69, 7.89, -112.77], 324.3),
    ("Waypoint 3", [899.44, -892.28, -12.28], 279.5),
    ("Waypoint 4", [-3.35, -907.1, -114.81], 55.2),
    ("Waypoint 5", [-894.61, -901.37, -114.0], 159.9),
    ("Waypoint 6", [-899.67, -5.87, -7.82], 111.0),
    ("Return Origin", [0.0, 148.0, -2.0], 90.0),
]

TELEPORT_WARN_ERROR_M = 5.0
FPV_MAX_CAMERA_OFFSET_M = 8.0
CHASE_MAX_CAMERA_OFFSET_M = 13.0
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
        return [
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None


def extract_pose_yaw_deg(pose_or_kinematics) -> Optional[float]:
    if not isinstance(pose_or_kinematics, dict):
        return None
    pose = pose_or_kinematics.get("pose", pose_or_kinematics)
    if not isinstance(pose, dict):
        return None
    return rotation_yaw_deg(pose.get("orientation") or pose.get("rotation"))


def image_pose_position(image) -> Optional[List[float]]:
    if not isinstance(image, dict):
        return None
    keys = ("pos_x", "pos_y", "pos_z")
    if not all(key in image for key in keys):
        return None
    try:
        return [float(image["pos_x"]), float(image["pos_y"]), float(image["pos_z"])]
    except (TypeError, ValueError):
        return None


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
    def __init__(self, route: Sequence[RoutePoint]):
        self.route = list(route)
        self.current_index = 0
        self.target_index = 1
        self.position = list(self.route[0].position)
        self.heading_deg = self.route[0].yaw_deg
        self.last_requested_label = "Scene origin"
        self.last_requested_position = list(self.route[0].position)
        self.last_requested_yaw_deg = self.route[0].yaw_deg
        self.diagnostics = {}
        self.mode = "teleport"
        self.mode_detail = "manual teleport mode"
        self.auto_flight_running = False
        self.camera_epoch = 0
        self.teleport_requests = queue.SimpleQueue()
        self.manual_teleport_requests = queue.SimpleQueue()
        self.stop_requested = False
        self.lock = Lock()

    def snapshot(self):
        with self.lock:
            return {
                "current_index": self.current_index,
                "target_index": self.target_index,
                "position": list(self.position),
                "heading_deg": self.heading_deg,
                "last_requested_label": self.last_requested_label,
                "last_requested_position": list(self.last_requested_position),
                "last_requested_yaw_deg": self.last_requested_yaw_deg,
                "diagnostics": dict(self.diagnostics),
                "mode": self.mode,
                "mode_detail": self.mode_detail,
                "auto_flight_running": self.auto_flight_running,
                "camera_epoch": self.camera_epoch,
                "stop_requested": self.stop_requested,
            }

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

    def bump_camera_epoch(self):
        with self.lock:
            self.camera_epoch += 1

    def mark_reached(self, index: int):
        with self.lock:
            self.current_index = index
            self.target_index = min(index + 1, len(self.route) - 1)

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
    ):
        self.state = state
        self.window_name = window_name
        self.width = width
        self.height = height
        self.pip_scale = max(0.1, min(0.5, pip_scale))
        self.max_fps = max(1.0, max_fps)
        self.fpv_images = queue.SimpleQueue()
        self.chase_images = queue.SimpleQueue()
        self.running = False
        self.thread = None
        self.error = None
        self.last_key_at = {}
        self.key_debounce_sec = 0.25
        self.camera_epoch_seen = state.snapshot()["camera_epoch"]
        self.waiting_for_fresh_fpv = False

        xs = [point.position[0] for point in self.state.route]
        ys = [point.position[1] for point in self.state.route]
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
        self.thread = None

    def receive_fpv(self, _, image):
        self.state.update_image_diagnostic("FPV camera msg", image)
        if not self.camera_frame_is_near_drone(
            image,
            FPV_MAX_CAMERA_OFFSET_M,
            "FPV camera msg",
        ):
            return
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
                        self.draw_diagnostics(cv2, frame)
                        if not created:
                            cv2.namedWindow(
                                self.window_name,
                                flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE,
                            )
                            created = True
                        cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(1)
                    self.handle_key(key)
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
                self.draw_diagnostics(cv2, frame)

                if not created:
                    cv2.namedWindow(
                        self.window_name,
                        flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE,
                    )
                    created = True

                cv2.imshow(self.window_name, frame)
                self.handle_key(cv2.waitKey(1))
                next_frame_at = time.monotonic() + frame_interval_sec
        except Exception as exc:
            self.error = exc
            self.state.request_stop()
        finally:
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

    def draw_chase_pip(self, cv2, frame, chase):
        height, width = frame.shape[:2]
        pip_w = int(width * self.pip_scale)
        pip_h = int(pip_w * 9 / 16)
        pip_h = min(pip_h, int(height * 0.35))
        pip = cv2.resize(chase, (pip_w, pip_h))
        x0 = width - pip_w - 16
        y0 = 16
        frame[y0 : y0 + pip_h, x0 : x0 + pip_w] = pip
        cv2.rectangle(frame, (x0, y0), (x0 + pip_w, y0 + pip_h), (255, 255, 255), 2)
        self.draw_text(cv2, frame, "Chase", (x0 + 8, y0 + 22), scale=0.55)

    def draw_route_overlay(self, cv2, frame):
        origin_x = 18
        origin_y = 56
        map_w = 300
        map_h = 300
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
        self.draw_text(cv2, frame, "Route", (origin_x, origin_y - 8), scale=0.55)

        route_pixels = [to_px(point.position) for point in self.state.route]
        for start, end in zip(route_pixels, route_pixels[1:]):
            cv2.line(frame, start, end, (255, 0, 0), 2, cv2.LINE_AA)

        snapshot = self.state.snapshot()
        for index, point in enumerate(self.state.route):
            color = (0, 0, 255)
            radius = 6
            if index == snapshot["current_index"]:
                color = (0, 210, 255)
                radius = 8
            elif index == snapshot["target_index"]:
                color = (0, 255, 255)
                radius = 8
            cv2.circle(frame, to_px(point.position), radius, color, -1, cv2.LINE_AA)

        pos_px = to_px(snapshot["position"])
        cv2.drawMarker(
            frame,
            pos_px,
            (80, 255, 80),
            markerType=cv2.MARKER_TRIANGLE_UP,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    def draw_status(self, cv2, frame):
        snapshot = self.state.snapshot()
        target = self.state.route[snapshot["target_index"]]
        position = snapshot["position"]
        distance = distance_between(position, target.position)
        lines = [
            f"Target: {target.label}  {distance:.1f} m",
            f"NED: x={position[0]:.1f} y={position[1]:.1f} z={position[2]:.1f}",
            f"Heading: {snapshot['heading_deg']:.1f} deg  Mode: {snapshot['mode']}",
            f"{snapshot['mode_detail']} | e/r teleport | t custom | q/esc quit",
        ]
        y = frame.shape[0] - 92
        for line in lines:
            self.draw_text(cv2, frame, line, (18, y), scale=0.55)
            y += 22

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
    def draw_text(cv2, frame, text: str, origin: Tuple[int, int], scale: float = 0.6):
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
            (255, 255, 255),
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
):
    return {
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


def ensure_camera(
    robot_config: Dict,
    camera_id: str,
    width: int,
    height: int,
    fov_degrees: float,
    capture_interval_sec: float,
):
    sensors = robot_config.setdefault("sensors", [])
    sensor = next((candidate for candidate in sensors if candidate.get("id") == camera_id), None)
    if sensor is None:
        if camera_id != "FrontCamera":
            raise RuntimeError(f"Camera '{camera_id}' is not present in the robot config")
        sensors.append(
            default_front_camera_sensor(width, height, fov_degrees, capture_interval_sec)
        )
        projectairsim_log().info("Runtime config added FrontCamera for FPV")
        return
    if sensor.get("type") != "camera":
        raise RuntimeError(f"Sensor '{camera_id}' exists but is not a camera")
    ensure_scene_camera_capture(sensor, width, height, fov_degrees, capture_interval_sec)


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
                )
                ensure_camera(
                    robot_config,
                    args.chase_camera,
                    args.camera_width,
                    args.camera_height,
                    args.camera_fov_degrees,
                    args.camera_capture_interval_sec,
                )

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


def route_from_constants() -> List[RoutePoint]:
    return [
        RoutePoint(label, [float(pos[0]), float(pos[1]), float(pos[2])], float(yaw_deg))
        for label, pos, yaw_deg in OFFSHORE_ROUTE
    ]


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
    with state.lock:
        state.current_index = index
        state.target_index = min(index + 1, len(state.route) - 1)
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
            projectairsim_log().info(
                "%s following segment to %s; pose NED %s; remaining %.2f m",
                label,
                format_vector3(target),
                format_vector3(current),
                distance,
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
            speed_scale = min(1.0, max(0.05, distance / max(slowdown_distance_m, 1e-6)))
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

    command_duration_sec = max(0.05, args.velocity_command_duration_sec)
    max_yaw_rate_radps = math.radians(max(0.0, args.waypoint_yaw_rate_dps))
    if max_yaw_rate_radps <= 0.0:
        return

    target_yaw_rad = math.radians(float(target_yaw_deg) % 360.0)
    yaw_acceptance_rad = math.radians(max(0.0, args.waypoint_yaw_acceptance_deg))
    yaw_response_sec = max(command_duration_sec, args.path_yaw_response_sec)
    started_at = time.time()
    last_report_at = 0.0

    while not state.snapshot()["stop_requested"]:
        current_ned = get_pose_position_ned(drone)
        current_yaw_rad = get_pose_yaw_ned(drone)
        yaw_error = wrap_angle_rad(target_yaw_rad - current_yaw_rad)
        state.update_pose(ned_to_route(current_ned), heading_deg_360(current_yaw_rad))

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
        if args.waypoint_yaw_timeout_sec > 0.0 and elapsed > args.waypoint_yaw_timeout_sec:
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
            current_ned = get_pose_position_ned(drone)
            state.update_pose(ned_to_route(current_ned), heading_deg_360(get_pose_yaw_ned(drone)))
            state.mark_reached(index)
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

        projectairsim_log().info("Auto flight complete")
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


async def run_teleport_viewer(
    drone: Drone,
    state: DemoState,
    args,
    world: Optional[World] = None,
    start_auto_flight: bool = False,
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
    auto_flight_task = (
        asyncio.create_task(run_route_auto_flight(drone, state, args, world))
        if start_auto_flight
        else None
    )

    try:
        while not state.snapshot()["stop_requested"]:
            current_ned = get_pose_position_ned(drone)
            current = ned_to_route(current_ned)
            current_heading = heading_deg_360(get_pose_yaw_ned(drone))
            state.update_pose(current, current_heading)
            if not state.snapshot()["auto_flight_running"]:
                state.set_route_index(nearest_route_index(state.route, current))

            now = time.time()
            if now - last_diagnostics_at >= 0.5:
                update_drone_diagnostics(drone, state, world, args.drone_name)
                last_diagnostics_at = now

            if auto_flight_task is not None and auto_flight_task.done():
                await auto_flight_task
                auto_flight_task = None

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


async def run_demo(args):
    route = route_from_constants()
    state = None
    temp_config_dir = None
    drone = None
    preview = None
    client = ProjectAirSimClient(
        address=args.server_ip,
        port_topics=args.topics_port,
        port_services=args.services_port,
    )

    try:
        mode_flight = "auto-diagnostic" if args.auto_diagnostic else args.mode_flight
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
            route = plan_obstacle_aware_route(world, args, route)

        state = DemoState(route)
        preview = OffshorePreview(
            state,
            args.window_name,
            args.preview_width,
            args.preview_height,
            args.pip_scale,
            args.preview_fps,
        )
        preview.start()
        client.subscribe(fpv_topic, preview.receive_fpv)
        client.subscribe(chase_topic, preview.receive_chase)
        client.subscribe(
            drone.robot_info["actual_pose"],
            lambda _, pose_msg: update_pose_topic_diagnostic(state, pose_msg),
        )
        projectairsim_log().info("Preview opened. Press e/r/t for teleport controls, q/esc quits.")

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
        else:
            state.set_mode("teleport", "manual teleport mode")
            await run_teleport_viewer(drone, state, args, world)
    finally:
        if state is not None:
            state.request_stop()
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
        choices=["teleport", "auto-flight", "auto-diagnostic"],
        default="teleport",
        help="Startup mode: manual teleport viewer, immediate PX4 route flight, or teleport diagnostics.",
    )
    parser.set_defaults(replan_on_object=True)
    parser.add_argument(
        "--replan-on-object",
        dest="replan_on_object",
        action="store_true",
        help="Before auto flight, scan route legs and add obstacle bypasses.",
    )
    parser.add_argument(
        "--no-replan-on-object",
        dest="replan_on_object",
        action="store_false",
        help="Disable obstacle-aware route expansion before auto flight.",
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
        default=3.0,
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
        default=4.0,
        help="Distance over which velocity route mode eases down near each waypoint.",
    )
    parser.add_argument(
        "--path-yaw-rate-dps",
        type=float,
        default=10.0,
        help="Maximum yaw rate used while facing the planned route segment.",
    )
    parser.add_argument(
        "--path-yaw-deadband-deg",
        type=float,
        default=5.0,
        help="Yaw error ignored while facing the planned route segment.",
    )
    parser.add_argument(
        "--path-yaw-response-sec",
        type=float,
        default=1.5,
        help="Seconds over which route-facing yaw tries to close heading error.",
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
        default=45.0,
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
        help="Keep PX4 yaw control from turning toward the travel direction.",
    )
    parser.add_argument("--fpv-camera", default="FrontCamera")
    parser.add_argument("--chase-camera", default="Chase")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov-degrees", type=float, default=90.0)
    parser.add_argument("--camera-capture-interval-sec", type=float, default=0.03)
    parser.add_argument("--window-name", default="Offshore PX4 Demo")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=720)
    parser.add_argument("--preview-fps", type=float, default=30.0)
    parser.add_argument("--pip-scale", type=float, default=0.30)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    asyncio.run(run_demo(parsed_args))
